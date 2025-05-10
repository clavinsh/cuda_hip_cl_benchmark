#include "benchmarkLogger.h"
#include "clStuff.h"
#include <CL/cl.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<std::string> passwordsfromFile(const std::string &fileName)
{
	std::ifstream file(fileName);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}

	std::vector<std::string> output;

	std::string line;
	while (std::getline(file, line))
	{
		output.emplace_back(line);
	}

	return output;
}

std::vector<cl_uint> hexStringToBytes(const std::string &hexHash)
{
	// 256 biti => 64 hex skaitļi
	if (hexHash.size() != 64)
	{
		throw std::runtime_error("SHA-256 hash as a hex string must be exactly 64 characters!");
	}

	std::vector<cl_uint> hash;

	// Process every 8 hex digits (32 bits) to create a uint32_t
	for (size_t i = 0; i < hexHash.length(); i += 8)
	{
		std::string chunk = hexHash.substr(i, 8);
		std::stringstream ss;
		uint32_t value;

		ss << std::hex << chunk;
		ss >> value;

		hash.push_back(value);
	}

	return hash;
}

std::string bytesToHexString(std::vector<cl_uint> &bytes)
{
	std::stringstream ss;
	ss << std::hex << std::setfill('0');

	for (const cl_uchar byte : bytes)
	{
		ss << std::setw(8) << byte;
	}

	return ss.str();
}

int hashCheck(ClStuffContainer &clStuffContainer, const std::vector<std::string> &passwords, std::vector<cl_uint> &hash,
			  BenchmarkLogger &logger)
{
	// sha256 hash vērtībai jābūt 256 biti / 32 baiti
	assert(hash.size() * sizeof(cl_uint) == 32);

	cl_int clResult;

	std::vector<cl_uchar> kernelPasswords;
	std::vector<cl_uint> offsets;
	cl_int crackedIdx = -1;

	int currentOffset = 0;
	for (const auto &password : passwords)
	{
		offsets.push_back(currentOffset);
		kernelPasswords.insert(kernelPasswords.end(), password.begin(), password.end());
		currentOffset += password.size();
	}

	auto bufferCreationStart = std::chrono::steady_clock::now();

	cl_mem passwordsBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   kernelPasswords.size() * sizeof(cl_uchar), kernelPasswords.data(), &clResult);
	assert(clResult == CL_SUCCESS);

	cl_mem offsetsBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										  offsets.size() * sizeof(cl_uint), offsets.data(), &clResult);
	assert(clResult == CL_SUCCESS);

	cl_mem targetHashBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											 hash.size() * sizeof(cl_uint), hash.data(), &clResult);
	assert(clResult == CL_SUCCESS);

	cl_mem crackedIdxBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
											 sizeof(cl_int), &crackedIdx, &clResult);
	assert(clResult == CL_SUCCESS);

	auto bufferCreationEnd = std::chrono::steady_clock::now();

	logger.chronoLog("buffer creation time", bufferCreationStart, bufferCreationEnd);

	cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/sha256.cl", "sha256_crack");

	cl_uint N = passwords.size();
	cl_uint charCount = kernelPasswords.size();

	clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &passwordsBuffer);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &offsetsBuffer);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 2, sizeof(cl_uint), &N);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 3, sizeof(cl_uint), &charCount);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 4, sizeof(cl_mem), &targetHashBuffer);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 5, sizeof(cl_mem), &crackedIdxBuffer);
	assert(clResult == CL_SUCCESS);

	cl_event profilingEvent;

	size_t kernelWorkGroupSize;
	clGetKernelWorkGroupInfo(kernel, clStuffContainer.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
							 &kernelWorkGroupSize, nullptr);

	size_t localSize = kernelWorkGroupSize;
	size_t globalSize = ((N + localSize - 1) / localSize) * localSize;

	clResult = clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr,
									  &profilingEvent);
	assert(clResult == CL_SUCCESS);

	cl_ulong start;
	cl_ulong end;

	clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
	clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_COMPLETE, sizeof(end), &end, nullptr);

	double kernelExecTime = static_cast<double>(end - start);

	logger.log("kernel exec time", kernelExecTime / 1e6); // 1e6, lai dabūtu rezultātu milisekundēs

	clResult = clEnqueueReadBuffer(clStuffContainer.queue, crackedIdxBuffer, CL_TRUE, 0, sizeof(int), &crackedIdx, 0,
								   nullptr, nullptr);
	assert(clResult == CL_SUCCESS);

	return crackedIdx;
}

int main(int argc, char *argv[])
{
	if (argc == 4)
	{
		const std::string inputFileName = argv[1];
		const std::string hexHash = argv[2];
		const std::string logFileName = argv[3];

		BenchmarkLogger logger(logFileName, "OpenCL");

		auto pwFileStart = std::chrono::steady_clock::now();

		std::vector<std::string> passwords = passwordsfromFile(inputFileName);
		std::vector<cl_uint> hash = hexStringToBytes(hexHash);

		auto pwFileEnd = std::chrono::steady_clock::now();

		logger.chronoLog("password file & hash processed", pwFileStart, pwFileEnd);

		ClStuffContainer clStuffContainer(logger);

		auto hashCheckStart = std::chrono::steady_clock::now();

		int crackedIdx = hashCheck(clStuffContainer, passwords, hash, logger);

		auto hashCheckEnd = std::chrono::steady_clock::now();

		logger.chronoLog("total pw cracker kernel time", hashCheckStart, hashCheckEnd);

		if (crackedIdx == -1)
		{
			std::cout << "Password not found!\n";
			return 0;
		}
		else if ((size_t)crackedIdx >= passwords.size())
		{
			std::cout << "Password out of bounds!\n"
					  << "Given found index " << crackedIdx << " >= " << passwords.size() << "\n";
			return -1;
		}

		std::cout << "Password found at index " << crackedIdx << ": " << passwords[crackedIdx] << "\n";
		return 0;
	}
	else
	{
		std::cout << "Correct program usage:\n"
				  << "\t\t" << argv[0] << " <passwords file> <password hash> <log file>\n";
		return -1;
	}
}
