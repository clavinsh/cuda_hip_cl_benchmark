#include "benchmarkLogger.h"
#include "clStuff.h"
#include <CL/cl.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
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

size_t current_pw_size(std::vector<cl_uint> &offsets, cl_uint password_count, cl_uint char_count, cl_uint idx)
{
	// paroļu buferis nesatur \0 simbolus, tāpēc jānosaka paroles garums pēc offsetiem
	if (idx < password_count - 1)
	{
		return offsets[idx + 1] - offsets[idx];
	}
	// pēdējai parolei nav nākamais offsets, tāpēc jāizmanto kopējais simbolu skaits
	else
	{
		return char_count - offsets[idx];
	}
}

int hashCheck_v2(ClStuffContainer &clStuffContainer, const std::string &pwFileName, std::vector<cl_uint> &hash,
				 std::string &foundPw, BenchmarkLogger &logger)
{
	// sha256 hash vērtībai jābūt 256 biti / 32 baiti
	assert(hash.size() * sizeof(cl_uint) == 32);

	cl_int clResult;

	const size_t batchSize = 1 << 18;

	std::ifstream file(pwFileName, std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + pwFileName);
	}

	std::vector<cl_uchar> batchedKernelPasswords(batchSize * 16); // +/- vajadzētu pietikt parasta garuma parolēm
	std::vector<cl_uint> batchedOffsets(batchSize + 1);

	std::string line;
	size_t lineIdx = 0;
	cl_int crackedIdx = -1;

	std::optional<size_t> kernelWorkGroupSize = {};

	while (file)
	{
		auto pwBatchStart = std::chrono::steady_clock::now();

		batchedKernelPasswords.clear();

		int currentOffset = 0;
		size_t i = 0;

		batchedOffsets[i] = currentOffset;

		for (; i < batchSize && std::getline(file, line); i++, lineIdx++)
		{
			batchedKernelPasswords.insert(batchedKernelPasswords.end(), line.begin(), line.end());
			currentOffset += line.size();
			batchedOffsets[i] = currentOffset;
		}

		auto pwBatchEnd = std::chrono::steady_clock::now();

		logger.chronoLog("pw batch loaded from file and processed", pwBatchStart, pwBatchEnd);

		auto bufferCreationStart = std::chrono::steady_clock::now();

		cl_mem passwordsBuffer =
			clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						   batchedKernelPasswords.size() * sizeof(cl_uchar), batchedKernelPasswords.data(), &clResult);
		assert(clResult == CL_SUCCESS);

		cl_mem offsetsBuffer =
			clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						   batchedOffsets.size() * sizeof(cl_uint), batchedOffsets.data(), &clResult);
		assert(clResult == CL_SUCCESS);

		cl_mem targetHashBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
												 hash.size() * sizeof(cl_uint), hash.data(), &clResult);
		assert(clResult == CL_SUCCESS);

		cl_mem crackedIdxBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
												 sizeof(cl_int), &crackedIdx, &clResult);
		assert(clResult == CL_SUCCESS);

		auto bufferCreationEnd = std::chrono::steady_clock::now();

		logger.chronoLog("kernel buffer creation time", bufferCreationStart, bufferCreationEnd);

		cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/sha256.cl", "sha256_crack");

		// lai nav katru loop reizi jānoskaidro šī vērtība,
		// visdrīzāk tā jau pēkšņi nemainīsies, jo kodols un GPU konteksts ir tas pats
		if (!kernelWorkGroupSize.has_value())
		{
			size_t temp;
			clGetKernelWorkGroupInfo(kernel, clStuffContainer.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &temp,
									 nullptr);
			kernelWorkGroupSize = temp;
		}

		cl_uint N = i;
		cl_uint charCount = batchedKernelPasswords.size();

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

		size_t localSize = kernelWorkGroupSize.value();
		size_t globalSize = ((N + localSize - 1) / localSize) * localSize;

		clResult = clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 1, nullptr, &globalSize, &localSize, 0,
										  nullptr, &profilingEvent);
		assert(clResult == CL_SUCCESS);

		cl_ulong start;
		cl_ulong end;

		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_COMPLETE, sizeof(end), &end, nullptr);

		double kernelExecTime = static_cast<double>(end - start);

		logger.log("kernel exec time", kernelExecTime / 1e6); // 1e6, lai dabūtu rezultātu milisekundēs

		clResult = clEnqueueReadBuffer(clStuffContainer.queue, crackedIdxBuffer, CL_TRUE, 0, sizeof(int), &crackedIdx,
									   0, nullptr, nullptr);
		assert(clResult == CL_SUCCESS);

		if (crackedIdx != -1)
		{
			// nez vai smukākais risinājums, bet strādā
			cl_uint pwStart = batchedOffsets[crackedIdx];
			size_t pwSize = current_pw_size(batchedOffsets, i, charCount, crackedIdx);
			foundPw = std::string(reinterpret_cast<const char *>(&batchedKernelPasswords[pwStart]), pwSize);

			crackedIdx += (lineIdx - i); // indekss ir relatīvs batcham, tāpēc vajag offsetu pieskaitīt
			return crackedIdx;
		}
	}

	return -1;
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

		ClStuffContainer clStuffContainer(logger);

		std::vector<cl_uint> hash = hexStringToBytes(hexHash);

		auto hashCheckStart = std::chrono::steady_clock::now();

		std::string foundPw;

		int crackedIdx = hashCheck_v2(clStuffContainer, inputFileName, hash, foundPw, logger);

		auto hashCheckEnd = std::chrono::steady_clock::now();

		logger.chronoLog("total pw cracker kernel time", hashCheckStart, hashCheckEnd);

		if (crackedIdx == -1)
		{
			std::cout << "Password not found!\n";
			return 0;
		}

		std::cout << "Password found index " << crackedIdx << ": " << foundPw << "\n";
		return 0;
	}
	else
	{
		std::cout << "Correct program usage:\n"
				  << "\t\t" << argv[0] << " <passwords file> <password hash> <log file>\n";
		return -1;
	}
}
