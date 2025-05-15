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

int hashCheck_v2_with_pinned_memory(ClStuffContainer &clStuffContainer, const std::string &pwFileName,
									std::vector<cl_uint> &hash, std::string &foundPw, BenchmarkLogger &logger)
{
	// sha256 hash vērtībai jābūt 256 biti / 32 baiti
	assert(hash.size() * sizeof(cl_uint) == 32);

	cl_int clResult;

	const size_t batchSize = 1 << 20;

	std::ifstream file(pwFileName, std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + pwFileName);
	}

	auto hostPinnedMemStart = std::chrono::steady_clock::now();

	cl_mem pinnedPasswordsHost = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
												batchSize * 16 * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem pinnedOffsetsHost = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
											  batchSize * sizeof(cl_uint), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_uchar *batchedKernelPasswords =
		(cl_uchar *)clEnqueueMapBuffer(clStuffContainer.queue, pinnedPasswordsHost, CL_TRUE, CL_MAP_WRITE, 0,
									   batchSize * 16 * sizeof(cl_uchar), 0, nullptr, nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_uint *batchedOffsets =
		(cl_uint *)clEnqueueMapBuffer(clStuffContainer.queue, pinnedOffsetsHost, CL_TRUE, CL_MAP_WRITE, 0,
									  batchSize * sizeof(cl_uint), 0, nullptr, nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	auto hostPinnedMemEnd = std::chrono::steady_clock::now();

	logger.chronoLog("host side pinned mem buffer creation time", hostPinnedMemStart, hostPinnedMemEnd);

	std::string line;
	size_t lineIdx = 0;
	cl_int crackedIdx = -1;

	cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/sha256.cl", "sha256_crack");

	size_t kernelWorkGroupSize;
	clGetKernelWorkGroupInfo(kernel, clStuffContainer.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
							 &kernelWorkGroupSize, nullptr);

	auto deviceFixedBuffersAndDataStart = std::chrono::steady_clock::now();

	cl_mem targetHashBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											 hash.size() * sizeof(cl_uint), hash.data(), &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem passwordsBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY,
											batchSize * 16 * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem offsetsBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY, batchSize * sizeof(cl_uint), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem crackedIdxBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	auto deviceFixedBuffersAndDataEnd = std::chrono::steady_clock::now();

	logger.chronoLog("device side fixed buffers and data time", deviceFixedBuffersAndDataStart,
					 deviceFixedBuffersAndDataEnd);

	while (file)
	{
		auto pwBatchStart = std::chrono::steady_clock::now();

		size_t passwordsSize = 0;
		uint currentOffset = 0;
		size_t i = 0;

		for (; i < batchSize && std::getline(file, line); i++, lineIdx++)
		{
			std::memcpy(batchedKernelPasswords + passwordsSize, line.data(), line.size());
			batchedOffsets[i] = currentOffset;

			passwordsSize += line.size();
			currentOffset += line.size();
		}

		auto pwBatchEnd = std::chrono::steady_clock::now();

		logger.chronoLog("pw batch loaded from file and processed", pwBatchStart, pwBatchEnd);

		auto bufferCreationStart = std::chrono::steady_clock::now();

		clEnqueueUnmapMemObject(clStuffContainer.queue, pinnedPasswordsHost, batchedKernelPasswords, 0, nullptr,
								nullptr);
		clEnqueueUnmapMemObject(clStuffContainer.queue, pinnedOffsetsHost, batchedOffsets, 0, nullptr, nullptr);

		clEnqueueCopyBuffer(clStuffContainer.queue, pinnedPasswordsHost, passwordsBuffer, 0, 0,
							passwordsSize * sizeof(cl_uchar), 0, nullptr, nullptr);
		clEnqueueCopyBuffer(clStuffContainer.queue, pinnedOffsetsHost, offsetsBuffer, 0, 0, i * sizeof(cl_uint), 0,
							nullptr, nullptr);

		crackedIdx = -1;
		clEnqueueWriteBuffer(clStuffContainer.queue, crackedIdxBuffer, CL_TRUE, 0, sizeof(cl_int), &crackedIdx, 0,
							 nullptr, nullptr);

		batchedKernelPasswords =
			(cl_uchar *)clEnqueueMapBuffer(clStuffContainer.queue, pinnedPasswordsHost, CL_TRUE, CL_MAP_WRITE, 0,
										   batchSize * 16 * sizeof(cl_uchar), 0, nullptr, nullptr, &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		batchedOffsets = (cl_uint *)clEnqueueMapBuffer(clStuffContainer.queue, pinnedOffsetsHost, CL_TRUE, CL_MAP_WRITE,
													   0, batchSize * sizeof(cl_uint), 0, nullptr, nullptr, &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		auto bufferCreationEnd = std::chrono::steady_clock::now();

		logger.chronoLog("kernel buffer creation time", bufferCreationStart, bufferCreationEnd);

		cl_uint N = i;
		cl_uint charCount = passwordsSize;

		clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &passwordsBuffer);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &offsetsBuffer);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 2, sizeof(cl_uint), &N);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 3, sizeof(cl_uint), &charCount);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 4, sizeof(cl_mem), &targetHashBuffer);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 5, sizeof(cl_mem), &crackedIdxBuffer);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		cl_event profilingEvent;

		size_t localSize = kernelWorkGroupSize;
		size_t globalSize = ((N + localSize - 1) / localSize) * localSize;

		clResult = clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 1, nullptr, &globalSize, &localSize, 0,
										  nullptr, &profilingEvent);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		cl_ulong start;
		cl_ulong end;

		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_COMPLETE, sizeof(end), &end, nullptr);

		double kernelExecTime = static_cast<double>(end - start);

		logger.log("kernel exec time", kernelExecTime / 1e6); // 1e6, lai dabūtu rezultātu milisekundēs

		clReleaseEvent(profilingEvent);

		clResult = clEnqueueReadBuffer(clStuffContainer.queue, crackedIdxBuffer, CL_TRUE, 0, sizeof(int), &crackedIdx,
									   0, nullptr, nullptr);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		if (crackedIdx != -1)
		{

			cl_uint pwStart = batchedOffsets[crackedIdx];
			size_t pwSize;

			if (static_cast<cl_uint>(crackedIdx) < N - 1)
			{
				pwSize = batchedOffsets[crackedIdx + 1] - pwStart;
			}
			else
			{
				pwSize = charCount - pwStart;
			}

			foundPw = std::string(reinterpret_cast<const char *>(&batchedKernelPasswords[pwStart]), pwSize);

			crackedIdx += (lineIdx - i); // indekss ir relatīvs batcham, tāpēc vajag offsetu pieskaitīt

			clReleaseMemObject(pinnedPasswordsHost);
			clReleaseMemObject(pinnedOffsetsHost);
			clReleaseMemObject(passwordsBuffer);
			clReleaseMemObject(offsetsBuffer);
			clReleaseMemObject(targetHashBuffer);
			clReleaseMemObject(crackedIdxBuffer);
			clReleaseKernel(kernel);

			return crackedIdx;
		}
	}

	file.close();

	clReleaseMemObject(pinnedPasswordsHost);
	clReleaseMemObject(pinnedOffsetsHost);
	clReleaseMemObject(passwordsBuffer);
	clReleaseMemObject(offsetsBuffer);
	clReleaseMemObject(targetHashBuffer);
	clReleaseMemObject(crackedIdxBuffer);
	clReleaseKernel(kernel);

	return -1;
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

		std::cout << "Starting search...\n";

		int crackedIdx = hashCheck_v2_with_pinned_memory(clStuffContainer, inputFileName, hash, foundPw, logger);

		auto hashCheckEnd = std::chrono::steady_clock::now();

		logger.chronoLog("hash check time", hashCheckStart, hashCheckEnd);

		if (crackedIdx == -1)
		{

			std::cout << "No matching password found." << "\n";
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
