#include "clBenchmark.h"
#include "clStuff.h"
#include <CL/cl.h>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// debug vajadzībām
void printCharacterBytes(char c)
{
	std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c) << " ";
	std::cout << std::dec << std::endl;
}

void processGridFileLine(std::vector<uint8_t> &grid, const std::string &line)
{
	for (char c : line)
	{
		if (c == '\n' || c == '\0')
		{
			return;
		}

		if (c != '0' && c != '1')
		{
			std::string s = "Invalid character '";
			s.append(1, c);
			s += "' in grid file";

			throw std::runtime_error(s);
		}

		grid.push_back(c == '0' ? 0 : 1);
	}
}

// izveido flat grid masīvu, automātiski nosakot width, height
// met ārā kļūdas ja nav atbilstošu simbolu (1, 0) vai ja kāda rindiņa nesatur tādu pašu simbolu skaitu kā pirmā
std::vector<cl_uchar> loadGridFromFile(const std::string &fileName, size_t &width, size_t &height)
{
	// ejam uz faila beigām uzreiz ar 'ate', lai noteiktu faila izmēru, pēc tam iesim uz sākumu
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}

	const size_t fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(fileSize);
	file.read(buffer.data(), fileSize);
	file.close();

	std::vector<cl_uchar> grid;
	grid.reserve(fileSize);

	size_t lineStartPos = 0;

	width = 0;
	height = 0;

	for (size_t i = 0; i <= buffer.size(); ++i)
	{
		if (i == buffer.size() || buffer[i] == '\n')
		{
			size_t lineLen = i - lineStartPos;

			if (lineLen == 0)
			{
				lineStartPos = i + 1;
				continue;
			}

			if (width == 0)
			{
				width = lineLen;
			}
			else if (lineLen != width)
			{
				throw std::runtime_error("Invalid line length at line idx: " + std::to_string(height));
			}

			for (size_t j = 0; j < width; ++j)
			{
				cl_uchar val = static_cast<cl_uchar>(buffer[lineStartPos + j] - '0'); // adjust as needed
				grid.push_back(val);
			}

			height++;
			lineStartPos = i + 1;
		}
	}

	return grid;
}

void writeGridToFile(std::vector<cl_uchar> &grid, cl_ulong width, cl_ulong height, std::string fileName)
{
	std::ofstream file(fileName);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}
	std::string line;
	line.reserve(width + 1);
	for (size_t h = 0; h < height; h++)
	{
		line.clear();
		for (size_t w = 0; w < width; w++)
		{
			line += std::to_string(static_cast<int>(grid[h * width + w]));
		}
		line += "\n";
		file.write(line.data(), line.size());
	}
}

void writeGridToBinaryFile(std::vector<cl_uchar> &grid, cl_ulong width, cl_ulong height, std::string fileName)
{
	std::ofstream file(fileName, std::ios::out | std::ios::binary);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}

	// Write dimensions as header (8 bytes each for width and height)
	file.write(reinterpret_cast<const char *>(&width), sizeof(width));
	file.write(reinterpret_cast<const char *>(&height), sizeof(height));

	// Write grid data directly (1 byte per cell)
	file.write(reinterpret_cast<const char *>(grid.data()), grid.size());

	file.close();
}

void writeGridToFile_v2(std::vector<cl_uchar> &grid, cl_ulong width, cl_ulong height, std::string fileName)
{
	std::ofstream file(fileName, std::ios::out | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}

	const size_t bufSize = (width + 1) * height; // +1, jo \n rindu beigās

	std::vector<char> buffer(bufSize);

	for (size_t h = 0; h < height; h++)
	{
		size_t lineStart = h + (width + 1);
		size_t gridRowStart = h * width;

		for (size_t w = 0; w < width; w++)
		{
			buffer[lineStart + w] = '0' + grid[gridRowStart + w]; // ASCII kodu 'haks'
		}

		buffer[lineStart + width] = '\n';
	}

	file.write(buffer.data(), bufSize);
	file.close();
}

// funkcija, kas sakārto visu kodola izpildei un datu savākšanai
// outputGrid izmēru saucēja fn var nenoteikt, jo šī pati funkcija sakārtos atmiņu
void GameOfLifeStep(ClStuffContainer &clStuffContainer, std::vector<cl_uchar> &grid, std::vector<cl_uchar> &outputGrid,
					cl_ulong width, cl_ulong height, size_t steps, BenchmarkLogger &logger)
{
	cl_int clResult;

	size_t gridSize = width * height;

	outputGrid.resize(gridSize);

	auto start = std::chrono::steady_clock::now();

	cl_mem gridBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									   gridSize * sizeof(cl_uchar), grid.data(), &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem outputGridBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
											 gridSize * sizeof(cl_uchar), outputGrid.data(), &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	auto end = std::chrono::steady_clock::now();

	logger.chronoLog("buffer creation time", start, end);

	cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/gol.cl", "gol");

	size_t preferredKernelWorkGroupSize;

	clGetKernelWorkGroupInfo(kernel, clStuffContainer.device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
							 sizeof(size_t), &preferredKernelWorkGroupSize, nullptr);

	size_t kernelWorkGroupSize;

	clGetKernelWorkGroupInfo(kernel, clStuffContainer.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
							 &kernelWorkGroupSize, nullptr);

	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &width);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &height);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem inputBuffer = gridBuffer;
	cl_mem outputBuffer = outputGridBuffer;

	size_t localSize = kernelWorkGroupSize;
	size_t globalSize = ((gridSize + localSize - 1) / localSize) * localSize;

	double totalTime = 0;

	for (size_t step = 0; step < steps; step++)
	{
		clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);

		cl_event profilingEvent;

		clResult = clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 1, nullptr, &globalSize, &localSize, 0,
										  nullptr, &profilingEvent);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clFinish(clStuffContainer.queue);

		cl_ulong start;
		cl_ulong end;

		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_COMPLETE, sizeof(end), &end, nullptr);

		double kernelExecTime = static_cast<double>(end - start);

		logger.log("kernel exec time", kernelExecTime / 1e6); // 1e6, lai dabūtu rezultātu milisekundēs

		totalTime += kernelExecTime;

		std::swap(inputBuffer, outputBuffer);
	}

	logger.log("total exec time", totalTime / 1e6);

	clResult = clEnqueueReadBuffer(clStuffContainer.queue, outputGridBuffer, CL_TRUE, 0, gridSize * sizeof(cl_uchar),
								   outputGrid.data(), 0, nullptr, nullptr);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	clReleaseMemObject(gridBuffer);
	clReleaseMemObject(outputGridBuffer);
}

void GameOfLifeStep_v2(ClStuffContainer &clStuffContainer, std::vector<cl_uchar> &grid,
					   std::vector<cl_uchar> &outputGrid, cl_ulong width, cl_ulong height, size_t steps,
					   BenchmarkLogger &logger)
{
	cl_int clResult;
	const size_t STEPS_PER_KERNEL = 4; // Process multiple steps per kernel invocation

	size_t gridSize = width * height;
	outputGrid.resize(gridSize);

	auto start = std::chrono::steady_clock::now();

	// Create pinned memory buffers for faster host-device transfers
	cl_mem hostPinnedInputBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
												  gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem hostPinnedOutputBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
												   gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	// Map pinned memory to get host accessible pointers
	void *mappedInputPtr = clEnqueueMapBuffer(clStuffContainer.queue, hostPinnedInputBuffer, CL_TRUE, CL_MAP_WRITE, 0,
											  gridSize * sizeof(cl_uchar), 0, nullptr, nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	void *mappedOutputPtr = clEnqueueMapBuffer(clStuffContainer.queue, hostPinnedOutputBuffer, CL_TRUE, CL_MAP_WRITE, 0,
											   gridSize * sizeof(cl_uchar), 0, nullptr, nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	// Copy input data to pinned memory
	std::memcpy(mappedInputPtr, grid.data(), gridSize * sizeof(cl_uchar));

	// Create device buffers (these stay on the device)
	cl_mem deviceInputBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE, gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem deviceOutputBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE, gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem deviceTempBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE, gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	auto end = std::chrono::steady_clock::now();
	logger.chronoLog("buffer creation time", start, end);

	// Create event for timing
	cl_event transferEvent;

	// Start timing the transfer
	start = std::chrono::steady_clock::now();

	// Transfer input data from pinned memory to device
	clResult = clEnqueueWriteBuffer(clStuffContainer.queue, deviceInputBuffer, CL_TRUE, 0, gridSize * sizeof(cl_uchar),
									mappedInputPtr, 0, nullptr, &transferEvent);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	// Wait for transfer to complete
	clWaitForEvents(1, &transferEvent);

	// Get timing info
	cl_ulong transferStart, transferEnd;
	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_START, sizeof(transferStart), &transferStart, nullptr);
	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_END, sizeof(transferEnd), &transferEnd, nullptr);

	double transferTime = static_cast<double>(transferEnd - transferStart) / 1e6; // Convert to ms
	logger.log("host-to-device transfer time", transferTime);

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total host-to-device transfer time", start, end);

	// Load and create kernel for multi-step processing
	cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/gol.cl", "gol_multi_step");

	// Determine optimal work group size
	size_t localSize[2];
	clStuffContainer.getOptimalWorkGroupSize(kernel, localSize);

	// Calculate global work size
	size_t globalSize[2] = {((width + localSize[0] - 1) / localSize[0]) * localSize[0],
							((height + localSize[1] - 1) / localSize[1]) * localSize[1]};

	// Set kernel arguments that don't change
	clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceInputBuffer);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceOutputBuffer);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceTempBuffer);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &width);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 4, sizeof(cl_ulong), &height);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	double totalTime = 0;

	// Process steps in batches
	for (size_t step = 0; step < steps; step += STEPS_PER_KERNEL)
	{
		cl_uchar stepsThisIteration = (cl_uchar)std::min(STEPS_PER_KERNEL, steps - step);

		// Set the number of steps to process
		clResult = clSetKernelArg(kernel, 5, sizeof(cl_uchar), &stepsThisIteration);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		cl_event profilingEvent;

		// Execute kernel
		clResult = clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr,
										  &profilingEvent);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clFinish(clStuffContainer.queue);

		// Measure kernel execution time
		cl_ulong start;
		cl_ulong end;

		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_COMPLETE, sizeof(end), &end, nullptr);

		double kernelExecTime = static_cast<double>(end - start);
		logger.log("batch kernel exec time", kernelExecTime / 1e6);
		totalTime += kernelExecTime;

		// Swap input and output buffers for next iteration if needed
		if (stepsThisIteration % 2 == 0)
		{
			// No swap needed, result is already in inputBuffer
		}
		else
		{
			std::swap(deviceInputBuffer, deviceOutputBuffer);
		}

		// Release the event
		clReleaseEvent(profilingEvent);
	}

	logger.log("total kernel exec time", totalTime / 1e6);

	// Start timing the transfer back
	start = std::chrono::steady_clock::now();

	// Read back final result to pinned memory
	clResult = clEnqueueReadBuffer(clStuffContainer.queue, deviceInputBuffer, CL_TRUE, 0, gridSize * sizeof(cl_uchar),
								   mappedOutputPtr, 0, nullptr, &transferEvent);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	// Wait for transfer to complete
	clWaitForEvents(1, &transferEvent);

	// Get timing info
	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_START, sizeof(transferStart), &transferStart, nullptr);
	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_END, sizeof(transferEnd), &transferEnd, nullptr);

	transferTime = static_cast<double>(transferEnd - transferStart) / 1e6; // Convert to ms
	logger.log("device-to-host transfer time", transferTime);

	// Copy from pinned memory to output vector
	std::memcpy(outputGrid.data(), mappedOutputPtr, gridSize * sizeof(cl_uchar));

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total device-to-host transfer time", start, end);

	// Unmap pinned memory
	clResult =
		clEnqueueUnmapMemObject(clStuffContainer.queue, hostPinnedInputBuffer, mappedInputPtr, 0, nullptr, nullptr);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	clResult =
		clEnqueueUnmapMemObject(clStuffContainer.queue, hostPinnedOutputBuffer, mappedOutputPtr, 0, nullptr, nullptr);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	// Release resources
	clReleaseMemObject(hostPinnedInputBuffer);
	clReleaseMemObject(hostPinnedOutputBuffer);
	clReleaseMemObject(deviceInputBuffer);
	clReleaseMemObject(deviceOutputBuffer);
	clReleaseMemObject(deviceTempBuffer);
	clReleaseKernel(kernel);
	clReleaseEvent(transferEvent);
}

int main(int argc, char *argv[])
{
	if (argc == 5)
	{
		const std::string inputFileName = argv[1];
		const std::string outputFileName = argv[2];
		const size_t gameSteps = std::stoll(argv[3]);
		const std::string logFileName = argv[4];

		BenchmarkLogger logger(logFileName, "OpenCL");

		auto start = std::chrono::steady_clock::now();

		size_t width;
		size_t height;
		std::vector<cl_uchar> grid = loadGridFromFile(inputFileName, width, height);

		auto end = std::chrono::steady_clock::now();

		logger.chronoLog("grid load time", start, end);

		std::vector<cl_uchar> outputGrid;

		auto clInitStart = std::chrono::steady_clock::now();

		ClStuffContainer clStuffContainer(logger);

		auto clInitEnd = std::chrono::steady_clock::now();

		logger.chronoLog("opencl init time", clInitStart, clInitEnd);

		size_t maxWorkItems;
		clGetDeviceInfo(clStuffContainer.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkItems, nullptr);

		cl_ulong w = static_cast<cl_ulong>(width);
		cl_ulong h = static_cast<cl_ulong>(height);

		std::cout << "Processing a " << width << "x" << height << " grid with " << gameSteps << " steps\n";

		auto GoLStart = std::chrono::steady_clock::now();

		GameOfLifeStep(clStuffContainer, grid, outputGrid, w, h, gameSteps, logger);

		auto GoLEnd = std::chrono::steady_clock::now();

		logger.chronoLog("total game of life time", GoLStart, GoLEnd);

		auto writeGridToFileStart = std::chrono::steady_clock::now();

		writeGridToFile(outputGrid, width, height, outputFileName);

		auto writeGridToFileEnd = std::chrono::steady_clock::now();

		logger.chronoLog("write output grid to file time", writeGridToFileStart, writeGridToFileEnd);
	}
	else
	{
		std::cout << "Correct program usage:\n"
				  << "\t\t" << argv[0] << " <grid file path> <output grid file path> <game steps> <log file path>\n";
	}
	return 0;
}
