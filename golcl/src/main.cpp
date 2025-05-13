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
				cl_uchar val = static_cast<cl_uchar>(buffer[lineStartPos + j] - '0');
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
	std::ofstream file(fileName, std::ios::out | std::ios::binary);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}

	const size_t totalSize = (width + 1) * height; // +1, jo rindas beigās ir \n

	std::vector<char> buffer(totalSize);

	for (size_t h = 0; h < height; h++)
	{
		size_t lineStart = h * (width + 1);
		size_t gridRowStart = h * width;

		for (size_t w = 0; w < width; w++)
		{
			buffer[lineStart + w] = '0' + grid[gridRowStart + w];
		}

		buffer[lineStart + width] = '\n';
	}

	file.write(buffer.data(), totalSize);

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

	cl_mem hostPinnedInputBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
												  gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem hostPinnedOutputBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
												   gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	void *mappedInputPtr = clEnqueueMapBuffer(clStuffContainer.queue, hostPinnedInputBuffer, CL_TRUE, CL_MAP_WRITE, 0,
											  gridSize * sizeof(cl_uchar), 0, nullptr, nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	void *mappedOutputPtr = clEnqueueMapBuffer(clStuffContainer.queue, hostPinnedOutputBuffer, CL_TRUE, CL_MAP_WRITE, 0,
											   gridSize * sizeof(cl_uchar), 0, nullptr, nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	std::memcpy(mappedInputPtr, grid.data(), gridSize * sizeof(cl_uchar));

	cl_mem deviceInputBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE, gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem deviceOutputBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE, gridSize * sizeof(cl_uchar), nullptr, &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	auto end = std::chrono::steady_clock::now();
	logger.chronoLog("buffer creation time", start, end);

	cl_event transferEvent;

	start = std::chrono::steady_clock::now();

	clResult = clEnqueueWriteBuffer(clStuffContainer.queue, deviceInputBuffer, CL_TRUE, 0, gridSize * sizeof(cl_uchar),
									mappedInputPtr, 0, nullptr, &transferEvent);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	clWaitForEvents(1, &transferEvent);

	cl_ulong transferStart, transferEnd;
	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_START, sizeof(transferStart), &transferStart, nullptr);
	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_END, sizeof(transferEnd), &transferEnd, nullptr);

	double transferTime = static_cast<double>(transferEnd - transferStart) / 1e6;

	logger.log("host-to-device transfer time", transferTime);

	end = std::chrono::steady_clock::now();

	logger.chronoLog("total host-to-device transfer time", start, end);

	cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/gol.cl", "gol");

	size_t localSize[2];
	clStuffContainer.getOptimalWorkGroupSize(kernel, localSize);

	size_t globalSize[2] = {((width + localSize[0] - 1) / localSize[0]) * localSize[0],
							((height + localSize[1] - 1) / localSize[1]) * localSize[1]};

	double totalTime = 0;

	cl_event profilingEvent;

	cl_mem currentInput = deviceInputBuffer;
	cl_mem currentOutput = deviceOutputBuffer;

	clResult = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &width);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &height);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	for (size_t step = 0; step < steps; step++)
	{

		clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &currentInput);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &currentOutput);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		clResult = clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr,
										  &profilingEvent);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clFinish(clStuffContainer.queue);

		cl_ulong start;
		cl_ulong end;

		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
		clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_COMPLETE, sizeof(end), &end, nullptr);

		double kernelExecTime = static_cast<double>(end - start);
		logger.log("batch kernel exec time", kernelExecTime / 1e6);
		totalTime += kernelExecTime;

		std::swap(currentInput, currentOutput);
	}

	clReleaseEvent(profilingEvent);

	logger.log("total kernel exec time", totalTime / 1e6);

	start = std::chrono::steady_clock::now();

	clResult = clEnqueueReadBuffer(clStuffContainer.queue, currentInput, CL_TRUE, 0, gridSize * sizeof(cl_uchar),
								   mappedOutputPtr, 0, nullptr, &transferEvent);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	clWaitForEvents(1, &transferEvent);

	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_START, sizeof(transferStart), &transferStart, nullptr);
	clGetEventProfilingInfo(transferEvent, CL_PROFILING_COMMAND_END, sizeof(transferEnd), &transferEnd, nullptr);

	transferTime = static_cast<double>(transferEnd - transferStart) / 1e6;
	logger.log("device-to-host transfer time", transferTime);

	std::memcpy(outputGrid.data(), mappedOutputPtr, gridSize * sizeof(cl_uchar));

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total device-to-host transfer time", start, end);

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
