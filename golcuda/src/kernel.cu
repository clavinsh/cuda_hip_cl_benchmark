// Game Of Life implementācija CUDA vidē

#include "benchmarkLogger.h"
#include <assert.h>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda/std/cstdint> // analogs C/C++ <cstdint>, bet nodrošina fiksētus datu tipu lielumus uz device
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

// macro priekš katra cuda API izsaukuma rezultāta pārbaudes
// ņemts no
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDA_CHECK(ans)                                                                                                \
	{                                                                                                                  \
		gpuAssert((ans), __FILE__, __LINE__);                                                                          \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

// izveido flat grid masīvu, automātiski nosakot width, height
// met ārā kļūdas ja nav atbilstošu simbolu (1, 0) vai ja kāda rindiņa nesatur tādu pašu simbolu skaitu kā pirmā
std::vector<unsigned char> loadGridFromFile(const std::string &fileName, size_t &width, size_t &height)
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

	std::vector<unsigned char> grid;
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
				unsigned char val = static_cast<unsigned char>(buffer[lineStartPos + j] - '0');
				grid.push_back(val);
			}

			height++;
			lineStartPos = i + 1;
		}
	}

	return grid;
}

void writeGridToFile(std::vector<unsigned char> &grid, size_t width, size_t height, std::string fileName)
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
__global__ void gol_step(const unsigned char *input, unsigned char *output, unsigned long long width,
						 unsigned long long height)
{
	// Calculate global position
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if within bounds
	if (x >= width || y >= height)
		return;

	// Calculate flat index
	size_t idx = y * width + x;

	// Count neighbors
	int neighbors = 0;

	// Top row
	if (y > 0)
	{
		// Top-left
		if (x > 0 && input[(y - 1) * width + (x - 1)] == 1)
			neighbors++;
		// Top-center
		if (input[(y - 1) * width + x] == 1)
			neighbors++;
		// Top-right
		if (x + 1 < width && input[(y - 1) * width + (x + 1)] == 1)
			neighbors++;
	}

	// Middle row
	// Left
	if (x > 0 && input[y * width + (x - 1)] == 1)
		neighbors++;
	// Right
	if (x + 1 < width && input[y * width + (x + 1)] == 1)
		neighbors++;

	// Bottom row
	if (y + 1 < height)
	{
		// Bottom-left
		if (x > 0 && input[(y + 1) * width + (x - 1)] == 1)
			neighbors++;
		// Bottom-center
		if (input[(y + 1) * width + x] == 1)
			neighbors++;
		// Bottom-right
		if (x + 1 < width && input[(y + 1) * width + (x + 1)] == 1)
			neighbors++;
	}

	// Apply Conway's Game of Life rules
	if (input[idx] == 1)
	{
		// Cell is alive
		output[idx] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
	}
	else
	{
		// Cell is dead
		output[idx] = (neighbors == 3) ? 1 : 0;
	}
}

void GameOfLifeStep(std::vector<unsigned char> &grid, std::vector<unsigned char> &outputGrid, unsigned long long width,
					unsigned long long height, size_t steps, BenchmarkLogger &logger)
{
	size_t gridSize = width * height;
	outputGrid.resize(gridSize);

	// Create CUDA events for timing
	cudaEvent_t start_event, stop_event, kernel_start, kernel_stop;
	CUDA_CHECK(cudaEventCreate(&start_event));
	CUDA_CHECK(cudaEventCreate(&stop_event));
	CUDA_CHECK(cudaEventCreate(&kernel_start));
	CUDA_CHECK(cudaEventCreate(&kernel_stop));
	float milliseconds = 0;

	// Create CUDA streams for overlapping operations
	cudaStream_t stream1, stream2;
	CUDA_CHECK(cudaStreamCreate(&stream1));
	CUDA_CHECK(cudaStreamCreate(&stream2));

	// Start timing memory allocation
	auto start = std::chrono::steady_clock::now();

	// Use pinned memory for faster transfers
	unsigned char *h_pinnedInput = nullptr;
	unsigned char *h_pinnedOutput = nullptr;
	CUDA_CHECK(cudaMallocHost(&h_pinnedInput, gridSize * sizeof(unsigned char)));
	CUDA_CHECK(cudaMallocHost(&h_pinnedOutput, gridSize * sizeof(unsigned char)));

	// Copy data to pinned memory
	std::memcpy(h_pinnedInput, grid.data(), gridSize * sizeof(unsigned char));

	// Allocate device memory
	unsigned char *d_input = nullptr;
	unsigned char *d_output = nullptr;
	CUDA_CHECK(cudaMalloc(&d_input, gridSize * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc(&d_output, gridSize * sizeof(unsigned char)));

	auto end = std::chrono::steady_clock::now();
	logger.chronoLog("buffer creation time", start, end);

	// Determine block size
	dim3 blockSize(16, 16); // Default block size
	dim3 gridDim((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	// Record transfer start time
	CUDA_CHECK(cudaEventRecord(start_event, stream1));

	// Asynchronously copy input data to device
	CUDA_CHECK(
		cudaMemcpyAsync(d_input, h_pinnedInput, gridSize * sizeof(unsigned char), cudaMemcpyHostToDevice, stream1));

	// Record transfer end time
	CUDA_CHECK(cudaEventRecord(stop_event, stream1));
	CUDA_CHECK(cudaEventSynchronize(stop_event));

	CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
	logger.log("host-to-device transfer time", milliseconds);

	// Record kernel execution start time
	CUDA_CHECK(cudaEventRecord(kernel_start, stream1));

	// Process steps using stream1
	for (size_t step = 0; step < steps; step++)
	{
		// Launch kernel for one step in stream1
		gol_step<<<gridDim, blockSize, 0, stream1>>>(d_input, d_output, width, height);

		// Swap buffers for next iteration
		std::swap(d_input, d_output);
	}

	// Make sure the last iteration's result is in d_input
	// (because we swap after each step)
	if (steps % 2 == 1)
	{
		std::swap(d_input, d_output);
	}

	// Record kernel execution end time
	CUDA_CHECK(cudaEventRecord(kernel_stop, stream1));
	CUDA_CHECK(cudaEventSynchronize(kernel_stop));
	CUDA_CHECK(cudaGetLastError()); // Check for any errors in kernel launch

	CUDA_CHECK(cudaEventElapsedTime(&milliseconds, kernel_start, kernel_stop));
	logger.log("total kernel exec time", milliseconds);

	// Start timing transfer back to host
	start = std::chrono::steady_clock::now();

	// Record device-to-host transfer start
	CUDA_CHECK(cudaEventRecord(start_event, stream2));

	// Asynchronously copy results back to pinned memory using stream2
	// This could potentially overlap with any remaining work in stream1
	CUDA_CHECK(
		cudaMemcpyAsync(h_pinnedOutput, d_input, gridSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream2));

	// Record device-to-host transfer end
	CUDA_CHECK(cudaEventRecord(stop_event, stream2));
	CUDA_CHECK(cudaEventSynchronize(stop_event));

	CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
	logger.log("device-to-host transfer time", milliseconds);

	// Copy from pinned memory to final output buffer
	std::memcpy(outputGrid.data(), h_pinnedOutput, gridSize * sizeof(unsigned char));

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total device-to-host transfer time", start, end);

	// Clean up resources
	CUDA_CHECK(cudaFreeHost(h_pinnedInput));
	CUDA_CHECK(cudaFreeHost(h_pinnedOutput));
	CUDA_CHECK(cudaFree(d_input));
	CUDA_CHECK(cudaFree(d_output));
	CUDA_CHECK(cudaEventDestroy(start_event));
	CUDA_CHECK(cudaEventDestroy(stop_event));
	CUDA_CHECK(cudaEventDestroy(kernel_start));
	CUDA_CHECK(cudaEventDestroy(kernel_stop));
	CUDA_CHECK(cudaStreamDestroy(stream1));
	CUDA_CHECK(cudaStreamDestroy(stream2));
}

int main(int argc, char *argv[])
{
	if (argc == 5)
	{
		const std::string inputFileName = argv[1];
		const std::string outputFileName = argv[2];
		const size_t gameSteps = std::stoll(argv[3]);
		const std::string logFileName = argv[4];

		BenchmarkLogger logger(logFileName, "CUDA");

		auto start = std::chrono::steady_clock::now();

		size_t width;
		size_t height;
		std::vector<unsigned char> grid = loadGridFromFile(inputFileName, width, height);

		auto end = std::chrono::steady_clock::now();

		logger.chronoLog("grid load time", start, end);

		std::vector<unsigned char> outputGrid;

		auto cudaInitStart = std::chrono::steady_clock::now();

		CUDA_CHECK(cudaSetDevice(0));

		auto cudaInitEnd = std::chrono::steady_clock::now();

		logger.chronoLog("cuda init time", cudaInitStart, cudaInitEnd);

		unsigned long long w = static_cast<unsigned long long>(width);
		unsigned long long h = static_cast<unsigned long long>(height);

		std::cout << "Processing a " << width << "x" << height << " grid with " << gameSteps << " steps\n";

		auto GoLStart = std::chrono::steady_clock::now();

		GameOfLifeStep(grid, outputGrid, w, h, gameSteps, logger);

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
