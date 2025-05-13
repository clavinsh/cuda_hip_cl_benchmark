// Game Of Life implementācija CUDA vidē

#include "benchmarkLogger.h"
#include <assert.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cuda/std/cstdint> // analogs C/C++ <cstdint>, bet nodrošina fiksētus datu tipu lielumus uz device
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
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

__constant__ size_t d_width;
__constant__ size_t d_height;

// Simplified CUDA kernel more closely matching the OpenCL version
__global__ void golMultiStepKernel(const unsigned char *input, unsigned char *output, unsigned char *temp,
								   size_t stepsToProcess)
{
	// Thread indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if thread is within grid bounds (early exit)
	if (x >= d_width || y >= d_height)
		return;

	const size_t flatIdx = y * d_width + x;

	// Copy input to temp
	temp[flatIdx] = input[flatIdx];

	// Ensure all threads have written to temp
	__syncthreads();

	for (size_t step = 0; step < stepsToProcess; step++)
	{
		// Ping-pong buffers (same as OpenCL version)
		const unsigned char *curr = (step % 2 == 0) ? temp : output;
		unsigned char *next = (step % 2 == 0) ? output : temp;

		// Count neighbors directly from global memory (like OpenCL version)
		int neighbors = 0;

		// Top row
		if (y > 0)
		{
			// Top-left
			if (x > 0)
				neighbors += curr[(y - 1) * d_width + (x - 1)];

			// Top-center
			neighbors += curr[(y - 1) * d_width + x];

			// Top-right
			if (x < d_width - 1)
				neighbors += curr[(y - 1) * d_width + (x + 1)];
		}

		// Middle row (excluding center)
		if (x > 0)
			neighbors += curr[y * d_width + (x - 1)]; // Middle-left

		if (x < d_width - 1)
			neighbors += curr[y * d_width + (x + 1)]; // Middle-right

		// Bottom row
		if (y < d_height - 1)
		{
			// Bottom-left
			if (x > 0)
				neighbors += curr[(y + 1) * d_width + (x - 1)];

			// Bottom-center
			neighbors += curr[(y + 1) * d_width + x];

			// Bottom-right
			if (x < d_width - 1)
				neighbors += curr[(y + 1) * d_width + (x + 1)];
		}

		// Apply Game of Life rules (same logic as OpenCL version)
		unsigned char cell = 0;
		if (curr[flatIdx] == 1)
		{
			if (neighbors == 2 || neighbors == 3)
				cell = 1;
		}
		else
		{
			if (neighbors == 3)
				cell = 1;
		}

		next[flatIdx] = cell;

		// Ensure all threads complete before next iteration
		__syncthreads();
	}

	// If odd number of steps, ensure output has the final state
	if (stepsToProcess % 2 == 1)
	{
		output[flatIdx] = temp[flatIdx];
	}
}

// Main function to run Game of Life simulation with CUDA
void GameOfLifeStep(std::vector<unsigned char> &grid, std::vector<unsigned char> &outputGrid, size_t width,
					size_t height, size_t steps, BenchmarkLogger &logger)
{

	const size_t stepsPerKernel = 1 << 10; // 1024 steps per kernel invocation (same as OpenCL)
	size_t gridSize = width * height;
	outputGrid.resize(gridSize);

	auto start = std::chrono::steady_clock::now();

	// Allocate pinned memory for faster host-device transfers
	unsigned char *hostPinnedInput = nullptr;
	unsigned char *hostPinnedOutput = nullptr;
	CUDA_CHECK(cudaMallocHost(&hostPinnedInput, gridSize * sizeof(unsigned char)));
	CUDA_CHECK(cudaMallocHost(&hostPinnedOutput, gridSize * sizeof(unsigned char)));

	// Copy grid to pinned memory
	std::memcpy(hostPinnedInput, grid.data(), gridSize * sizeof(unsigned char));

	// Copy width and height to constant memory
	CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(size_t)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(size_t)));

	// Allocate device memory
	unsigned char *deviceInput = nullptr;
	unsigned char *deviceOutput = nullptr;
	unsigned char *deviceTemp = nullptr;
	CUDA_CHECK(cudaMalloc(&deviceInput, gridSize * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc(&deviceOutput, gridSize * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc(&deviceTemp, gridSize * sizeof(unsigned char)));

	auto end = std::chrono::steady_clock::now();
	logger.chronoLog("buffer creation time", start, end);

	// Transfer data to device
	start = std::chrono::steady_clock::now();
	cudaEvent_t transferEvent, startEvent, endEvent;
	CUDA_CHECK(cudaEventCreate(&transferEvent));
	CUDA_CHECK(cudaEventCreate(&startEvent));
	CUDA_CHECK(cudaEventCreate(&endEvent));

	CUDA_CHECK(cudaEventRecord(startEvent));
	CUDA_CHECK(cudaMemcpy(deviceInput, hostPinnedInput, gridSize * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaEventRecord(transferEvent));
	CUDA_CHECK(cudaEventSynchronize(transferEvent));

	float transferTime = 0;
	CUDA_CHECK(cudaEventElapsedTime(&transferTime, startEvent, transferEvent));
	logger.log("host-to-device transfer time", transferTime);

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total host-to-device transfer time", start, end);

	// Setup kernel execution parameters
	// Try different block sizes to find what works best for the hardware
	dim3 blockSize(32, 8); // 32x8 for better memory coalescing
	dim3 gridDim((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	double totalTime = 0;

	for (size_t step = 0; step < steps; step += stepsPerKernel)
	{
		size_t stepsThisIteration = std::min(stepsPerKernel, steps - step);

		CUDA_CHECK(cudaEventRecord(startEvent));

		// Launch kernel without shared memory
		golMultiStepKernel<<<gridDim, blockSize>>>(deviceInput, deviceOutput, deviceTemp, stepsThisIteration);

		CUDA_CHECK(cudaEventRecord(endEvent));
		CUDA_CHECK(cudaEventSynchronize(endEvent));

		// Check for kernel execution errors
		CUDA_CHECK(cudaGetLastError());

		float kernelExecTime = 0;
		CUDA_CHECK(cudaEventElapsedTime(&kernelExecTime, startEvent, endEvent));
		logger.log("batch kernel exec time", kernelExecTime);
		totalTime += kernelExecTime;

		// Swap buffers if needed (same logic as OpenCL version)
		if (stepsThisIteration % 2 != 0)
		{
			std::swap(deviceInput, deviceOutput);
		}
	}

	logger.log("total kernel exec time", totalTime);

	// Transfer data back to host
	start = std::chrono::steady_clock::now();

	CUDA_CHECK(cudaEventRecord(startEvent));
	CUDA_CHECK(cudaMemcpy(hostPinnedOutput, deviceInput, gridSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaEventRecord(transferEvent));
	CUDA_CHECK(cudaEventSynchronize(transferEvent));

	float transferBackTime = 0;
	CUDA_CHECK(cudaEventElapsedTime(&transferBackTime, startEvent, transferEvent));
	logger.log("device-to-host transfer time", transferBackTime);

	// Copy the result to output grid
	std::memcpy(outputGrid.data(), hostPinnedOutput, gridSize * sizeof(unsigned char));

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total device-to-host transfer time", start, end);

	// Cleanup
	CUDA_CHECK(cudaEventDestroy(transferEvent));
	CUDA_CHECK(cudaEventDestroy(startEvent));
	CUDA_CHECK(cudaEventDestroy(endEvent));
	CUDA_CHECK(cudaFreeHost(hostPinnedInput));
	CUDA_CHECK(cudaFreeHost(hostPinnedOutput));
	CUDA_CHECK(cudaFree(deviceInput));
	CUDA_CHECK(cudaFree(deviceOutput));
	CUDA_CHECK(cudaFree(deviceTemp));
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
