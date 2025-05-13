// Game Of Life implementācija HIP vidē

#include "benchmarkLogger.h"
#include <assert.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <hip/hip_runtime.h>
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
#define HIP_CHECK(ans)                                                                                                  \
	{                                                                                                                   \
		gpuAssert((ans), __FILE__, __LINE__);                                                                           \
	}
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true)
{
	if (code != hipSuccess)
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", hipGetErrorString(code), file, line);
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

inline __device__ int neighborCount(int x, int y, const unsigned char *grid)
{
	int neighbors = 0;

	if (y > 0)
	{
		if (x > 0)
			neighbors += grid[(y - 1) * d_width + (x - 1)];

		neighbors += grid[(y - 1) * d_width + x];

		if (x < d_width - 1)
			neighbors += grid[(y - 1) * d_width + (x + 1)];
	}

	if (x > 0)
		neighbors += grid[y * d_width + (x - 1)];

	if (x < d_width - 1)
		neighbors += grid[y * d_width + (x + 1)];

	if (y < d_height - 1)
	{
		if (x > 0)
			neighbors += grid[(y + 1) * d_width + (x - 1)];

		neighbors += grid[(y + 1) * d_width + x];

		if (x < d_width - 1)
			neighbors += grid[(y + 1) * d_width + (x + 1)];
	}

	return neighbors;
}

__global__ void GoL(unsigned char *input, unsigned char *output)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= d_width || y >= d_height)
		return;

	const size_t flatIdx = y * d_width + x;

	int neighbors = neighborCount(x, y, input);

	unsigned char cell = 0;
	if (input[flatIdx] == 1)
	{
		if (neighbors == 2 || neighbors == 3)
			cell = 1;
	}
	else
	{
		if (neighbors == 3)
			cell = 1;
	}

	output[flatIdx] = cell;
}

void GameOfLifeStep(std::vector<unsigned char> &grid, std::vector<unsigned char> &outputGrid, size_t width,
					size_t height, size_t steps, BenchmarkLogger &logger)
{

	size_t gridSize = width * height;
	outputGrid.resize(gridSize);

	auto start = std::chrono::steady_clock::now();

	unsigned char *hostPinnedInput = nullptr;
	unsigned char *hostPinnedOutput = nullptr;
	HIP_CHECK(hipHostMalloc(&hostPinnedInput, gridSize * sizeof(unsigned char), hipHostMallocDefault));
	HIP_CHECK(hipHostMalloc(&hostPinnedOutput, gridSize * sizeof(unsigned char), hipHostMallocDefault));

	std::memcpy(hostPinnedInput, grid.data(), gridSize * sizeof(unsigned char));

	HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_width), &width, sizeof(size_t)));
	HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_height), &height, sizeof(size_t)));

	unsigned char *deviceInput = nullptr;
	unsigned char *deviceOutput = nullptr;
	HIP_CHECK(hipMalloc(&deviceInput, gridSize * sizeof(unsigned char)));
	HIP_CHECK(hipMalloc(&deviceOutput, gridSize * sizeof(unsigned char)));

	auto end = std::chrono::steady_clock::now();

	logger.chronoLog("buffer creation time", start, end);

	start = std::chrono::steady_clock::now();

	hipEvent_t transferEvent, startEvent, endEvent;
	HIP_CHECK(hipEventCreate(&transferEvent));
	HIP_CHECK(hipEventCreate(&startEvent));
	HIP_CHECK(hipEventCreate(&endEvent));

	HIP_CHECK(hipEventRecord(startEvent));
	HIP_CHECK(hipMemcpy(deviceInput, hostPinnedInput, gridSize * sizeof(unsigned char), hipMemcpyHostToDevice));
	HIP_CHECK(hipEventRecord(transferEvent));
	HIP_CHECK(hipEventSynchronize(transferEvent));

	float transferTime = 0;
	HIP_CHECK(hipEventElapsedTime(&transferTime, startEvent, transferEvent));
	logger.log("host-to-device transfer time", transferTime);

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total host-to-device transfer time", start, end);

	// lokālais bloka izmērs, šis likās diezgan ok
	dim3 blockSize(32, 8);
	dim3 gridDim((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	double totalTime = 0;

	unsigned char *currentInput = deviceInput;
	unsigned char *currentOutput = deviceOutput;

	for (size_t step = 0; step < steps; step++)
	{

		HIP_CHECK(hipEventRecord(startEvent));

		GoL<<<gridDim, blockSize>>>(currentInput, currentOutput);

		HIP_CHECK(hipEventRecord(endEvent));
		HIP_CHECK(hipEventSynchronize(endEvent));

		HIP_CHECK(hipGetLastError());

		float kernelExecTime = 0;
		HIP_CHECK(hipEventElapsedTime(&kernelExecTime, startEvent, endEvent));
		logger.log("kernel exec time", kernelExecTime);
		totalTime += kernelExecTime;

		std::swap(currentInput, currentOutput);
	}

	logger.log("total kernel exec time", totalTime);

	start = std::chrono::steady_clock::now();

	HIP_CHECK(hipEventRecord(startEvent));
	// ņemot vērā pēdējo std::swap ar buferiem, pēdējā izeja atrodas input bufeŗi
	HIP_CHECK(hipMemcpy(hostPinnedOutput, currentInput, gridSize * sizeof(unsigned char), hipMemcpyDeviceToHost));
	HIP_CHECK(hipEventRecord(transferEvent));
	HIP_CHECK(hipEventSynchronize(transferEvent));

	float transferBackTime = 0;
	HIP_CHECK(hipEventElapsedTime(&transferBackTime, startEvent, transferEvent));

	logger.log("device-to-host transfer time", transferBackTime);

	std::memcpy(outputGrid.data(), hostPinnedOutput, gridSize * sizeof(unsigned char));

	end = std::chrono::steady_clock::now();
	logger.chronoLog("total device-to-host transfer time", start, end);

	HIP_CHECK(hipEventDestroy(transferEvent));
	HIP_CHECK(hipEventDestroy(startEvent));
	HIP_CHECK(hipEventDestroy(endEvent));
	HIP_CHECK(hipHostFree(hostPinnedInput));
	HIP_CHECK(hipHostFree(hostPinnedOutput));
	HIP_CHECK(hipFree(deviceInput));
	HIP_CHECK(hipFree(deviceOutput));
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

		HIP_CHECK(hipSetDevice(0));

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
