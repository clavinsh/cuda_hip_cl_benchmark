#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// makro assertam ar ziņojumu
// ņemts no: https://stackoverflow.com/questions/3767869/adding-message-to-assert
#ifndef NDEBUG
#define ASSERT(condition, message)                                                                                     \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(condition))                                                                                              \
		{                                                                                                              \
			std::cerr << "Assertion `" #condition "` failed in " << __FILE__ << " line " << __LINE__ << ": "           \
					  << message << std::endl;                                                                         \
			std::terminate();                                                                                          \
		}                                                                                                              \
	} while (false)
#else
#define ASSERT(condition, message)                                                                                     \
	do                                                                                                                 \
	{                                                                                                                  \
	} while (false)
#endif

// metode OpenCL kļūdu kodu pārveidei uz tekstu, iedvesmojoties no hashcat val2cstr_cl
// https://github.com/hashcat/hashcat/blob/master/src/ext_OpenCL.c
std::string ClErrorCodesToString(cl_int clError)
{
#define CLERR(a)                                                                                                       \
	case a:                                                                                                            \
		return #a

	switch (clError)
	{
		CLERR(CL_SUCCESS);
		CLERR(CL_BUILD_PROGRAM_FAILURE);
		CLERR(CL_COMPILE_PROGRAM_FAILURE);
		CLERR(CL_COMPILER_NOT_AVAILABLE);
		CLERR(CL_DEVICE_NOT_FOUND);
		CLERR(CL_DEVICE_NOT_AVAILABLE);
		CLERR(CL_DEVICE_PARTITION_FAILED);
		CLERR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
		CLERR(CL_IMAGE_FORMAT_MISMATCH);
		CLERR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
		CLERR(CL_INVALID_ARG_INDEX);
		CLERR(CL_INVALID_ARG_SIZE);
		CLERR(CL_INVALID_ARG_VALUE);
		CLERR(CL_INVALID_BINARY);
		CLERR(CL_INVALID_BUFFER_SIZE);
		CLERR(CL_INVALID_BUILD_OPTIONS);
		CLERR(CL_INVALID_COMMAND_QUEUE);
		CLERR(CL_INVALID_COMPILER_OPTIONS);
		CLERR(CL_INVALID_CONTEXT);
		CLERR(CL_INVALID_DEVICE);
		CLERR(CL_INVALID_DEVICE_PARTITION_COUNT);
		CLERR(CL_INVALID_DEVICE_QUEUE);
		CLERR(CL_INVALID_DEVICE_TYPE);
		CLERR(CL_INVALID_EVENT);
		CLERR(CL_INVALID_EVENT_WAIT_LIST);
		CLERR(CL_INVALID_GLOBAL_OFFSET);
		CLERR(CL_INVALID_GLOBAL_WORK_SIZE);
		CLERR(CL_INVALID_HOST_PTR);
		CLERR(CL_INVALID_IMAGE_DESCRIPTOR);
		CLERR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
		CLERR(CL_INVALID_IMAGE_SIZE);
		CLERR(CL_INVALID_KERNEL);
		CLERR(CL_INVALID_KERNEL_ARGS);
		CLERR(CL_INVALID_KERNEL_DEFINITION);
		CLERR(CL_INVALID_KERNEL_NAME);
		CLERR(CL_INVALID_LINKER_OPTIONS);
		CLERR(CL_INVALID_MEM_OBJECT);
		CLERR(CL_INVALID_OPERATION);
		CLERR(CL_INVALID_PIPE_SIZE);
		CLERR(CL_INVALID_PLATFORM);
		CLERR(CL_INVALID_PROGRAM);
		CLERR(CL_INVALID_PROGRAM_EXECUTABLE);
		CLERR(CL_INVALID_PROPERTY);
		CLERR(CL_INVALID_QUEUE_PROPERTIES);
		CLERR(CL_INVALID_SAMPLER);
		CLERR(CL_INVALID_SPEC_ID);
		CLERR(CL_INVALID_VALUE);
		CLERR(CL_INVALID_WORK_DIMENSION);
		CLERR(CL_INVALID_WORK_GROUP_SIZE);
		CLERR(CL_INVALID_WORK_ITEM_SIZE);
		CLERR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
		CLERR(CL_LINK_PROGRAM_FAILURE);
		CLERR(CL_LINKER_NOT_AVAILABLE);
		CLERR(CL_MAP_FAILURE);
		CLERR(CL_MEM_COPY_OVERLAP);
		CLERR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
		CLERR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
		CLERR(CL_OUT_OF_HOST_MEMORY);
		CLERR(CL_OUT_OF_RESOURCES);
		CLERR(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
		CLERR(CL_PROFILING_INFO_NOT_AVAILABLE);
		// CLERR(CL_INVALID_COMMAND_BUFFER_KHR);
		// CLERR(CL_INVALID_SYNC_POINT_WAIT_LIST_KHR);
		// CLERR(CL_INCOMPATIBLE_COMMAND_QUEUE_KHR);
		// CLERR(CL_INVALID_MUTABLE_COMMAND_KHR);
		// CLERR(CL_INVALID_D3D10_DEVICE_KHR);
		// CLERR(CL_INVALID_D3D10_RESOURCE_KHR);
		// CLERR(CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR);
		// CLERR(CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR);
		// CLERR(CL_INVALID_D3D11_DEVICE_KHR);
		// CLERR(CL_INVALID_D3D11_RESOURCE_KHR);
		// CLERR(CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR);
		// CLERR(CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR);
		// CLERR(CL_INVALID_DX9_MEDIA_ADAPTER_KHR);
		// CLERR(CL_INVALID_DX9_MEDIA_SURFACE_KHR);
		// CLERR(CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR);
		// CLERR(CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR);
		// CLERR(CL_EGL_RESOURCE_NOT_ACQUIRED_KHR);
		// CLERR(CL_INVALID_EGL_OBJECT_KHR);
		CLERR(CL_INVALID_GL_OBJECT);
		// CLERR(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR);
		// CLERR(CL_PLATFORM_NOT_FOUND_KHR);
		// CLERR(CL_INVALID_SEMAPHORE_KHR);
		// CLERR(CL_CONTEXT_TERMINATED_KHR);
	}

#undef CLERR

	return "UNKOWN CL ERROR";
}

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
	std::vector<cl_uchar> grid;
	height = 0; // noteiks iteratīvi pēc rindiņu skaita failā, tāpēc sākumā 0

	std::ifstream file(fileName);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}

	std::string line;

	// pēc pirmās rindas nosakām width
	if (std::getline(file, line))
	{
		width = line.size();
		height++;
		processGridFileLine(grid, line);
	}
	else
	{
		throw std::runtime_error(fileName + " file empty?");
	}

	while (std::getline(file, line))
	{
		height++;

		if (line.size() != width)
		{
			throw std::runtime_error("Line width (" + std::to_string(line.size()) +
									 ")"
									 " at line " +
									 std::to_string(height) + " does not match the first line's width (" +
									 std::to_string(width) + ")");
		}

		processGridFileLine(grid, line);
	}

	return grid;
}

// funkcija paredzēta OpenCL kodolu failu atvēršanai un satura (pirmkoda) iegūšanai
std::string readKernelFile(const std::string &fileName)
{
	std::ifstream file(fileName);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open kernel file: " + fileName);
	}

	std::stringstream sourceCodeBuffer;

	sourceCodeBuffer << file.rdbuf();
	return sourceCodeBuffer.str();
}

class ClStuffContainer
{
  public:
	cl_int clResult; // paredzēts openCL funkciju izsaukumu rezultātu saglabāšanai un pārbaudei
	cl_platform_id platform;
	cl_device_id device;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_context context;
	cl_command_queue queue;

	ClStuffContainer()
	{
		clResult = clGetPlatformIDs(1, &platform, &numPlatforms);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		clResult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		queue = clCreateCommandQueueWithProperties(context, device, nullptr, &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	}

	~ClStuffContainer()
	{

		clResult = clReleaseCommandQueue(queue);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		clResult = clReleaseContext(context);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	}

	cl_kernel loadAndCreateKernel(const std::string &fileName, const std::string &kernelName)
	{
		std::string kernelSource = readKernelFile(fileName);

		const char *kernelSourceCstring = kernelSource.c_str();
		size_t kernelSize = kernelSource.length();

		cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCstring, &kernelSize, &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		clResult = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		return kernel;
	}
};

// funkcija, kas sakārto visu kodola izpildei un datu savākšanai
// outputGrid izmēru saucēja fn var nenoteikt, jo šī pati funkcija sakārtos atmiņu
void GameOfLifeStep(ClStuffContainer &clStuffContainer, std::vector<cl_uchar> &grid, std::vector<cl_uchar> &outputGrid,
					cl_ulong width, cl_ulong height, size_t steps)
{
	cl_int clResult;

	size_t gridSize = width * height;

	outputGrid.resize(gridSize);

	cl_mem gridBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									   gridSize * sizeof(cl_uchar), grid.data(), &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem outputGridBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
											 gridSize * sizeof(cl_uchar), outputGrid.data(), &clResult);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/gol.cl", "gol");

	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &width);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
	clResult = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &height);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	cl_mem inputBuffer = gridBuffer;
	cl_mem outputBuffer = outputGridBuffer;

    size_t localSize = 256;
    size_t globalSize = ((gridSize + localSize - 1) / localSize) * localSize;

	for (size_t step = 0; step < steps; step++)
	{
		clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));
		clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);

		clResult =
			clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		std::swap(inputBuffer, outputBuffer);
	}

	clResult = clEnqueueReadBuffer(clStuffContainer.queue, outputGridBuffer, CL_TRUE, 0, gridSize * sizeof(cl_uchar),
								   outputGrid.data(), 0, nullptr, nullptr);
	ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

	clReleaseMemObject(gridBuffer);
	clReleaseMemObject(outputGridBuffer);
}

int main(int argc, char *argv[])
{
	if (argc == 3)
	{
		const std::string inputFileName = argv[1];
		const size_t gameSteps = std::stoll(argv[2]);

		size_t width;
		size_t height;
		std::vector<cl_uchar> grid = loadGridFromFile(inputFileName, width, height);

		std::cout << "Input grid (" << width << "x" << height << "):\n";

		for (size_t h = 0; h < height; h++)
		{
			for (size_t w = 0; w < width; w++)
			{
				std::cout << std::to_string(grid[h * width + w]);
			}
			std::cout << "\n";
		}

		std::vector<cl_uchar> outputGrid;

		ClStuffContainer clStuffContainer;

        size_t maxWorkItems;
        clGetDeviceInfo(clStuffContainer.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkItems, nullptr);

        std::cout << "maxWorkItems: " << maxWorkItems << "\n";

		cl_ulong w = static_cast<cl_ulong>(width);
		cl_ulong h = static_cast<cl_ulong>(height);

		GameOfLifeStep(clStuffContainer, grid, outputGrid, w, h, gameSteps);

		std::cout << "Output grid:\n";

		for (size_t h = 0; h < height; h++)
		{
			for (size_t w = 0; w < width; w++)
			{
				std::cout << std::to_string(outputGrid[h * width + w]);
			}
			std::cout << "\n";
		}
	}
	else
	{
		std::cout << "Correct program usage:\n"
				  << "\t\t" << argv[0] << " <grid file path> <game steps>\n";
	}
	return 0;
}
