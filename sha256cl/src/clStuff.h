#pragma once

#include "benchmarkLogger.h"
#include <CL/cl.h>
#include <string>

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
std::string ClErrorCodesToString(cl_int clError);

// funkcija paredzēta OpenCL kodolu failu atvēršanai un satura (pirmkoda) iegūšanai
std::string readKernelFile(const std::string &fileName);

class ClStuffContainer
{
  private:
	BenchmarkLogger &logger;

  public:
	cl_int clResult; // paredzēts openCL funkciju izsaukumu rezultātu saglabāšanai un pārbaudei
	cl_platform_id platform;
	cl_device_id device;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_context context;
	cl_command_queue queue;

	ClStuffContainer(BenchmarkLogger &logger) : logger(logger)
	{
		clResult = clGetPlatformIDs(1, &platform, &numPlatforms);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		clResult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

		queue = clCreateCommandQueueWithProperties(context, device, properties, &clResult);
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

		auto start = std::chrono::steady_clock::now();

		cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCstring, &kernelSize, &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		clResult = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &clResult);
		ASSERT(clResult == CL_SUCCESS, ClErrorCodesToString(clResult));

		auto end = std::chrono::steady_clock::now();

		logger.chronoLog("kernel compile time", start, end);

		return kernel;
	}
};
