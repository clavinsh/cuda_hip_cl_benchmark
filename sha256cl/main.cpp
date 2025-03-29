#include <CL/cl.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

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




int main(int argc, char *argv[])
{
	cl_platform_id platforms[64];
	unsigned int platformCount;

	cl_int platformResult = clGetPlatformIDs(64, platforms, &platformCount);

	assert(platformResult == CL_SUCCESS);

	cl_device_id device = nullptr;

	for (size_t i = 0; i < platformCount; i++)
	{
		cl_device_id devices[64];
		unsigned int deviceCount;
		cl_int deviceResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);

		if (deviceResult == CL_SUCCESS)
		{
			for (size_t j = 0; j < deviceCount; j++)
			{
				char vendorName[256];
				size_t vendorNameLength;
				cl_int deviceInfoResult =
					clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLength);
				if (deviceInfoResult == CL_SUCCESS &&
					std::string(vendorName).substr(0, vendorNameLength) == "NVIDIA CORPORATION")
				{
					device = devices[j];
					break;
				}
			}
		}
	}

	cl_int contextResult;
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &contextResult);

	assert(contextResult == CL_SUCCESS);

	cl_int commandQueueResult;
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &commandQueueResult);

	const char *programSource = "";
	size_t length = 0;
	cl_int programResult;
	cl_program program = clCreateProgramWithSource(context, 1, &programSource, &length, &programResult);
	assert(programResult == CL_SUCCESS);

	cl_int programBuildResult = clBuildProgram(program, 1, &device, "", nullptr, nullptr);

	if (programResult != CL_SUCCESS)
	{
		char log[256];
		size_t logLength;
		cl_int programBuildInfoResult =
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 256, log, &logLength);

		assert(programBuildInfoResult == CL_SUCCESS);
	}

	cl_int kernelResult;
	cl_kernel kernel = clCreateKernel(program, "vector_sum", &kernelResult);

	assert(kernelResult == CL_SUCCESS);

	return 0;
}
