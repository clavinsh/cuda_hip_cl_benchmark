#include <CL/cl.h>
#include <cassert>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

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

int main()
{

	const int N = 1024;
	const float a = 2.0f;
	std::vector<float> x(N, 7.0f);
	std::vector<float> y(N, 2.0f);

	try
	{
		cl_int clResult; // paredzēts openCL funkciju izsaukumu rezultātu saglabāšanai un pārbaudei
						 //
		cl_platform_id platform;
		cl_device_id device;
		cl_uint numPlatforms;
		cl_uint numDevices;

		clResult = clGetPlatformIDs(1, &platform, &numPlatforms);
		assert(clResult == CL_SUCCESS);
		clResult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
		assert(clResult == CL_SUCCESS);

		cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clResult);
		assert(clResult == CL_SUCCESS);
		cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &clResult);
		assert(clResult == CL_SUCCESS);

		cl_mem xBuffer =
			clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, x.data(), &clResult);
		assert(clResult == CL_SUCCESS);

		cl_mem yBuffer =
			clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, y.data(), &clResult);
		assert(clResult == CL_SUCCESS);

		std::string kernelSource = readKernelFile("saxpy.cl");

		const char *kernelSourceCstring = kernelSource.c_str();
		size_t kernelSize = kernelSource.length();

		cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCstring, &kernelSize, &clResult);
		assert(clResult == CL_SUCCESS);

		clResult = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
		assert(clResult == CL_SUCCESS);

		cl_kernel kernel = clCreateKernel(program, "saxpy", &clResult);
		assert(clResult == CL_SUCCESS);

		clResult = clSetKernelArg(kernel, 0, sizeof(float), &a);
		assert(clResult == CL_SUCCESS);
		clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &xBuffer);
		assert(clResult == CL_SUCCESS);
		clResult = clSetKernelArg(kernel, 2, sizeof(cl_mem), &yBuffer);
		assert(clResult == CL_SUCCESS);
		clResult = clSetKernelArg(kernel, 3, sizeof(int), &N);
		assert(clResult == CL_SUCCESS);

		size_t globalSize = N;
		clResult = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
		assert(clResult == CL_SUCCESS);

        clResult = clEnqueueReadBuffer(queue, yBuffer, CL_TRUE, 0, sizeof(float) * N, y.data(), 0, nullptr, nullptr);
		assert(clResult == CL_SUCCESS);

        for(int i = 0; i < 5; i++)
        {
            std::cout << "y[" << i << "] = " << y[i] << std::endl;
        }

        clResult = clReleaseKernel(kernel);
		assert(clResult == CL_SUCCESS);
        clResult = clReleaseProgram(program);
		assert(clResult == CL_SUCCESS);
        clResult = clReleaseMemObject(xBuffer);
		assert(clResult == CL_SUCCESS);
        clResult = clReleaseMemObject(yBuffer);
		assert(clResult == CL_SUCCESS);
        clResult = clReleaseCommandQueue(queue);
		assert(clResult == CL_SUCCESS);
        clResult = clReleaseContext(context);
		assert(clResult == CL_SUCCESS);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
