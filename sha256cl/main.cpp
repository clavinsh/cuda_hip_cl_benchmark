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

int main()
{

	cl_int clResult; // paredzēts openCL funkciju izsaukumu rezultātu saglabāšanai un pārbaudei

	cl_uint numPlatforms;

	clResult = clGetPlatformIDs(0, NULL, &numPlatforms);
	assert(clResult == CL_SUCCESS);

	if (numPlatforms == 0)
	{
		printf("Couldn't find OpenCL capable platforms!");
		return -1;
	}

	cl_platform_id cl_platforms[numPlatforms];

	clResult = clGetPlatformIDs(numPlatforms, cl_platforms, NULL);
	assert(clResult == CL_SUCCESS);

	char str_buffer[1024];

	for (size_t i = 0; i < numPlatforms; i++)
	{
		clResult = clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_VENDOR, sizeof(str_buffer), &str_buffer, NULL);
		assert(clResult == CL_SUCCESS);
		printf("[Platform %zu], %s\n", i, str_buffer);

		cl_uint numDevices = 0;
		clResult = clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		assert(clResult == CL_SUCCESS);
		printf("%d devices available \n", numDevices);

		cl_device_id devices[numDevices];

		clResult = clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		assert(clResult == CL_SUCCESS);

		for (size_t k = 0; k < numDevices; k++)
		{
			clResult = clGetDeviceInfo(devices[k], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
			assert(clResult == CL_SUCCESS);

			printf("Device name: %s\n", str_buffer);
		}
	}

	return 0;
}
