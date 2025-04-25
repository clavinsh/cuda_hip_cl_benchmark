#include <CL/cl.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
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

std::vector<std::string> passwordsfromFile(const std::string &fileName)
{
	std::ifstream file(fileName);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + fileName);
	}

	std::vector<std::string> output;

	std::string line;
	while (std::getline(file, line))
	{
		output.emplace_back(line);
	}

	return output;
}

// bool testSha(const std::string &password, const std::string hexExpectedHash)
// {
// 	return false;
// }

static uint8_t parseHexByte(const std::string &hexHash, int offset)
{
	std::string hexByteString = hexHash.substr(offset, 2);
	return static_cast<uint8_t>(std::stoi(hexByteString, nullptr, 16));
}

std::vector<uint8_t> hexStringToBytes(const std::string &hexHash)
{
	// 256 biti => 64 hex skaitļi
	if (hexHash.size() != 64)
	{
		throw std::runtime_error("SHA-256 hash as a hex string must be exactly 64 characters!");
	}

	std::vector<uint8_t> hash(32);

	for (int i = 0; i < 32; i++)
	{

		hash[i] = parseHexByte(hexHash, i * 2);
	}

	return hash;
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
		assert(clResult == CL_SUCCESS);
		clResult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
		assert(clResult == CL_SUCCESS);
		context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clResult);
		assert(clResult == CL_SUCCESS);
		queue = clCreateCommandQueueWithProperties(context, device, nullptr, &clResult);
		assert(clResult == CL_SUCCESS);
	}

	~ClStuffContainer()
	{

		clResult = clReleaseCommandQueue(queue);
		assert(clResult == CL_SUCCESS);
		clResult = clReleaseContext(context);
		assert(clResult == CL_SUCCESS);
	}

	cl_kernel loadAndCreateKernel(const std::string &fileName, const std::string &kernelName)
	{
		std::string kernelSource = readKernelFile(fileName);

		const char *kernelSourceCstring = kernelSource.c_str();
		size_t kernelSize = kernelSource.length();

		cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCstring, &kernelSize, &clResult);
		assert(clResult == CL_SUCCESS);

		clResult = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
		assert(clResult == CL_SUCCESS);

		cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &clResult);
		assert(clResult == CL_SUCCESS);
		return kernel;
	}
};

int hashCheck(ClStuffContainer &clStuffContainer, const std::vector<std::string> &passwords, std::vector<uint8_t> &hash)
{
	cl_int clResult;

	std::vector<char> kernelPasswords;
	std::vector<uint32_t> offsets;
	int crackedIdx = -1;

	int currentOffset = 0;
	for (const auto &password : passwords)
	{
		offsets.push_back(currentOffset);
		kernelPasswords.insert(kernelPasswords.end(), password.begin(), password.end());
		currentOffset += password.size();
	}

	cl_mem passwordsBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											kernelPasswords.size() * sizeof(char), kernelPasswords.data(), &clResult);
	assert(clResult == CL_SUCCESS);

	cl_mem offsetsBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										  offsets.size() * sizeof(size_t), offsets.data(), &clResult);
	assert(clResult == CL_SUCCESS);

	cl_mem targetHashBuffer = clCreateBuffer(clStuffContainer.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											 hash.size() * sizeof(uint8_t), hash.data(), &clResult);
	assert(clResult == CL_SUCCESS);

	cl_mem crackedIdxBuffer =
		clCreateBuffer(clStuffContainer.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int),
					   &crackedIdx, &clResult);

	assert(clResult == CL_SUCCESS);

	cl_kernel kernel = clStuffContainer.loadAndCreateKernel("kernels/sha256.cl", "sha256_crack");

	size_t N = passwords.size();
    size_t charCount = kernelPasswords.size();

	clResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &passwordsBuffer);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &offsetsBuffer);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 2, sizeof(uint), &N);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 3, sizeof(uint), &charCount);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 4, sizeof(cl_mem), &targetHashBuffer);
	assert(clResult == CL_SUCCESS);
	clResult = clSetKernelArg(kernel, 5, sizeof(cl_mem), &crackedIdxBuffer);
	assert(clResult == CL_SUCCESS);

	size_t globalSize = N;
	clResult =
		clEnqueueNDRangeKernel(clStuffContainer.queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
	assert(clResult == CL_SUCCESS);

	clResult = clEnqueueReadBuffer(clStuffContainer.queue, crackedIdxBuffer, CL_TRUE, 0, sizeof(int), &crackedIdx, 0,
								   nullptr, nullptr);
	assert(clResult == CL_SUCCESS);

	return crackedIdx;
}

int main(int argc, char *argv[])
{
	if (argc == 3)
	{
		const std::string inputFileName = argv[1];
		const std::string hexHash = argv[2];

		std::vector<std::string> passwords = passwordsfromFile(inputFileName);
		std::vector<uint8_t> hash = hexStringToBytes(hexHash);

		ClStuffContainer clStuffContainer;

		int crackedIdx = hashCheck(clStuffContainer, passwords, hash);

		if (crackedIdx == -1)
		{
			std::cout << "Password not found!\n";
            return 0;
		}
		else if ((size_t)crackedIdx >= passwords.size())
		{
			std::cout << "Password out of bounds!\n"
					  << "Given cracked idx " << crackedIdx << " >= " << passwords.size() << "\n";
            return -1;
		}

		std::cout << "Password cracked at index " << crackedIdx << ": " << passwords[crackedIdx] << "\n";
        return 0;
	}
	else
	{
		std::cout << "Correct program usage:\n"
				  << "\tRunning tests:\n"
				  << "\t\t" << argv[0] << " --test\n"
				  << "\tPassword cracking:\n"
				  << "\t\t" << argv[0] << " <passwords file> <password hash>\n";
		return -1;
	}
}
