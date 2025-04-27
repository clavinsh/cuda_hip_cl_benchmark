#include <CL/cl.h>
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
#include <vector>

class GameOfLife
{
  public:
	std::vector<uint8_t> grid;
	size_t width;
	size_t height;

	GameOfLife(size_t width, size_t height)
	{
		width = width;
		height = height;
		grid = std::vector<uint8_t>(width * height, 0);
	}
};

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
std::vector<uint8_t> loadGridFromFile(const std::string &fileName, size_t &width, size_t &height)
{
	std::vector<uint8_t> grid;
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

int main(int argc, char *argv[])
{
	if (argc == 2)
	{
		const std::string inputFileName = argv[1];

		size_t width;
		size_t height;
		std::vector<uint8_t> grid = loadGridFromFile(inputFileName, width, height);

		for (size_t h = 0; h < height; h++)
		{
			for (size_t w = 0; w < width; w++)
			{
				std::cout << std::to_string(grid[h * width + w]);
			}
			std::cout << "\n";
		}
	}
	else
	{
		std::cout << "Correct program usage:\n"
				  << "\t\t" << argv[0] << " <grid file path>\n";
	}
	return 0;
}
