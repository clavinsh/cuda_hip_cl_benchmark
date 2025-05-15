// Conway's Game of Life implementācija

inline int neighborCount(const size_t x, const size_t y, const ulong width, const ulong height,
						 __global const uchar *grid)
{
	int neighbors = 0;

	if (y > 0)
	{
		if (x > 0)
			neighbors += grid[(y - 1) * width + (x - 1)];

		neighbors += grid[(y - 1) * width + x];

		if (x < width - 1)
			neighbors += grid[(y - 1) * width + (x + 1)];
	}

	if (x > 0)
		neighbors += grid[y * width + (x - 1)];

	if (x < width - 1)
		neighbors += grid[y * width + (x + 1)];

	if (y < height - 1)
	{
		if (x > 0)
			neighbors += grid[(y + 1) * width + (x - 1)];

		neighbors += grid[(y + 1) * width + x];

		if (x < width - 1)
			neighbors += grid[(y + 1) * width + (x + 1)];
	}

	return neighbors;
}

__kernel void gol(__global const uchar *input, __global uchar *output, ulong width, ulong height)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	if (x >= width || y >= height)
		return;

	const size_t flatIdx = y * width + x;

	int neighbors = neighborCount(x, y, width, height, input);

	uchar cell = 0;
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
