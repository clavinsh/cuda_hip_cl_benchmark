// Conway's Game of Life implementācija

int neighborCount(__constant uchar *grid, size_t width, size_t height, size_t x, size_t y)
{
	int neighbors = 0;

	for (size_t diff_y = 0; diff_y < 3; diff_y++)
	{
		size_t h = y + diff_y - 1;

		if (h >= height)
		{
			continue;
		}

		for (size_t diff_x = 0; diff_x < 3; diff_x++)
		{
			size_t w = x + diff_x - 1;

			if (w >= width)
			{
				continue;
			}

			if (diff_x == 1 && diff_y == 1)
			{
				continue;
			}

			if (grid[h * width + w] == 1)
			{
				neighbors++;
			}
		}
	}

	return neighbors;
}

__kernel void gol(__constant uchar *grid, __global uchar *output_grid, ulong width, ulong height)
{
	size_t idx = get_global_id(0);

	if (idx >= width * height)
	{
		return;
	}

	size_t x = idx % width;
	size_t y = idx / width;

	int neighbors = neighborCount(grid, width, height, x, y);

	uchar cell = 0;

	if (grid[idx] == 1)
	{
		if (neighbors == 2 || neighbors == 3)
		{
			cell = 1;
		}
	}
	else
	{
		if (neighbors == 3)
		{
			cell = 1;
		}
	}

	output_grid[idx] = cell;
}
