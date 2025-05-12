// Conway's Game of Life implementācija

int neighborCount(__global const uchar *grid, size_t width, size_t height, size_t x, size_t y)
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

__kernel void gol(__global const uchar *grid, __global uchar *output_grid, ulong width, ulong height)
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

__kernel void gol_v2(__global const uchar *input, __global uchar *output, ulong width, ulong height)
{

	// Get 2D position directly
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	// Early return if out of bounds
	if (x >= width || y >= height)
		return;

	// Get flat index
	size_t idx = y * width + x;

	// Local memory for a tile of cells
	__local uchar local_grid[18][18]; // 16x16 workgroup + 1-cell border

	// Local position within workgroup
	size_t lx = get_local_id(0);
	size_t ly = get_local_id(1);

	// Local workgroup size
	size_t local_width = get_local_size(0);
	size_t local_height = get_local_size(1);

	// Workgroup origin in global space
	size_t group_x = get_group_id(0) * local_width;
	size_t group_y = get_group_id(1) * local_height;

	// Load cell and its neighbors into local memory
	for (int dy = -1; dy <= 1; dy++)
	{
		int gy = (int)y + dy;
		int ly_local = (int)ly + dy + 1;

		// Skip if globally out of bounds
		if (gy < 0 || gy >= height)
			continue;

		for (int dx = -1; dx <= 1; dx++)
		{
			int gx = (int)x + dx;
			int lx_local = (int)lx + dx + 1;

			// Skip if globally out of bounds
			if (gx < 0 || gx >= width)
				continue;

			// Skip if locally out of bounds
			if (lx_local < 0 || lx_local >= local_width + 2 || ly_local < 0 || ly_local >= local_height + 2)
				continue;

			// Load cell into local memory
			local_grid[ly_local][lx_local] = input[gy * width + gx];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Count neighbors using local memory
	int neighbors = 0;
	for (int dy = -1; dy <= 1; dy++)
	{
		int ly_local = (int)ly + dy + 1;
		if (ly_local < 0 || ly_local >= local_height + 2)
			continue;

		for (int dx = -1; dx <= 1; dx++)
		{
			int lx_local = (int)lx + dx + 1;
			if (lx_local < 0 || lx_local >= local_width + 2)
				continue;

			if (dx == 0 && dy == 0)
				continue;

			if (local_grid[ly_local][lx_local] == 1)
				neighbors++;
		}
	}

	// Apply Game of Life rules
	uchar cell = 0;
	if (input[idx] == 1)
	{
		if (neighbors == 2 || neighbors == 3)
			cell = 1;
	}
	else
	{
		if (neighbors == 3)
			cell = 1;
	}

	output[idx] = cell;
}

__kernel void gol_multi_step(__global const uchar *input, __global uchar *output, __global uchar *temp, ulong width,
							 ulong height, uchar steps_to_process)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	if (x >= width || y >= height)
		return;

	size_t idx = y * width + x;

	temp[idx] = input[idx];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (uchar step = 0; step < steps_to_process; step++)
	{
		__global const uchar *curr = (step % 2 == 0) ? temp : output;
		__global uchar *next = (step % 2 == 0) ? output : temp;

		int neighbors = 0;
		for (int dy = -1; dy <= 1; dy++)
		{
			int gy = (int)y + dy;
			if (gy < 0 || gy >= height)
				continue;

			for (int dx = -1; dx <= 1; dx++)
			{
				int gx = (int)x + dx;
				if (gx < 0 || gx >= width)
					continue;

				if (dx == 0 && dy == 0)
					continue;

				if (curr[gy * width + gx] == 1)
					neighbors++;
			}
		}

		uchar cell = 0;
		if (curr[idx] == 1)
		{
			if (neighbors == 2 || neighbors == 3)
				cell = 1;
		}
		else
		{
			if (neighbors == 3)
				cell = 1;
		}

		next[idx] = cell;

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Ensure final result is in output buffer if using an odd number of steps
	if (steps_to_process % 2 == 1)
	{
		output[idx] = temp[idx];
	}
}
