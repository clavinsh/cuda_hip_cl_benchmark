// Conway's Game of Life implementācija

__kernel void gol_multi_step(__global const uchar *input, __global uchar *output, __global uchar *temp, ulong width,
							 ulong height, ulong steps_to_process)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	if (x >= width || y >= height)
		return;

	size_t flatIdx = y * width + x;


	temp[flatIdx] = input[flatIdx];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (ulong step = 0; step < steps_to_process; step++)
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
		if (curr[flatIdx] == 1)
		{
			if (neighbors == 2 || neighbors == 3)
				cell = 1;
		}
		else
		{
			if (neighbors == 3)
				cell = 1;
		}

		next[flatIdx] = cell;

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (steps_to_process % 2 == 1)
	{
		output[flatIdx] = temp[flatIdx];
	}
}
