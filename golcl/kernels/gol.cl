// Conway's Game of Life implementācija

uchar neighborCount(__constant uchar *grid, size_t width, size_t height, size_t x, size_t y)
{
    uchar neighbors = 0;

    for (size_t h = (y - 1); h <= (y+1); h++)
    {
        if (h < 0)
        {
            continue;
        }

        if (h >= height)
        {
            continue;
        }

        for (size_t w = (x - 1); w <= (x + 1); w++)
        {
            if (w < 0)
            {
                continue;
            }

            if (w >= width)
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
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if(x >= width || y >= height)
    {
        return;
    }

    uchar neighbors = neighborCount(grid, width, height, x, y);

    size_t flat_buffer_idx = width * y + x;
    uchar cell = 0;

    if(grid[flat_buffer_idx] == 1)
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

    output_grid[flat_buffer_idx] = cell;
}
