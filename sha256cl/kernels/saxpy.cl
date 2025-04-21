__kernel void saxpy(const float a, __global float *x, __global float *y, const int n)
{
	int i = get_global_id(0);

	if (i < n)
	{
		y[i] = a * x[i] + y[i];
	}
}

