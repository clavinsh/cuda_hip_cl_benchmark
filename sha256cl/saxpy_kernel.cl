__kernel void saxpy(__global float *x, __global float *y, const float a, __global float *result, const int n)
{
    int i = get_global_id(0);

    if (i < n)
    {
        result[i] = a * x[i] + y[i];
    }
}
