## Kompilēšana

`$ cmake -S . -B build` (priekš ROCm) vai `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (priekš CUDA)
`$ cmake --build build`

### Uz Docker

`$ docker buildx build . --target cuda -t golhip:cuda` (priekš CUDA)
`$ docker buildx build . --target rocm -t golhip:rocm` (priekš ROCM)
