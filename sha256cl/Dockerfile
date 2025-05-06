FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update                                  \
    && apt-get install -y --no-install-recommends   \
    build-essential                                 \
    cmake                                           \
    ocl-icd-libopencl1                              \
    ocl-icd-opencl-dev                              \
    opencl-headers                                  \
    clinfo                                          \
    libspdlog-dev                                   \
    && rm -rf /var/lib/apt/lists/*

# Smylinks priekš OpenCL
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

WORKDIR /app
COPY . .

RUN mkdir -p build  \
    && cd build     \
    && cmake ..

RUN cmake --build build

WORKDIR /app
ENTRYPOINT ["./build/PasswordCracker"]
