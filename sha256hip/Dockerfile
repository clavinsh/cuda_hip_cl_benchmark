FROM rocm/dev-ubuntu-24.04 AS rocm

RUN apt-get update                                  \
    && apt-get install -y --no-install-recommends   \
    build-essential                                 \
    cmake                                           \
    libspdlog-dev                                   \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -S . -B build && cmake --build build

WORKDIR /app
ENTRYPOINT ["./build/HipPwCracker"]


# !!! JĀBŪT UZBŪVĒTAM HIP-CUDA IMAGE-AM AR OTRU DOCKER FAILU !!!
FROM hip-cuda:cuda12.8.1-rocm6.4.0-ubuntu24.04 AS cuda
USER root
RUN apt-get update                                  \
    && apt-get install -y --no-install-recommends   \
    libspdlog-dev                                   \
    && rm -rf /var/lib/apt/lists/*
USER developer
WORKDIR /app
COPY . .

RUN cmake -S . -B build -D GPU_RUNTIME=CUDA && cmake --build build

WORKDIR /app
ENTRYPOINT ["./build/HipPwCracker"]
