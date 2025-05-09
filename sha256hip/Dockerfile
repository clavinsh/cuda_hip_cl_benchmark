FROM rocm/dev-ubuntu-24.04

RUN apt-get update                                  \
    && apt-get install -y --no-install-recommends   \
    build-essential                                 \
    cmake                                           \
    libspdlog-dev                                   \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -S . -B build -D GPU_RUNTIME=CUDA

WORKDIR /app
ENTRYPOINT ["./build/HipPwCracker"]
