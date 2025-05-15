FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update                                  \
    && apt-get install -y --no-install-recommends   \
    build-essential                                 \
    cmake                                           \
    libspdlog-dev                                   \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN mkdir -p build  \
    && cd build     \
    && cmake ..

RUN cmake --build build

WORKDIR /app
ENTRYPOINT ["./build/GameOfLifeCuda"]
