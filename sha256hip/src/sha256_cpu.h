#ifndef SHA256_CPU_H
#define SHA256_CPU_H

#include <cstddef>
#include <cstdint>

void cpu_sha256(const uint8_t *input, size_t length, uint8_t *output);

#endif
