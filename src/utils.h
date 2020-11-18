#ifndef __UTILS_H_
#define __UTILS_H_

#include "config.h"
#include <stdint.h>
#include <stddef.h>

#define align_ptr(min_addr) ((void*)(((intptr_t)(min_addr) + ALIGN_SIZE - 1) & -(intptr_t)ALIGN_SIZE))

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif
#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

void nop(void *p);
uint32_t simple_cksum(const uint32_t *x, size_t len);

#endif
