#include "utils.h"

void nop(void *p) {}

uint32_t simple_cksum(const uint32_t *x, size_t len) {
	uint32_t ret = 0;
	while (len--)
		ret += *x++;
	return ret;
}

