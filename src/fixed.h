#ifndef __FIXED_H_
#define __FIXED_H_

#include <stdint.h>

typedef int32_t fixed32_8_t;//(31) => sign, (30:24) => int, (23:0) => frac

static fixed32_8_t float_to_fixed32_8(float x) {
	typedef fixed32_8_t T;
	return x * ((T)1 << 24);
}
static float fixed32_8_to_float(fixed32_8_t x) {
	typedef fixed32_8_t T;
	return (float)x / ((T)1 << 24);
}

#endif
