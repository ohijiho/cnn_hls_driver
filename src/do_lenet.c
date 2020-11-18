#include <xil_cache.h>
#include <xlenet1.h>
#include <xtime_l.h>
#include <stdbool.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>

typedef float w_t;

#include "./data/weights_conv1.h"
#include "./data/weights_conv2.h"
#include "./data/weights_ip1.h"
#include "./data/weights_ip2.h"
#include "./data/bias_conv1.h"
#include "./data/bias_conv2.h"
#include "./data/bias_ip1.h"
#include "./data/bias_ip2.h"
#include "./data/test_set.h"
#include "./data/label.h"

#include "config.h"
#include "utils.h"
#include "fixed.h"

typedef fixed32_8_t value_t;
typedef struct {
	value_t data[BATCH_SIZE];
} minibatch_t;

#define encode_value float_to_fixed32_8
#define decode_value fixed32_8_to_float

#define TIMEOUT_MS 100
#define TIMEOUT_CLOCK ((XTime)COUNTS_PER_SECOND * TIMEOUT_MS / 1000)

static void encode_batch(minibatch_t *dst, const float *src, size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < BATCH_SIZE; j++) {
			dst[i].data[j] = encode_value(src[j * n + i]);
		}
	}
}
static void decode_batch(float *dst, const minibatch_t *src, size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < BATCH_SIZE; j++) {
			dst[j * n + i] = decode_value(src[i].data[j]);
		}
	}
}

static void copy_transpose(value_t *dst, const float *src, size_t m, size_t n) {
	for (size_t j = 0; j < n; j++) {
		for (size_t i = 0; i < m; i++) {
			dst[j * m + i] = encode_value(src[i * n + j]);
		}
	}
}

static void prepare_parameters(
		value_t *weights_conv1, value_t *bias_conv1,
		value_t *weights_conv2, value_t *bias_conv2,
		value_t *weights_ip1, value_t *bias_ip1,
		value_t *weights_ip2, value_t *bias_ip2) {
#define COPY_PARAMETER(name, m) do { \
		copy_transpose(name, _##name, m, sizeof(_##name) / sizeof(float) / (m)); \
		Xil_DCacheFlushRange((u32)name, sizeof(value_t) * sizeof(_##name) / sizeof(float)); \
	} while (0)
#define COPY_WB_PAIR(name) do { \
		COPY_PARAMETER(weights_##name, sizeof(_bias_##name) / sizeof(float)); \
		COPY_PARAMETER(bias_##name, 1); \
	} while (0)
	COPY_WB_PAIR(conv1);
	COPY_WB_PAIR(conv2);
	COPY_WB_PAIR(ip1);
	COPY_WB_PAIR(ip2);
#undef COPY_PARAMETER
#undef COPY_WB_PAIR
}

static inline void *brk_alloc(void **brk_addr, size_t len) {
	void *ret = *brk_addr;
	*brk_addr = align_ptr((void*)((intptr_t)*brk_addr + len));
	return ret;
}

void do_lenet1_h(XLenet1 *lenet1, u32 base_addr_0, u32 base_addr_1) {
	value_t *weights_conv1;
	value_t *bias_conv1;
	value_t *weights_conv2;
	value_t *bias_conv2;
	value_t *weights_ip1;
	value_t *bias_ip1;
	value_t *weights_ip2;
	value_t *bias_ip2;
	float *input, *output;
#define PIPE_LENGTH 8
	minibatch_t *buffer[PIPE_LENGTH][2];
	const size_t bufsize[PIPE_LENGTH] = {
			1 * 28 * 28,
			5 * 24 * 24,
			5 * 12 * 12,
			5 * 8 * 8,
			5 * 4 * 4,
			40,
			40,
			10
	};

	void (* const setfunc[PIPE_LENGTH - 1][2])(XLenet1 *, u64) = {
			{XLenet1_Set_layer1_x, XLenet1_Set_layer1_y},
			{XLenet1_Set_layer2_x, XLenet1_Set_layer2_y},
			{XLenet1_Set_layer3_x, XLenet1_Set_layer3_y},
			{XLenet1_Set_layer4_x, XLenet1_Set_layer4_y},
			{XLenet1_Set_layer5_x, XLenet1_Set_layer5_y},
			{XLenet1_Set_layer6_x, XLenet1_Set_layer6_y},
			{XLenet1_Set_layer7_x, XLenet1_Set_layer7_y},
	};

	const size_t N_TEST_SET = 1000;

	{
		void *brk_addr[2] = {(void*)base_addr_0, (void*)base_addr_1};
		size_t i;
#define alloc(i, len) brk_alloc(&brk_addr[i], len)
#define alloc_value_t(i, n) ((value_t*)alloc(i, (n) * sizeof(value_t)))
#define alloc_minibatch_t(i, n) ((minibatch_t*)alloc(i, (n) * sizeof(minibatch_t)))
#define alloc_float(i, n) ((float*)alloc(i, (n) * sizeof(float)))

#define ALLOC_PARAMETER(i, name) \
		name = alloc_value_t(i, sizeof(_##name) / sizeof(float))
		ALLOC_PARAMETER(0, weights_conv1);
		ALLOC_PARAMETER(1, bias_conv1);
		ALLOC_PARAMETER(1, weights_conv2);
		ALLOC_PARAMETER(0, bias_conv2);
		ALLOC_PARAMETER(1, weights_ip1);
		ALLOC_PARAMETER(0, bias_ip1);
		ALLOC_PARAMETER(0, weights_ip2);
		ALLOC_PARAMETER(1, bias_ip2);
#undef DEFINE_PARAMETER

		for (i = 0; i < PIPE_LENGTH; i++) {
			buffer[i][0] = alloc_minibatch_t(0, bufsize[i]);
			buffer[i][1] = alloc_minibatch_t(1, bufsize[i]);
		}
		input = alloc_float(0, BATCH_SIZE * bufsize[0]);
		output = alloc_float(0, BATCH_SIZE * bufsize[PIPE_LENGTH - 1]);

#undef alloc
#undef alloc_value_t
#undef alloc_minibatch_t
#undef alloc_float
	}

	prepare_parameters(weights_conv1, bias_conv1, weights_conv2, bias_conv2, weights_ip1, bias_ip1, weights_ip2, bias_ip2);

#define MAKE_SIZE2(w, h) ((u64)(w) | ((u64)(h) << 32))
	XLenet1_Set_layer1_weight(lenet1, (u32)weights_conv1);
	XLenet1_Set_layer1_bias(lenet1, (u32)bias_conv1);
	XLenet1_Set_layer1_input_size(lenet1, MAKE_SIZE2(28, 28));
	XLenet1_Set_layer1_in_channels(lenet1, (u32)1);
	XLenet1_Set_layer1_out_channels(lenet1, (u32)5);
	XLenet1_Set_layer1_kernel_size(lenet1, MAKE_SIZE2(5, 5));
	XLenet1_Set_layer1_stride(lenet1, MAKE_SIZE2(1, 1));
	XLenet1_Set_layer1_padding(lenet1, MAKE_SIZE2(0, 0));
	XLenet1_Set_layer1_dilation(lenet1, MAKE_SIZE2(1, 1));

	XLenet1_Set_layer2_channels(lenet1, (u32)5);
	XLenet1_Set_layer2_input_size(lenet1, MAKE_SIZE2(24, 24));
	XLenet1_Set_layer2_kernel_size(lenet1, MAKE_SIZE2(2, 2));
	XLenet1_Set_layer2_stride(lenet1, MAKE_SIZE2(2, 2));
	XLenet1_Set_layer2_padding(lenet1, MAKE_SIZE2(0, 0));
	XLenet1_Set_layer2_dilation(lenet1, MAKE_SIZE2(1, 1));

	XLenet1_Set_layer3_weight(lenet1, (u32)weights_conv2);
	XLenet1_Set_layer3_bias(lenet1, (u32)bias_conv2);
	XLenet1_Set_layer3_input_size(lenet1, MAKE_SIZE2(12, 12));
	XLenet1_Set_layer3_in_channels(lenet1, (u32)5);
	XLenet1_Set_layer3_out_channels(lenet1, (u32)5);
	XLenet1_Set_layer3_kernel_size(lenet1, MAKE_SIZE2(5, 5));
	XLenet1_Set_layer3_stride(lenet1, MAKE_SIZE2(1, 1));
	XLenet1_Set_layer3_padding(lenet1, MAKE_SIZE2(0, 0));
	XLenet1_Set_layer3_dilation(lenet1, MAKE_SIZE2(1, 1));

	XLenet1_Set_layer4_channels(lenet1, (u32)5);
	XLenet1_Set_layer4_input_size(lenet1, MAKE_SIZE2(8, 8));
	XLenet1_Set_layer4_kernel_size(lenet1, MAKE_SIZE2(2, 2));
	XLenet1_Set_layer4_stride(lenet1, MAKE_SIZE2(2, 2));
	XLenet1_Set_layer4_padding(lenet1, MAKE_SIZE2(0, 0));
	XLenet1_Set_layer4_dilation(lenet1, MAKE_SIZE2(1, 1));

	XLenet1_Set_layer5_weight(lenet1, (u32)weights_ip1);
	XLenet1_Set_layer5_bias(lenet1, (u32)bias_ip1);
	XLenet1_Set_layer5_in_features(lenet1, (u32)80);
	XLenet1_Set_layer5_out_features(lenet1, (u32)40);

	XLenet1_Set_layer6_features(lenet1, (u32)40);

	XLenet1_Set_layer7_weight(lenet1, (u32)weights_ip2);
	XLenet1_Set_layer7_bias(lenet1, (u32)bias_ip2);
	XLenet1_Set_layer7_in_features(lenet1, (u32)40);
	XLenet1_Set_layer7_out_features(lenet1, (u32)10);

	{
		size_t i, N_MINIBATCH = (N_TEST_SET - 1) / BATCH_SIZE + 1;
		size_t correct_counter = 0;
		for (i = 0; i < N_MINIBATCH + PIPE_LENGTH; i++) {
			const size_t bufw = i % 2;
			const size_t bufr = 1 - bufw;
			const bool b_input = i < N_MINIBATCH,
					b_lenet = i >= 1 && i < N_MINIBATCH + PIPE_LENGTH - 1,
					b_output = i >= 8,
					b_layer6 = i >= 6 && i < N_MINIBATCH + 6;

			if (b_lenet) {
				size_t j;
				for (j = 0; j < PIPE_LENGTH - 1; j++) {
					setfunc[j][0](lenet1, (u32)buffer[j][bufr]);
					setfunc[j][1](lenet1, (u32)buffer[j + 1][bufw]);
				}
				XLenet1_Start(lenet1);
			}

			{
#define PRINT_CKSUM 0
#if PRINT_CKSUM
				size_t j, k;
				uint32_t cksum[PIPE_LENGTH][2];
				for (j = 0; j < PIPE_LENGTH; j++) {
					for (k = 0; k < 2; k++) {
						Xil_DCacheInvalidateRange((INTPTR)buffer[j][k], bufsize[j] * sizeof(minibatch_t));
						cksum[j][k] =
								((simple_cksum((uint32_t*)buffer[j][k], bufsize[j] * BATCH_SIZE / 2) & 0xFFFF) << 16) |
								(simple_cksum((uint32_t*)buffer[j][k] + bufsize[j] * BATCH_SIZE / 2, (bufsize[j] * BATCH_SIZE + 1) / 2) & 0xFFFF);
					}
				}
				printf("cksum: {");
				for (j = 0; j < PIPE_LENGTH; j++) {
					printf("{%08"PRIx32", %08"PRIx32"}, ", cksum[j][0], cksum[j][1]);
				}
				printf("}\n");
#endif
			}

			if (b_input) {
				const size_t stage = 0;
				memcpy(input, ts + bufsize[stage] * i * BATCH_SIZE,
						MIN(BATCH_SIZE, N_TEST_SET - i * BATCH_SIZE) * (bufsize[stage] * sizeof(float)));
				encode_batch(buffer[stage][bufw], input, bufsize[stage]);
				Xil_DCacheFlushRange((INTPTR)buffer[stage][bufw], bufsize[stage] * sizeof(minibatch_t));
			}

			if (b_output) {
				const size_t stage = PIPE_LENGTH - 1;
				size_t j, iter;
				Xil_DCacheInvalidateRange((INTPTR)buffer[stage][bufr], bufsize[stage] * sizeof(minibatch_t));
				decode_batch(output, buffer[stage][bufr], bufsize[stage]);
				for (j = 0, iter = (i - (stage + 1)) * BATCH_SIZE; j < BATCH_SIZE && iter < N_TEST_SET; j++, iter++) {
					uint32_t output_label = 0;
					float *cur_output = output + j * bufsize[stage];
					for (int i = 1; i < 10; i++) {
						if (cur_output[i] > cur_output[output_label])
							output_label = i;
					}
					if (ls[iter] == output_label) {
//						printf("Correct(%"PRIu32", %"PRIu32", %f)\n", output_label, ls[iter], cur_output[output_label]);
						correct_counter++;
					} else {
//						printf("Incorrect(%"PRIu32", %"PRIu32", %f, %f)\n", output_label, ls[iter], cur_output[output_label], cur_output[ls[iter]]);
					}
				}
			}

#if LAYER6_TANH_CPU
			if (b_layer6) {
				const size_t stage = 5;
				const size_t len = bufsize[stage];
				const minibatch_t *xbuf = buffer[stage][bufr];
				minibatch_t *ybuf = buffer[stage + 1][bufw];
				size_t j;
				Xil_DCacheInvalidateRange((INTPTR)xbuf, len * sizeof(minibatch_t));
				for (j = 0; j < bufsize[stage]; j++) {
					size_t k;
					for (k = 0; k < BATCH_SIZE; k++) {
						ybuf[j].data[k] = encode_value(tanh(decode_value(xbuf[j].data[k])));
					}
				}
				Xil_DCacheFlushRange((INTPTR)ybuf, len * sizeof(minibatch_t));
			}
#endif

			if (b_lenet) {
				XTime start_time;
				XTime_GetTime(&start_time);
				while (!XLenet1_IsDone(lenet1)) {
//					printf(".");fflush(stdout);
					XTime current_time;
					XTime_GetTime(&current_time);
					if (current_time - start_time >= TIMEOUT_CLOCK) {
						printf("TIMEOUT\n");
						break;
					}
				}
			}
		}
//		printf("\n");
		printf("Accuracy: %lf\n", (double)correct_counter / N_TEST_SET);
	}
}
