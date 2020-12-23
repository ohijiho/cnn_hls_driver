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

#define N_TEST_SET 1000

#include "do_lenet.h"
#include "config.h"
#include "utils.h"
#include "fixed.h"

typedef lenet1_value_t value_t;
typedef lenet1_minibatch_t minibatch_t;

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

#define PIPE_LENGTH LENET1_PIPE_LENGTH

static void (* const LENET1_SETFUNC[PIPE_LENGTH - 1][2])(XLenet1 *, u64) = {
		{XLenet1_Set_layer1_x, XLenet1_Set_layer1_y},
		{XLenet1_Set_layer2_x, XLenet1_Set_layer2_y},
		{XLenet1_Set_layer3_x, XLenet1_Set_layer3_y},
		{XLenet1_Set_layer4_x, XLenet1_Set_layer4_y},
		{XLenet1_Set_layer5_x, XLenet1_Set_layer5_y},
		{XLenet1_Set_layer6_x, XLenet1_Set_layer6_y},
		{XLenet1_Set_layer7_x, XLenet1_Set_layer7_y},
};

typedef struct npu_lenet1_state npu_t;

static npu_t *lenet1_select(npu_t *npus, size_t nnpus) {
	bool driver_bottleneck = true;
	XTime start_time;
	XTime_GetTime(&start_time);
	for (;;) {
		size_t k;
		bool end = true;
		{
			XTime current_time;
			XTime_GetTime(&current_time);
			if (current_time - start_time >= TIMEOUT_CLOCK) {
				printf("TIMEOUT\n");
				return NULL;
			}
		}
		for (k = 0; k < nnpus; k++) {
			npu_t *npu = &npus[k];
			if (npu->iter < 2)
				driver_bottleneck = false;
			if (npu->iter < npu->iter_end + PIPE_LENGTH) {
				end = false;
				if (XLenet1_IsIdle(npu->lenet1)) {
					const size_t i = npu->iter++;
					const bool b_input = i < npu->iter_end,
							b_lenet = i >= 1 && i < npu->iter_end + PIPE_LENGTH - 1,
							b_output = i >= 8,
							b_layer6 = i >= 6 && i < npu->iter_end + 6;
					const size_t bufw = i % 2;
					const size_t bufr = 1 - bufw;
					size_t j;
					if (b_lenet) {
						for (j = 0; j < PIPE_LENGTH - 1; j++) {
							LENET1_SETFUNC[j][0](npu->lenet1, (u32)npu->buffer[j][bufr]);
							LENET1_SETFUNC[j][1](npu->lenet1, (u32)npu->buffer[j + 1][bufw]);
						}
						XLenet1_Start(npu->lenet1);
					}

					if (b_input) {
						const size_t stage = 0;
						encode_batch(npu->buffer[stage][bufw], npu->input, npu->bufsize[stage]);
						Xil_DCacheFlushRange((INTPTR)npu->buffer[stage][bufw], npu->bufsize[stage] * sizeof(minibatch_t));
					}

					if (b_output) {
						const size_t stage = PIPE_LENGTH - 1;
						Xil_DCacheInvalidateRange((INTPTR)npu->buffer[stage][bufr], npu->bufsize[stage] * sizeof(minibatch_t));
						decode_batch(npu->output, npu->buffer[stage][bufr], npu->bufsize[stage]);
						npu->iter_output = i - PIPE_LENGTH;
					}
					npu->b_output = b_output;

//					{
//						if (i == 4) {
//							const size_t stage = 3;
//							Xil_DCacheInvalidateRange((INTPTR)npu->buffer[stage][bufr], npu->bufsize[stage] * sizeof(minibatch_t));
//							for (j = 0; j < npu->bufsize[stage]; j++) {
//								size_t k;
//								for (k = 0; k < BATCH_SIZE; k++) {
//									printf("0x%08"PRIx32"\n", npu->buffer[stage][bufr][j].data[k]);
//								}
//							}
//						}
//					}

#if LAYER6_TANH_CPU
					if (b_layer6) {
						const size_t stage = 5;
						const size_t len = npu->bufsize[stage];
						const minibatch_t *xbuf = npu->buffer[stage][bufr];
						minibatch_t *ybuf = npu->buffer[stage + 1][bufw];
						size_t j;
						Xil_DCacheInvalidateRange((INTPTR)xbuf, len * sizeof(minibatch_t));
						for (j = 0; j < len; j++) {
							size_t k;
							for (k = 0; k < BATCH_SIZE; k++) {
								ybuf[j].data[k] = encode_value(tanh(decode_value(xbuf[j].data[k])));
							}
						}
						Xil_DCacheFlushRange((INTPTR)ybuf, len * sizeof(minibatch_t));
					}
#endif

					if (driver_bottleneck) {
						fprintf(stderr, "WARNING: driver bottleneck\n");
					}

					return npu;
				}
			}
		}
		driver_bottleneck = false;
		if (end)
			return NULL;
	}
}

static void lenet1_h_init(npu_t *npu, XLenet1 *lenet1,
		u32 base_addrs[4],
		const int (*alloc_regions)[2], int alloc_region_xor) {
	value_t *weights_conv1;
	value_t *bias_conv1;
	value_t *weights_conv2;
	value_t *bias_conv2;
	value_t *weights_ip1;
	value_t *bias_ip1;
	value_t *weights_ip2;
	value_t *bias_ip2;
	float *input, *output;
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

	{
		void *brk_addr[4];
		size_t i;
		for (i = 0; i < 4; i++) {
			brk_addr[i] = (void*)base_addrs[i];
		}
		static const int DEFAULT_ALLOC_REGION[PIPE_LENGTH][2] = {
				{0, 1},
				{2, 3},
				{0, 1},
				{2, 3},
				{0, 1},
				{2, 3},
				{0, 1},
				{2, 3},
		};
#define alloc(i, len) brk_alloc(&brk_addr[i], len)
#define alloc_value_t(i, n) ((value_t*)alloc(i, (n) * sizeof(value_t)))
#define alloc_minibatch_t(i, n) ((minibatch_t*)alloc(i, (n) * sizeof(minibatch_t)))
#define alloc_float(i, n) ((float*)alloc(i, (n) * sizeof(float)))

#define ALLOC_PARAMETER(i, name) \
		name = alloc_value_t(i, sizeof(_##name) / sizeof(float))
		ALLOC_PARAMETER(1, weights_conv1);
		ALLOC_PARAMETER(1, bias_conv1);
		ALLOC_PARAMETER(1, weights_conv2);
		ALLOC_PARAMETER(1, bias_conv2);
		ALLOC_PARAMETER(0, weights_ip1);
		ALLOC_PARAMETER(0, bias_ip1);
		ALLOC_PARAMETER(0, weights_ip2);
		ALLOC_PARAMETER(0, bias_ip2);
#undef DEFINE_PARAMETER

		if (alloc_regions == NULL) {
			alloc_regions = DEFAULT_ALLOC_REGION;
		}

		for (i = 0; i < PIPE_LENGTH; i++) {
			buffer[i][0] = alloc_minibatch_t(alloc_regions[i][0] ^ alloc_region_xor, bufsize[i]);
			buffer[i][1] = alloc_minibatch_t(alloc_regions[i][1] ^ alloc_region_xor, bufsize[i]);
		}
		input = alloc_float(0, BATCH_SIZE * bufsize[0]);
		output = alloc_float(0, BATCH_SIZE * bufsize[PIPE_LENGTH - 1]);

#undef alloc
#undef alloc_value_t
#undef alloc_minibatch_t
#undef alloc_float
		for (i = 0; i < 4; i++) {
			base_addrs[i] = (u32)brk_addr[i];
		}
	}

	prepare_parameters(weights_conv1, bias_conv1, weights_conv2, bias_conv2, weights_ip1, bias_ip1, weights_ip2, bias_ip2);

#define MAKE_SIZE2(w, h) ((u64)(w) | ((u64)(h) << 32))
	// TODO: w, h or h, w?
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

#define p(x) npu->x = x;
	{
		size_t i;
		p(lenet1);
		p(weights_conv1);
		p(bias_conv1);
		p(weights_conv2);
		p(bias_conv2);
		p(weights_ip1);
		p(bias_ip1);
		p(weights_ip2);
		p(bias_ip2);
		for (i = 0; i < PIPE_LENGTH; i++) {
			p(buffer[i][0]);
			p(buffer[i][1]);
			p(bufsize[i]);
		}
		p(input);
		p(output);
		npu->iter = 0;
		npu->iter_end = (N_TEST_SET - 1) / BATCH_SIZE + 1;
	}
#undef p
}

static void lenet1_h_copy_input(npu_t *npu, size_t i) {
	const size_t stage = 0;
	memcpy(npu->input, ts + npu->bufsize[stage] * i * BATCH_SIZE,
			MIN(BATCH_SIZE, N_TEST_SET - i * BATCH_SIZE) * (npu->bufsize[stage] * sizeof(float)));
}
static size_t lenet1_h_process_output(npu_t *npu, size_t i) {
	const size_t stage = PIPE_LENGTH - 1;
	size_t correct_counter = 0;
	size_t j, iter;
	for (j = 0, iter = i * BATCH_SIZE; j < BATCH_SIZE && iter < N_TEST_SET; j++, iter++) {
		uint32_t output_label = 0;
		float *cur_output = npu->output + j * npu->bufsize[stage];
		for (int k = 1; k < 10; k++) {
			if (cur_output[k] > cur_output[output_label])
				output_label = k;
		}
		if (ls[iter] == output_label) {
//						printf("Correct(%"PRIu32", %"PRIu32", %f)\n", output_label, ls[iter], cur_output[output_label]);
			correct_counter++;
		} else {
//						printf("Incorrect(%"PRIu32", %"PRIu32", %f, %f)\n", output_label, ls[iter], cur_output[output_label], cur_output[ls[iter]]);
		}
	}
	return correct_counter;
}

void do_lenet1_h_multinpu(XLenet1 *const *lenet1s, const u32 base_addrs[4], size_t nnpus) {
	static const size_t MAX_NNPUS = 4;
	typedef struct {
		size_t ioff, iend;
		size_t correct_counter;
		size_t n_test_set;
	} user_t;
	user_t user_buf[MAX_NNPUS];
	npu_t npus[MAX_NNPUS];
	const size_t N_MINIBATCH = (N_TEST_SET - 1) / BATCH_SIZE + 1;
	const size_t split_size = (N_MINIBATCH - 1) / nnpus + 1;
	printf("lenet1, header data, %zu npus\n", nnpus);
	{
		u32 brk_addrs[4];
		size_t k;
		memcpy(brk_addrs, base_addrs, 4 * sizeof(u32));
		for (k = 0; k < nnpus; k++) {
			npu_t *npu = &npus[k];
			user_t *user = &user_buf[k];
			lenet1_h_init(npu, lenet1s[k], brk_addrs, NULL, k);
			npu->user = user;
			user->ioff = k * split_size;
			user->iend = MIN(user->ioff + split_size, N_MINIBATCH);
			user->correct_counter = 0;
			user->n_test_set = MIN(user->iend * BATCH_SIZE, N_TEST_SET) - user->ioff * BATCH_SIZE;
			npu->iter_end = user->iend - user->ioff;
			lenet1_h_copy_input(npu, user->ioff);
		}
	}
	{
		npu_t *npu;
		while ((npu = lenet1_select(npus, nnpus)) != NULL) {
			user_t *user = (user_t*)npu->user;
			const size_t i = user->ioff + npu->iter;
			if (i >= user->ioff && i < user->iend) {
				lenet1_h_copy_input(npu, i);
			}
			if (npu->b_output) {
				user->correct_counter +=
						lenet1_h_process_output(npu, user->ioff + npu->iter_output);
			}
		}
	}
	{
		size_t total_correct = 0;
		size_t k;
		for (k = 0; k < nnpus; k++) {
			user_t *user = (user_t*)npus[k].user;
			total_correct += user->correct_counter;
		}
		printf("Accuracy: %lf\n", (double)total_correct / N_TEST_SET);
		for (k = 0; k < nnpus; k++) {
			user_t *user = (user_t*)npus[k].user;
			printf(" npu%zu: %lf\n", k, (double)user->correct_counter / user->n_test_set);
		}
	}
}

void do_lenet1_h(XLenet1 *lenet1,
		const u32 base_addrs[4]) {
	npu_t npus[1];
	size_t correct_counter = 0;
	const size_t N_MINIBATCH = (N_TEST_SET - 1) / BATCH_SIZE + 1;
	printf("lenet1, header data, single npu\n");
	{
		npu_t *npu = &npus[0];
		u32 brk_addrs[4];
		memcpy(brk_addrs, base_addrs, 4 * sizeof(u32));
		lenet1_h_init(npu, lenet1, brk_addrs, NULL, 0);
		npu->user = NULL;
		lenet1_h_copy_input(npu, 0);
	}

	{
		npu_t *npu;
		while ((npu = lenet1_select(npus, 1)) != NULL) {
			const size_t i = npu->iter;
			if (i < N_MINIBATCH) {
				lenet1_h_copy_input(npu, i);
			}
			if (npu->b_output) {
				correct_counter +=
						lenet1_h_process_output(npu, npu->iter_output);
			}
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
	}
	printf("Accuracy: %lf\n", (double)correct_counter / N_TEST_SET);
}
