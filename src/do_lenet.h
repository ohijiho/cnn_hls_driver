#ifndef __DO_LENET_H_
#define __DO_LENET_H_

#include "config.h"
#include "fixed.h"
#include <xlenet1.h>
#include <stdbool.h>

typedef fixed32_8_t lenet1_value_t;
typedef struct {
	lenet1_value_t data[BATCH_SIZE];
} lenet1_minibatch_t;

#define LENET1_PIPE_LENGTH 8

struct npu_lenet1_state {
	XLenet1 *lenet1;
	const lenet1_value_t *weights_conv1;
	const lenet1_value_t *bias_conv1;
	const lenet1_value_t *weights_conv2;
	const lenet1_value_t *bias_conv2;
	const lenet1_value_t *weights_ip1;
	const lenet1_value_t *bias_ip1;
	const lenet1_value_t *weights_ip2;
	const lenet1_value_t *bias_ip2;
	lenet1_minibatch_t *buffer[LENET1_PIPE_LENGTH][2];
	size_t bufsize[LENET1_PIPE_LENGTH];
	float *input;
	float *output;
	size_t iter, iter_end;
	bool b_output;
	size_t iter_output;
	void *user;
};

void do_lenet1_h(XLenet1 *lenet1, const u32 base_addrs[4]);
void do_lenet1_h_multinpu(XLenet1 *const *lenet1s, const u32 *base_addrs, size_t nnpus);


#endif
