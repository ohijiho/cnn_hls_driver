/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "platform.h"
#include <xil_cache.h>
#include <inttypes.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <xtime_l.h>

#include "config.h"

#ifndef NO_MATMUL
#include <xnaive_matmul_top.h>
#else
typedef int XNaive_matmul_top;
#endif
#ifndef NO_CDMA
#include <xaxicdma.h>
#else
typedef int XAxiCdma;
#endif
#include <xlenet1.h>

#include "utils.h"

#include "do_lenet.h"

#define ECHO_INPUT 1
#define TIMEOUT_MS 1000000
#define TIMEOUT_CLOCK ((XTime)COUNTS_PER_SECOND * TIMEOUT_MS / 1000)

typedef float value_t;

#ifndef NO_MATMUL
static void ref_impl(const value_t *A, const value_t *B, value_t *C,
		size_t size_m, size_t size_k, size_t size_n) {
	for (ptrdiff_t i = 0; i < (ptrdiff_t)size_m; i++) {
		for (ptrdiff_t j = 0; j < (ptrdiff_t)size_n; j++) {
			C[i * size_n + j] = 0;
			for (ptrdiff_t k = 0; k < (ptrdiff_t)size_k; k++) {
				C[i * size_n + j] += A[i * size_k + k] * B[k * size_n + j];
			}
		}
	}
}

#define cmp_matrix(a, b, m, n) (memcmp(a, b, (m) * (n) * sizeof(value_t)))
#define gen_matrix(x, m, n) (gen_array(x, (m) * (n)))
#define fill_matrix(x, m, n, c) (fill_array(x, (m) * (n), (c)))

static void gen_array(value_t *x, size_t n) {
	while (n--)
		*x++ = (value_t)rand() / RAND_MAX;
}

static void dump_matrix(value_t *x, size_t m, size_t n) {
	for (ptrdiff_t i = 0; i < (ptrdiff_t)m; i++) {
		for (ptrdiff_t j = 0; j < (ptrdiff_t)n; j++) {
			if (j) printf(" ");
			printf("%f", x[i * n + j]);
		}
		printf("\n");
	}
}

static void fill_array(value_t *x, size_t n, value_t c) {
	while (n--)
		*x++ = c;
}
#endif

bool init(XNaive_matmul_top *top, XAxiCdma *cdma, XLenet1 *lenet1) {
#ifndef NO_MATMUL
	XNaive_matmul_top_Config *top_cfg;
#endif
#ifndef NO_CDMA
	XAxiCdma_Config *cdma_cfg;
#endif
	XLenet1_Config *lenet1_cfg;
	int status;

	init_platform();
//	print("Hello World\n");
//	print("Successfully ran Hello World application\n");

#ifndef NO_MATMUL
	top_cfg = XNaive_matmul_top_LookupConfig(XPAR_NAIVE_MATMUL_TOP_0_DEVICE_ID);
	if (top_cfg == NULL) {
		fprintf(stderr, "Error naive_matmul_top LookupConfig\n");
		return false;
	}
	status = XNaive_matmul_top_CfgInitialize(top, top_cfg);
	if (status != XST_SUCCESS) {
		fprintf(stderr, "Error initializing naive_matmul_top core: %d\n", status);
		return false;
	}
#endif

#ifndef NO_CDMA
	cdma_cfg = XAxiCdma_LookupConfig(XPAR_AXICDMA_0_DEVICE_ID);
	if (cdma_cfg == NULL) {
		fprintf(stderr, "Error CDMA LookupConfig\n");
		return false;
	}
	status = XAxiCdma_CfgInitialize(cdma, cdma_cfg, cdma_cfg->BaseAddress);
	if (status != XST_SUCCESS) {
		fprintf(stderr, "Error initializing CDMA core: %d\n", status);
		return false;
	}
#endif

	lenet1_cfg = XLenet1_LookupConfig(XPAR_LENET1_0_DEVICE_ID);
	if (lenet1_cfg == NULL) {
		fprintf(stderr, "Error lenet1_0 LookupConfig\n");
		return false;
	}
	status = XLenet1_CfgInitialize(&lenet1[0], lenet1_cfg);
	if (status != XST_SUCCESS) {
		fprintf(stderr, "Error initializing lenet1_0 core: %d\n", status);
		return false;
	}

	lenet1_cfg = XLenet1_LookupConfig(XPAR_LENET1_1_DEVICE_ID);
	if (lenet1_cfg == NULL) {
		fprintf(stderr, "Error lenet1_1 LookupConfig\n");
		return false;
	}
	status = XLenet1_CfgInitialize(&lenet1[1], lenet1_cfg);
	if (status != XST_SUCCESS) {
		fprintf(stderr, "Error initializing lenet1_1 core: %d\n", status);
		return false;
	}

	return true;
}

#ifndef NO_MATMUL
void do_matmul(XNaive_matmul_top *top,
		size_t size_m, size_t size_k, size_t size_n, unsigned int seed) {
	value_t *a, *b, *c, *c_ref;
	a = (value_t*) ((value_t*) MEM_BASE_ADDR);
	b = (value_t*) (a + size_m * size_k);
	c = (value_t*) (b + size_k * size_n);
	c_ref = (value_t*) align_ptr(c + size_m * size_n);
	srand(seed);
	printf("generating matrices..\n");
	gen_matrix(a, size_m, size_k);
	gen_matrix(b, size_k, size_n);
	Xil_DCacheFlushRange((u32) a, size_m * size_k * sizeof(value_t));
	Xil_DCacheFlushRange((u32) b, size_k * size_n * sizeof(value_t));
	fill_matrix(c, size_m, size_n, NAN);
	Xil_DCacheFlushRange((u32) c, size_m * size_n * sizeof(value_t));
	printf("transferring parameters..\n");
	XNaive_matmul_top_Set_A(top, (u64) (intptr_t) a);
	XNaive_matmul_top_Set_B(top, (u64) (intptr_t) b);
	XNaive_matmul_top_Set_C(top, (u64) (intptr_t) c);
	XNaive_matmul_top_Set_size_m(top, (u64) size_m);
	XNaive_matmul_top_Set_size_k(top, (u64) size_k);
	XNaive_matmul_top_Set_size_n(top, (u64) size_n);

#define dump_state do {\
		printf("ready: %lu, idle: %lu, done: %lu\n", \
		XNaive_matmul_top_IsReady(&top), \
		XNaive_matmul_top_IsIdle(&top), \
		XNaive_matmul_top_IsDone(&top)); \
	} while (0);

//	dump_state;
	XNaive_matmul_top_Start(top);
//	dump_state;
	printf("start!\n");
	ref_impl(a, b, c_ref, size_m, size_k, size_n);
	printf("cpu run finished\n");
//	dump_state;
	{
		XTime start_time;
		XTime_GetTime(&start_time);
		while (!XNaive_matmul_top_IsDone(top)) {
			XTime current_time;
//			dump_state;
//			Xil_DCacheInvalidateRange((u32)c, size_m * size_n * sizeof(value_t));
//			printf("c: \n");
//			dump_matrix(c, size_m, size_n);
//			usleep(1000 * 250);
			XTime_GetTime(&current_time);
			if (current_time - start_time >= TIMEOUT_CLOCK) {
				printf("TIMEOUT\n");
				break;
			}
//			printf("elapsed: %ld\n", (long)(current_time - start_time));
		}
	}
//	dump_state;
#undef dump_state
	printf("done!\n");
	Xil_DCacheInvalidateRange((u32) c, size_m * size_n * sizeof(value_t));
	if (cmp_matrix(c, c_ref, size_m, size_n) == 0) {
//		printf("c = c_ref, c[0][0] = %f, c_ref[0][0] = %f\n", c[0], c_ref[0]);
		printf("c = c_ref\n");
	} else {
//		printf("c /= c_ref, c[0][0] = %f, c_ref[0][0] = %f\n", c[0], c_ref[0]);
		printf("c /= c_ref\n");
		printf("a: \n");
		dump_matrix(a, size_m, size_k);
		printf("b: \n");
		dump_matrix(b, size_k, size_n);
		printf("c: \n");
		dump_matrix(c, size_m, size_n);
		printf("c_ref: \n");
		dump_matrix(c_ref, size_m, size_n);
	}
}
#else
void do_matmul(XNaive_matmul_top *top,
		size_t size_m, size_t size_k, size_t size_n, unsigned int seed) {
	fprintf(stderr, "no matmul\n");
}
#endif

#ifndef NO_CDMA
void do_cdma(XAxiCdma *cdma, size_t n) {
	value_t *a, *b, *b_ref;
	a = (value_t*) MEM_BASE_ADDR;
	b = a + n * sizeof(value_t);
	b_ref = b + n * sizeof(value_t);
	gen_array(a, n);
	Xil_DCacheFlushRange((u32) a, n * sizeof(value_t));
	XAxiCdma_SimpleTransfer(cdma, (u32) a, (u32) b, n * sizeof(value_t), NULL, NULL);
	printf("start!\n");
	memcpy(b_ref, a, n * sizeof(value_t));
	printf("cpu run finished!\n");
	Xil_DCacheInvalidateRange((u32) b, n * sizeof(value_t));
	while (XAxiCdma_IsBusy(cdma));

	printf("done!\n");
	if (memcmp(b, b_ref, n * sizeof(value_t)) == 0) {
		printf("b = b_ref, b[0] = %f, b_ref[0] = %f\n", b[0], b_ref[0]);
	} else {
		printf("b /= b_ref, b[0] = %f, b_ref[0] = %f\n", b[0], b_ref[0]);
	}
}
#else
void do_cdma(XAxiCdma *cdma, size_t n) {
	fprintf(stderr, "no cdma\n");
}
#endif

int main()
{
    XNaive_matmul_top top;
    XAxiCdma cdma;
    XLenet1 lenet1[2];

	if (!init(&top, &cdma, lenet1))
		return 1;

    for (;;) {
    	char command[256];
    	printf("Command: ");
//    	fgets(command, sizeof(command), stdin);
//    	gets(command);
    	scanf("%s", command);
#if ECHO_INPUT
    	printf("%s\n", command);
#endif
//    	printf("got command: %s\n", command);
    	{
    		size_t i = strlen(command);
    		if (command[i - 1] == '\n')
    			command[i - 1] = 0;
    	}
    	if (strncmp("matmul", command, sizeof(command)) == 0) {
        	size_t size_m, size_k, size_n;
        	unsigned int seed;
			printf("Enter m k n seed: ");
			scanf("%zu%zu%zu", &size_m, &size_k, &size_n);
			scanf("%u", &seed);
#if ECHO_INPUT
			printf("%zu %zu %zu %u\n", size_m, size_k, size_n, seed);
#endif
			do_matmul(&top, size_m, size_k, size_n, seed);
    	} else if (strncmp("cdma", command, sizeof(command)) == 0) {
    		size_t n;
    		printf("n: ");
    		scanf("%zu", &n);
#if ECHO_INPUT
    		printf("%zu\n", n);
#endif
			do_cdma(&cdma, n);
    	} else if (strncmp("lenet1-h", command, sizeof(command)) == 0 ||
    			strncmp("1", command, sizeof(command)) == 0) {
    		XTime start_time, end_time;
    		long t;
    		XTime_GetTime(&start_time);
    		do_lenet1_h(&lenet1[0], MEM_BASE_ADDR_2, MEM_BASE_ADDR_3);
			XTime_GetTime(&end_time);
			t = (end_time - start_time + (COUNTS_PER_SECOND / 2000)) / (COUNTS_PER_SECOND / 1000);
			printf("%ld.%03lds\n", t / 1000, t % 1000);
    	} else if (strncmp("lenet1-h-m", command, sizeof(command)) == 0 ||
    			strncmp("-m", command, sizeof(command)) == 0) {
    		XTime start_time, end_time;
    		long t;
    		size_t nnpus;
    		static const u32 base_addrs[4] = {
    				MEM_BASE_ADDR_0, MEM_BASE_ADDR_1, MEM_BASE_ADDR_2, MEM_BASE_ADDR_3
    		};
    		XLenet1 *const lenet1s[2] = {
    				&lenet1[0], &lenet1[1]
    		};
    		scanf("%zu", &nnpus);
    		XTime_GetTime(&start_time);
    		do_lenet1_h_multinpu(lenet1s, base_addrs, nnpus);
			XTime_GetTime(&end_time);
			t = (end_time - start_time + (COUNTS_PER_SECOND / 2000)) / (COUNTS_PER_SECOND / 1000);
			printf("%ld.%03lds\n", t / 1000, t % 1000);
    	} else if (strncmp("exit", command, sizeof(command)) == 0) {
    		break;
    	}
    }

    cleanup_platform();
    printf("(exit)\n");
    return 0;
}
