#ifndef __CONFIG_H_
#define __CONFIG_H_

#include "hls_config.h"

#define MEM_BASE_ADDR 0x01000000
#define ALIGN_SIZE 65536

#define NO_MATMUL
#define NO_CDMA

#define MEM_CHANNEL_SIZE 0x08000000
#define MEM_BASE_ADDR_0 MEM_BASE_ADDR
#define MEM_BASE_ADDR_1 MEM_CHANNEL_SIZE
#define MEM_BASE_ADDR_2 (MEM_CHANNEL_SIZE * 2)
#define MEM_BASE_ADDR_3 (MEM_CHANNEL_SIZE * 3)

#endif
