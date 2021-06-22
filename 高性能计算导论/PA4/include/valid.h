#ifndef SPMM_H
#define SPMM_H

#include "util.h"
// #define LENFEATURE 64

// #define LENFEATURE2 64
// #define MAXNUMNEIGHBOR 294
// #define TB2 32

__global__ void validate_float(float *ref, float *ans, int num, int *diffnum);

int valid(float *y, float *y2, int num);
int valid(int *y, int *y2, int num);

#endif
