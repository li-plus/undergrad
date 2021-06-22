#include "spmm_cusparse.h"

void SpMMCuSparse::preprocess(float *vin, float *vout)
{
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, kNumV, kNumV, kNumE,
        d_ptr, d_idx, d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, kNumV, kLen, kLen, vin,
            CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, kNumV, kLen, kLen, vout,
            CUDA_R_32F, CUSPARSE_ORDER_ROW);
    size_t bufferSize = 0;
// allocate an external buffer if needed
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    checkCudaErrors(cudaMalloc2((void**)&buf, bufferSize));
}

void SpMMCuSparse::run(float *vin, float *vout)
{
    cusparseSpMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, buf);
}