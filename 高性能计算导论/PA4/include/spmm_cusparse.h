#ifndef SpMM_CUSPARSE_H
#define SpMM_CUSPARSE_H
#include "spmm_base.h"
#include <cusparse.h>

class SpMMCuSparse : public SpMM
{
public:
    SpMMCuSparse(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMCuSparse(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

private:
    cusparseHandle_t handle=0;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    float alpha = 1.0;
    float beta = 0.0;
    float *buf = NULL;
};
#endif