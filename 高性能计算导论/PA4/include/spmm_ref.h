#ifndef SpMM_REF_H
#define SpMM_REF_H
#include "spmm_base.h"

__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE);

class SpMMRef : public SpMM
{
public:
    SpMMRef(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMRef(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);
};
#endif