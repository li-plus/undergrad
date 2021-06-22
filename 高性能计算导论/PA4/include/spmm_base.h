#ifndef SpMM_BASE_H
#define SpMM_BASE_H

#include "util.h"
#include "data.h"
__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE);

class SpMM
{
public:
    SpMM(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : d_ptr(dev_out_ptr), d_idx(dev_out_idx), num_v(out_num_v), num_e(out_num_e), feat_in(out_feat_in)
    {
    }
    SpMM(CSR *g, int out_feat_in) : feat_in(out_feat_in)
    {
        d_ptr = g->ptr;
        d_idx = g->idx;
        d_val = g->val;
        num_v = g->num_v;
        num_e = g->num_e;
    }
    ~SpMM()
    {
    }

    inline void set_feat(int given_feat)
    {
        this->feat_in = given_feat;
    }

    virtual void preprocess(float *vin, float *vout) = 0;
    virtual void run(float *vin, float *vout) = 0;
    
protected:
    int *d_ptr = NULL;
    int *d_idx = NULL;
    float *d_val = NULL;

    int feat_in = 0;

    int num_v = 0;
    int num_e = 0;

    dim3 grid;
    dim3 block;
};
#endif