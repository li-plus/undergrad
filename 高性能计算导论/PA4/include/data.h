#ifndef DATA_H
#define DATA_H
#include "util.h"

template <class T>
void testGPUBuffer(T *dptr, int outnum = 128)
{
    const int nums = outnum;
    T temp[nums];
    checkCudaErrors(cudaMemcpy(temp, dptr, nums * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    cout.precision(4);
    for (int j = 0; j < nums; ++j)
    {
        if (j % 32 == 0)
            cout << endl;
        cout << temp[j] << ' ';
    }
    cout << '\n';
}

void load_graph(std::string dset, int &num_v, int &num_e, int *&indptr, int *&indices);

template <typename T>
T* allocate(int num, vector<void*> *tensor_ptr=NULL, bool random=true)
{
    T *tmp = NULL;
    checkCudaErrors(cudaMalloc2((void**)&tmp, sizeof(T) * ((num + 511) / 512 * 512)));
    if (random)
    {
        checkCudaErrors(curandGenerateNormal(kCuRand, (float*)tmp, ((num + 511) / 512 * 512), 0.f, 0.1));

    }
    if (tensor_ptr != NULL)
        tensor_ptr->push_back((void*)tmp);
    return tmp;
}
#endif