#include "Atom.cuh"
#include <cuda_runtime.h>
#include <iostream>

__device__
double atomicMinDouble(double* address, double val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__
void adaptiveTimeStep(Atom* atom, double* ts, double K, double gamma){
    //calculate in parallel timestep for all velocity magnitudes present, select lowest one
    int N = atom->n;
    real *vx = atom->vx;
    real *vy = atom->vy;
    real *vz = atom->vz;
    real *r = atom->r;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;
    if(index >= N)
        return;
    for (int i = index; i < N; i+=gridStride) {
        double magnitude = sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        if(magnitude==0)
            return;
        double scaling_factor = (r[i] * K * gamma)/( 100 * magnitude);
        //printf("Scaling Factor: %f \n", scaling_factor);
        //determine smallest time step required and write into adress
        atomicMinDouble(ts, scaling_factor);
    }

}
