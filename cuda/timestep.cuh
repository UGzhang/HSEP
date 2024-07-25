#include "Atom.cuh"

__device__
double atomicMinDouble(double* address, double val);

__global__
void adaptiveTimeStep(Atom* atom, double* ts, double K, double gamma);
