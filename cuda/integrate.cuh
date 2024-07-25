#include "Atom.cuh"


__global__
void integrate(Atom* atom, real ts, double domain_size);