#include "Atom.cuh"

__global__
void findNeighbor(Atom* atom, int nrCells, double cutoff);