#include "Atom.cuh"
#include <cuda.h>

void initMemory(Atom* atom, int N, int nrCells);
void prefetchGPU(Atom* atom, int N, int nrCells, CUdevice device);
void prefetchCPU(Atom* atom, Atom* atomCPU, int N, int nrCells);
void initAtom(const char* filePath, Atom* atom, int nrCells, real r);
void freeMemory(Atom* atom);