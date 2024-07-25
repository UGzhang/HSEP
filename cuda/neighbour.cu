#include "Atom.cuh"
#include <cuda.h>

//calculate current grid and neighbours, nrCells is number of cells in each Direction, cutoff the cell side length
__global__
void findNeighbor(Atom* atom, int nrCells, double cutoff){
    int N = atom->n;
    int* cells = atom->cells;
    int* particles = atom->particles;
    double* x = atom->x;
    double* y = atom->y;
    double* z = atom->z;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;


    //init cells all with -1
    for(int i = index; i < nrCells*nrCells*nrCells; i+=gridStride){
        cells[i] = -1;
    }

    //populate Arrays
    for(int i = index; i < N; i+=gridStride){
        //Determine Cell Index
        int idx = int(x[i]/cutoff);
        int idy = int(y[i]/cutoff);
        int idz = int(z[i]/cutoff);

        // Calculate the flat cell index
        int cellIndex = idx + nrCells * (idy + nrCells * idz);

        // Check bounds to avoid illegal memory access
        if (cellIndex < nrCells * nrCells * nrCells && cellIndex >= 0) {
            //populate particles
            particles[i] = i;
            //load new particles idx in cells and store the old idx in particles
            particles[i] = atomicExch(&cells[cellIndex], particles[i]);
        }
    }
}