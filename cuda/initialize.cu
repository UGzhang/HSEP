#include "Atom.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cuda.h>


#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// Initialize managed CUDA memory
void initMemory(Atom* atom, int N, int nrCells){
    atom->n = N;
    CHECK(cudaMallocManaged(&atom->m, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->x, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->y, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->z, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->vx, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->vy, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->vz, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->fx, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->fy, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->fz, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->r, N * sizeof(real)));
    CHECK(cudaMallocManaged(&atom->cells, nrCells * nrCells * nrCells * sizeof(int)));
    CHECK(cudaMallocManaged(&atom->particles, N * sizeof(int)));
    CHECK(cudaMallocManaged(&atom->I, N * 9 * sizeof(real))); // Added for inertia tensors
    CHECK(cudaMallocManaged(&atom->tx, N * sizeof(real))); // Added for torques
    CHECK(cudaMallocManaged(&atom->ty, N * sizeof(real))); // Added for torques
    CHECK(cudaMallocManaged(&atom->tz, N * sizeof(real))); // Added for torques
    CHECK(cudaMallocManaged(&atom->wx, N * sizeof(real))); // Added for angular velocities
    CHECK(cudaMallocManaged(&atom->wy, N * sizeof(real))); // Added for angular velocities
    CHECK(cudaMallocManaged(&atom->wz, N * sizeof(real))); // Added for angular velocities
    CHECK(cudaMallocManaged(&atom->qx, N * sizeof(real))); // Added for quaternions
    CHECK(cudaMallocManaged(&atom->qy, N * sizeof(real))); // Added for quaternions
    CHECK(cudaMallocManaged(&atom->qz, N * sizeof(real))); // Added for quaternions
    CHECK(cudaMallocManaged(&atom->qw, N * sizeof(real))); // Added for quaternions
}

// Load data to GPU
void prefetchGPU(Atom* atom, int N, int nrCells, CUdevice device){
    CHECK(cudaMemPrefetchAsync(atom, sizeof(Atom), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->m, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->x, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->y, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->z, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->vx, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->vy, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->vz, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->fx, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->fy, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->fz, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->r, N * sizeof(real), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->cells, nrCells*nrCells*nrCells * sizeof(int), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->particles, N * sizeof(int), device, 0));
    CHECK(cudaMemPrefetchAsync(atom->I, N * 9 * sizeof(real), device)); // Added for inertia tensors
    CHECK(cudaMemPrefetchAsync(atom->tx, N * sizeof(real), device, 0)); // Added for torques
    CHECK(cudaMemPrefetchAsync(atom->ty, N * sizeof(real), device, 0)); // Added for torques
    CHECK(cudaMemPrefetchAsync(atom->tz, N * sizeof(real), device, 0)); // Added for torques
    CHECK(cudaMemPrefetchAsync(atom->wx, N * sizeof(real), device, 0)); // Added for angular velocities
    CHECK(cudaMemPrefetchAsync(atom->wy, N * sizeof(real), device, 0)); // Added for angular velocities
    CHECK(cudaMemPrefetchAsync(atom->wz, N * sizeof(real), device, 0)); // Added for angular velocities
    CHECK(cudaMemPrefetchAsync(atom->qx, N * sizeof(real), device, 0)); // Added for quaternions
    CHECK(cudaMemPrefetchAsync(atom->qy, N * sizeof(real), device, 0)); // Added for quaternions
    CHECK(cudaMemPrefetchAsync(atom->qz, N * sizeof(real), device, 0)); // Added for quaternions
    CHECK(cudaMemPrefetchAsync(atom->qw, N * sizeof(real), device, 0)); // Added for quaternions
}

// Load data to CPU
void prefetchCPU(Atom* atom, Atom* atomCPU, int N, int nrCells){
    CHECK(cudaMemcpyAsync(atomCPU, atom, sizeof(Atom), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->m, atom->m, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->x, atom->x, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->y, atom->y, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->z, atom->z, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->vx, atom->vx, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->vy, atom->vy, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->vz, atom->vz, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->fx, atom->fx, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->fy, atom->fy, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->fz, atom->fz, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->r, atom->r, N * sizeof(real), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->cells, atom->cells, nrCells*nrCells*nrCells * sizeof(int), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->particles, atom->particles, N * sizeof(int), cudaMemcpyDeviceToHost, 0));
    CHECK(cudaMemcpyAsync(atomCPU->I, atom->I, N * 9 * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for inertia tensors
    CHECK(cudaMemcpyAsync(atomCPU->tx, atom->tx, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for torques
    CHECK(cudaMemcpyAsync(atomCPU->ty, atom->ty, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for torques
    CHECK(cudaMemcpyAsync(atomCPU->tz, atom->tz, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for torques
    CHECK(cudaMemcpyAsync(atomCPU->wx, atom->wx, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for angular velocities
    CHECK(cudaMemcpyAsync(atomCPU->wy, atom->wy, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for angular velocities
    CHECK(cudaMemcpyAsync(atomCPU->wz, atom->wz, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for angular velocities
    CHECK(cudaMemcpyAsync(atomCPU->qx, atom->qx, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for quaternions
    CHECK(cudaMemcpyAsync(atomCPU->qy, atom->qy, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for quaternions
    CHECK(cudaMemcpyAsync(atomCPU->qz, atom->qz, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for quaternions
    CHECK(cudaMemcpyAsync(atomCPU->qw, atom->qw, N * sizeof(real), cudaMemcpyDeviceToHost, 0)); // Added for quaternions
}

// Free memory
void freeMemory(Atom* atom){
    cudaFree(atom->m);
    cudaFree(atom->x);
    cudaFree(atom->y);
    cudaFree(atom->z);
    cudaFree(atom->vx);
    cudaFree(atom->vy);
    cudaFree(atom->vz);
    cudaFree(atom->fx);
    cudaFree(atom->fy);
    cudaFree(atom->fz);
    cudaFree(atom->r);
    cudaFree(atom->cells);
    cudaFree(atom->particles);
    cudaFree(atom->I); // Added for inertia tensors
    cudaFree(atom->tx); // Added for torques
    cudaFree(atom->ty); // Added for torques
    cudaFree(atom->tz); // Added for torques
    cudaFree(atom->wx); // Added for angular velocities
    cudaFree(atom->wy); // Added for angular velocities
    cudaFree(atom->wz); // Added for angular velocities
    cudaFree(atom->qx); // Added for quaternions
    cudaFree(atom->qy); // Added for quaternions
    cudaFree(atom->qz); // Added for quaternions
    cudaFree(atom->qw); // Added for quaternions
}

// Setting up the domain and particles
void initAtom(const char* filePath, Atom* atom, int nrCells, real r){
    std::string line;
    std::ifstream filestr;
    filestr.open(filePath, std::ios::in);
    // Check if the file was opened successfully
    if (!filestr.is_open()) {
        std::cerr << "Failed to open the file: " << filePath << std::endl;
        return;
    }
    int numPoints = 0;
    while (std::getline(filestr, line))
    {
        if (line.find("POINTS") != std::string::npos) {
            std::istringstream iss(line);
            std::string keyword;

            iss >> keyword >> numPoints;
            initMemory(atom, numPoints, nrCells);
            atom->n = numPoints;
            for(int i = 0; i < nrCells * nrCells * nrCells; i++){
                atom->cells[i] = -1;
            }
            for (int i = 0; i < numPoints; ++i) {
                std::getline(filestr, line);
                std::istringstream pointIss(line);
                pointIss >> atom->x[i] >> atom->y[i] >> atom->z[i];
            }
        } else if(line.find("LOOKUP_TABLE") != std::string::npos ){
            for (int i = 0; i < numPoints; ++i) {
                std::getline(filestr, line);
                std::istringstream pointIss(line);
                pointIss >> atom->m[i];
            }
        }else if(line.find("VECTORS") != std::string::npos ){
            for (int i = 0; i < numPoints; ++i) {
                std::getline(filestr, line);
                std::istringstream pointIss(line);
                pointIss >> atom->vx[i] >> atom->vy[i] >> atom->vz[i];
            }
        }

    }
    filestr.close();

    //initial the rest variables
    for (int i = 0; i < numPoints; ++i) {
        atom->fx[i] = 0.0;
        atom->fy[i] = 0.0;
        atom->fz[i] = 0.0;
        atom->r[i] = r;

        atom->tx[i] = 0.0; // Initialize torques
        atom->ty[i] = 0.0; // Initialize torques
        atom->tz[i] = 0.0; // Initialize torques
        atom->wx[i] = 0.0; // Initialize angular velocities
        atom->wy[i] = 0.0; // Initialize angular velocities
        atom->wz[i] = 0.0; // Initialize angular velocities

        // Initialize the inertia tensor for each sphere
        real mass = atom->m[i];
        real radius = atom->r[i];
        real inertia = (2.0 / 5.0) * mass * radius * radius;
        atom->I[i * 9 + 0] = inertia; // Ixx
        atom->I[i * 9 + 4] = inertia; // Iyy
        atom->I[i * 9 + 8] = inertia; // Izz
        atom->I[i * 9 + 1] = atom->I[i * 9 + 2] = atom->I[i * 9 + 3] = 0.0;
        atom->I[i * 9 + 5] = atom->I[i * 9 + 6] = atom->I[i * 9 + 7] = 0.0;

        // Initialize quaternions to identity (0,0,0,1)
        atom->qx[i] = 0.0;
        atom->qy[i] = 0.0;
        atom->qz[i] = 0.0;
        atom->qw[i] = 1.0;
    }
}