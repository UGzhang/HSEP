#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "Atom.cuh"
#include "force.cuh"
#include "initialize.cuh"
#include "quaternions.cuh"
#include "integrate.cuh"
#include "utils.cuh"
#include "neighbour.cuh"
#include "surface.cuh"
//Adaptive timestep calculation
#include "timestep.cuh"


#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

int main(int argc, char **argv) {
    // Reading in everything from input or else use standard predefined values
    double RUNTIME = get_argval<double>(argv, argv + argc, "-time", 20.);//simulation time in s
    const double base_ts = get_argval<real>(argv, argv + argc, "-ts", 0.01);//initial timesteps size, adaptive timestep lowers from there
    const int domain_size = get_argval<int>(argv, argv + argc, "-nx", 12);//domain size in each direction
    const real g = get_argval<real>(argv, argv + argc, "-g", 0.981);//0.0981);//value for g
    const real r = get_argval<real>(argv, argv + argc, "-r", 0.1);//value for avg radius of particles
    const real K = get_argval<real>(argv, argv + argc, "-K", 50);//value for K in Force clalc (ref value 10)
    const real gamma = get_argval<real>(argv, argv + argc, "-gamma", 5.);//value for gamma in Force calc (ref value 0.1)
    const real K_wall = get_argval<real>(argv, argv + argc, "-K_wall", 800);//value for K in Force clalc (ref value 10)
    const real gamma_wall = get_argval<real>(argv, argv + argc, "-gamma_wall", 60);//value for gamma in Force calc (ref value 0.1)
    const real cutoff = get_argval<real>(argv, argv + argc, "-c", r*2); //Define the cutoff value which governs the size of grid cells
    const int verbosity = get_argval<int>(argv, argv + argc, "-verbosity", 1); //Defines the output given by the program. 0 is none, 1 olnly time step information, 3 is for all particels vel position
    const real triangle_cutoff = get_argval<real>(argv, argv + argc, "-c", 4); //Defines the range in which triangle particle interactions are ignored if any triangle point is more than that value away
    auto filename = "particles.vtk";
    auto stl_file = "sphere.stl";

    // Get CUDA device and determine optimal nr of blocks (Nr SMs *2)
    cudaDeviceProp deviceProp;
    cudaSetDevice(0);
    cudaCheckError();
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaCheckError();
    size_t threadsPerBlock = 128;
    size_t numberOfBlocks = deviceProp.multiProcessorCount * 2;

    // Define Interval for Console reports and intervals in which vtk is printed. both refer to the nr of timesteps
    const double reportInterval = 0.1;//in s
    const double printInterval = 0.05;//in s


    // Initializing Domain Atoms etc.
    int nrCells = static_cast<int>(domain_size / cutoff);
    Atom *atom{};
    Atom *atomCPU{};
    Mesh *mesh{};//STL

    //Additional variable definitions for loop printing
    double* ts;
    cudaMallocManaged(&ts, sizeof(double));
    *ts  = base_ts;
    double SIMTIME = 0.;
    double reportTime = 0.;
    double printTime = 0.;
    int printNumber = 0;

    cudaMallocManaged(&mesh, sizeof(Mesh));
    loadMesh(stl_file, mesh,domain_size);
    cudaMallocManaged(&atom, sizeof(Atom) * nrCells);
    cudaCheckError();
    cudaMallocManaged(&atomCPU, sizeof(Atom) * nrCells);
    cudaCheckError();
    initAtom(filename, atom, nrCells, r);
    initAtom(filename, atomCPU, nrCells, r);
    int N = atom->n;
    // Already load data to GPU you are sure is needed to avoid cache misses if possible
    prefetchGPU(atom, N, nrCells, 0);
    cudaCheckError();

    // Simulation Loop
    while(SIMTIME < RUNTIME){
        //reset timestep for new optimization step
        *ts = base_ts;
        //Calculate adequate time step
        adaptiveTimeStep<<<numberOfBlocks, threadsPerBlock>>>(atom, ts, K, gamma);
        cudaDeviceSynchronize();
        // Integration Step
        integrate<<<numberOfBlocks, threadsPerBlock>>>(atom, *ts, domain_size);
        cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();
        // Recalculating particle grid positions
        findNeighbor<<<numberOfBlocks, threadsPerBlock>>>(atom, nrCells, cutoff);
        cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();
        // Compute all forces acting on particles (Collisions etc.)
        computeForce<<<numberOfBlocks, threadsPerBlock>>>(atom, K, gamma, domain_size, cutoff, g);
        cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();
        //Compute wall collisions
        wallCollisionDetectionSTL<<<numberOfBlocks, threadsPerBlock>>>(atom, mesh, K_wall, gamma_wall, triangle_cutoff);
        cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();

        //increment simtime with time step
        SIMTIME+= *ts;

        // Calculate if there is a report interval to prefetch data to CPU for reporting
        if(SIMTIME >= reportTime || SIMTIME >= printTime){
            prefetchCPU(atom, atomCPU, N, nrCells);
            cudaCheckError();
        }

        //Console output
        if(SIMTIME >= reportTime) {
            if(verbosity != 0) {
                printf("\n\nTimestep: %f \t Stepsize: %f\n", SIMTIME, *ts);
                if(verbosity == 3) {
                    for (int k = 0; k < atomCPU->n; k++) {
                        printf("Location: x=%f  y =%f    z=%f \n", atomCPU->x[k], atomCPU->y[k], atomCPU->z[k]);
                        printf("Velocity: x=%f  y =%f    z=%f \n", atomCPU->vx[k], atomCPU->vy[k], atomCPU->vz[k]);
                        printf("Velocity Magnitude: v=%f\n",
                               sqrt(atomCPU->vx[k] * atomCPU->vx[k] + atomCPU->vy[k] * atomCPU->vy[k] +
                                    atomCPU->vz[k] * atomCPU->vz[k]));
                    }
                }
            }
            reportTime += reportInterval;
        }

        //Vtk file output
        if(SIMTIME >= printTime){
            std::ostringstream oss;
            oss << "./output/output_" << printNumber << ".vtp";
            exportVTKXML(oss.str().c_str(), atomCPU);
            printTime += printInterval;
            printNumber++;
        }
    }
    // Free allocated memory
    freeMemory(atom);
    freeMesh(mesh);
    cudaCheckError();
    return 0;
}