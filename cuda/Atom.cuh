#pragma once
#ifdef USE_FP
typedef float real;
#else
typedef double real;
#endif

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cuda.h>



//Boltzmann constant not sure if still needed
const real K_B = 8.617343e-5;


struct Atom {
    int n; // number of atoms
    real* m; // masses
    real* x; // x positions
    real* y; // y positions
    real* z; // z positions
    real* vx; // x velocities
    real* vy; // y velocities
    real* vz; // z velocities
    real* fx; // x forces
    real* fy; // y forces
    real* fz; // z forces
    real* r; // radii
    int* cells; // cells
    int* particles; // particles
    real* I; // inertia tensors
    real* tx; // x torques
    real* ty; // y torques
    real* tz; // z torques
    real* wx; // x angular velocities
    real* wy; // y angular velocities
    real* wz; // z angular velocities
    real* qx; // quaternion x component
    real* qy; // quaternion y component
    real* qz; // quaternion z component
    real* qw; // quaternion w component
};

