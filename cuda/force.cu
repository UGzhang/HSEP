#include "Atom.cuh"
#include "surface.cuh"
#include "force.cuh"
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "quaternions.cuh"


// Collision handling for inter-particle collisions
__device__
void collision(Atom* atom, int i, int j, double r, double sigma, double K, double gamma){
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;

    // Distance between particles
    double x_ij = x[i] - x[j];
    double y_ij = y[i] - y[j];
    double z_ij = z[i] - z[j];
    double dist = sqrt(x_ij * x_ij + y_ij * y_ij + z_ij * z_ij);

    // eps etc equivalent to spring dampener model form lecture slides
    double eps_x = x_ij / dist * (sigma - dist);
    double eps_y = y_ij / dist * (sigma - dist);
    double eps_z = z_ij / dist * (sigma - dist);

    double x_norm = (x_ij / dist);
    double y_norm = (y_ij / dist);
    double z_norm = (z_ij / dist);

    double vx = atom->vx[i] - atom->vx[j];
    double vy = atom->vy[i] - atom->vy[j];
    double vz = atom->vz[i] - atom->vz[j];

    double eps_dif_x = -x_norm * (x_norm * vx);
    double eps_dif_y = -y_norm * (y_norm * vy);
    double eps_dif_z = -z_norm * (z_norm * vz);

    //update force for both particles
    atomicAdd(&fx[i], K * eps_x + gamma * eps_dif_x);
    atomicAdd(&fy[i], K * eps_y + gamma * eps_dif_y);
    atomicAdd(&fz[i], K * eps_z + gamma * eps_dif_z);

    atomicAdd(&fx[j], -(K * eps_x + gamma * eps_dif_x));
    atomicAdd(&fy[j], -(K * eps_y + gamma * eps_dif_y));
    atomicAdd(&fz[j], -(K * eps_z + gamma * eps_dif_z));

    // Calculate the contact point (midpoint of collision)
    real contactPoint[3] = {
        (x[i] + x[j]) / 2.0,
        (y[i] + y[j]) / 2.0,
        (z[i] + z[j]) / 2.0
    };

    // Calculate torques
    real force[3] = {K * eps_x + gamma * eps_dif_x, K * eps_y + gamma * eps_dif_y, K * eps_z + gamma * eps_dif_z};
    calculateTorque(atom, i, force, contactPoint);
    calculateTorque(atom, j, force, contactPoint);
}

// General calculation of force on particle
__global__
void computeForce(Atom* atom, const real K, const real gamma, double domain_size, double cutoff, double g){
    int N = atom->n;
    real *m = atom->m;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;
    int* cells = atom->cells;
    int* particles = atom->particles;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;

    // Initialize all forces, in y direction, include gravity
    for (int n = index; n < N; n += gridStride)
    {
        fx[n] = fz[n] = 0.0;
        //enforce gravity
        fy[n] = -m[n] * g;
    }

    // Cutoff is used to determine grid size
    int nrCells = (int)(domain_size / cutoff);
    for (int i = index; i < N; i += gridStride)
    {
        // Iterate through all neighbouring cells and own
        for(int x_dev = -1; x_dev <= 1; x_dev++){ // Corrected range
            for(int y_dev = -1; y_dev <= 1; y_dev++){ // Corrected range
                for(int z_dev = -1; z_dev <= 1; z_dev++){ // Corrected range
                    // Getting the actual idx of the current grid
                    int idx = int(x[i] / cutoff) + x_dev;
                    int idy = int(y[i] / cutoff) + y_dev;
                    int idz = int(z[i] / cutoff) + z_dev;
                    idx -= nrCells * (int)std::floor(idx / (double)nrCells);
                    idy -= nrCells * (int)std::floor(idy / (double)nrCells);
                    idz -= nrCells * (int)std::floor(idz / (double)nrCells);

                    // Calculate idx of cell
                    int j = cells[idx + nrCells * (idy + nrCells * idz)];
                    // Compute interparticular forces by iterating through all neighbours
                    while(j != -1){
                        if(j > i){
                            // Calculate distance between i and j
                            double x_ij = x[i] - x[j];
                            double y_ij = y[i] - y[j];
                            double z_ij = z[i] - z[j];
                            double r = sqrt(x_ij * x_ij + y_ij * y_ij + z_ij * z_ij);

                            // Collision detection:
                            double sigma = atom->r[i] + atom->r[j];
                            if(r <= sigma){
                                collision(atom, i, j, r, sigma, K, gamma);
                            }
                        }
                        // Continue in this list
                        j = particles[j];
                    }
                }
            }
        }
    }
}

//Called when collision ensured. Force implementation of collision
__device__
void wallCollisionSTL(Atom* atom, Mesh* mesh, real wall_x, real wall_y, real wall_z, int i, int t, double K_wall, double gamma_wall, real distance){
    //printf("Wall collision happening!");
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;

    //Collision detection: if distance is smaller than radius->Wall collision
    double sigma = atom->r[i];

    //Get distance between Wall and particle
    double x_ij = x[i] - wall_x;
    double y_ij = y[i] - wall_y;
    double z_ij = z[i] - wall_z;
    //apply l1 norm to also correctly calculate the corners
    double r = distance;
    //eps etc equivalent to spring dampener model form lecture slides
    double eps_x = x_ij/r * (sigma - r);
    double eps_y = y_ij/r * (sigma - r);
    double eps_z = z_ij/r * (sigma - r);

    double x_norm = (x_ij/r);
    double y_norm = (y_ij/r);
    double z_norm = (z_ij/r);

    double vx = atom->vx[i];
    double vy = atom->vy[i];
    double vz = atom->vz[i];

    double eps_dif_x = -x_norm * (x_norm * vx);
    double eps_dif_y = -y_norm * (y_norm * vy);
    double eps_dif_z = -z_norm * (z_norm * vz);

    atomicAdd(&fx[i], K_wall * eps_x + gamma_wall * eps_dif_x);
    atomicAdd(&fy[i], K_wall * eps_y + gamma_wall * eps_dif_y);
    atomicAdd(&fz[i], K_wall * eps_z + gamma_wall * eps_dif_z);
}

//Calculate global collisions between particles and wall triangles
__global__
void wallCollisionDetectionSTL(Atom* atom, Mesh* mesh, double K_wall, double gamma_wall, real triangle_cutoff){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;

    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;

    int N = atom->n;

    int num_faces = mesh->num_faces;

    //vertex
    real* v1_x = mesh->v1_x;
    real* v1_y = mesh->v1_y;
    real* v1_z = mesh->v1_z;

    real* v2_x = mesh->v2_x;
    real* v2_y = mesh->v2_y;
    real* v2_z = mesh->v2_z;

    real* v3_x = mesh->v3_x;
    real* v3_y = mesh->v3_y;
    real* v3_z = mesh->v3_z;

    //normal
    real *n_x = mesh->n_x;
    real *n_y = mesh->n_y;
    real *n_z = mesh->n_z;

    //Iterate through all particles
    for (int p = index; p < N; p+=gridStride){
        //Iterate through all triangles
        for(int t = 0; t < num_faces; t++){
            //calculate distance between particle and a point of the triangle, when larger cutoff skip this triangle
            real distance = sqrt((v1_x[t] - x[p])*(v1_x[t] - x[p]) + (v1_y[t] - y[p])*(v1_y[t] - y[p]) + (v1_z[t] - z[p])*(v1_z[t] - z[p]));
            if(distance > triangle_cutoff)
                continue;
            //calculate distance between plane of triangle and point
            //Equation for plane: Ax + By + Cz = D where A,B,C are x,y,z of normal vector
            //To get D use one of the triangle points to calculate D
            real D = n_x[t] * v1_x[t] + n_y[t] * v1_y[t] + n_z[t] * v1_z[t];
            //calculate parameter var for which normal from point intersects plane:
            real numerator = -(n_x[t] * x[p] + n_y[t] * y[p] + n_z[t] * z[p] - D);
            real denominator = n_x[t] * n_x[t] + n_y[t] * n_y[t] + n_z[t] * n_z[t];
            real var = numerator / denominator;
            //calc intersection points:
            real int_x = x[p] + n_x[t] * var;
            real int_y = y[p] + n_y[t] * var;
            real int_z = z[p] + n_z[t] * var;

            //Compute if intersection point is inside triangle
            // Compute vectors
            float3 v0 = make_float3(v3_x[t] - v1_x[t], v3_y[t] - v1_y[t], v3_z[t] - v1_z[t]);
            float3 v1 = make_float3(v2_x[t] - v1_x[t], v2_y[t] - v1_y[t], v2_z[t] - v1_z[t]);
            float3 v2 = make_float3(x[p] - v1_x[t], y[p] - v1_y[t], z[p] - v1_z[t]);

            // Compute dot products
            float dot00 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
            float dot01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
            float dot02 = v0.x * v2.x + v0.y * v2.y + v0.z * v2.z;
            float dot11 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
            float dot12 = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
            // Compute barycentric coordinates
            float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

            // Check if point is in triangle, if these conditions are met it is in the triangle
            if((u >= 0) && (v >= 0) && (u + v < 1)){
                //calculate distance between point and triangle
                distance = sqrt((int_x - x[p])*(int_x - x[p]) + (int_y - y[p])*(int_y - y[p]) + (int_z - z[p])*(int_z - z[p]));

                //if point is in collision range apply forces
                if (distance <= atom->r[p])
                    wallCollisionSTL(atom, mesh, int_x, int_y, int_z, p, t, K_wall, gamma_wall, distance);
            }
        }
    }
}

