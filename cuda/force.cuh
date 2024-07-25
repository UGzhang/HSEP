#include "Atom.cuh"
#include "surface.cuh"

__global__
void computeForce(Atom* atom, const real K, const real gamma, double domain_size, double cutoff, double g);
__device__
void collision(Atom* atom, int i, int j, double r, double sigma, double K, double gamma);
__global__
void wallCollisionDetectionSTL(Atom* atom, Mesh* mesh, double K_wall, double gamma_wall, real triangle_cutoff);
__device__
void wallCollisionSTL(Atom* atom, Mesh* mesh, real wall_x, real wall_y, real wall_z, int i, int t, double K_wall, double gamma_wall);

