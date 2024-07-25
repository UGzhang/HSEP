#pragma once
#ifdef USE_FP
typedef float real;
#else
typedef double real;
#endif
#include <string>

typedef double real;


struct Mesh{

    int num_faces;

    //vertex
    real* v1_x;
    real* v1_y;
    real* v1_z;

    real* v2_x;
    real* v2_y;
    real* v2_z;

    real* v3_x;
    real* v3_y;
    real* v3_z;

    //normal
    real *n_x;
    real *n_y;
    real *n_z;

};

void loadMesh(const std::string& filename, Mesh* mesh, const int domain_size);

void freeMesh(Mesh* mesh);