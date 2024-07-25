#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <vector>
#include <string>
#include "surface.cuh"

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


void loadMesh(const std::string& filename, Mesh* mesh, const int domain_size) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    std::string line;
    std::vector<real> normals;
    std::vector<real> vertices;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;

        if (word == "facet") {
            real nx, ny, nz;
            iss >> word >> nx >> ny >> nz; // skip "normal" and read the normal values
            //printf("Vertex: %f  %f  %f\n", nx, ny, nz);
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);
        } else if (word == "vertex") {
            real vx, vy, vz;
            iss >> vx >> vy >> vz;
            //printf("Vertex OG: %f  %f  %f\n", vx, vy, vz);
            vertices.push_back(vx);
            vertices.push_back(vy);
            vertices.push_back(vz);
        }
    }

    int num_faces = normals.size() / 3;
    mesh->num_faces = num_faces;

    real* n_x = new real[num_faces];
    real* n_y = new real[num_faces];
    real* n_z = new real[num_faces];

    real* v1_x = new real[num_faces];
    real* v1_y = new real[num_faces];
    real* v1_z = new real[num_faces];

    real* v2_x = new real[num_faces];
    real* v2_y = new real[num_faces];
    real* v2_z = new real[num_faces];

    real* v3_x = new real[num_faces];
    real* v3_y = new real[num_faces];
    real* v3_z = new real[num_faces];

    
    for (int i = 0; i < num_faces; ++i) {
        n_x[i] = normals[i * 3];
        n_y[i] = normals[i * 3 + 1];
        n_z[i] = normals[i * 3 + 2];

        v1_x[i] = vertices[i * 9];
        v1_y[i] = vertices[i * 9 + 1];
        v1_z[i] = vertices[i * 9 + 2];

        v2_x[i] = vertices[i * 9 + 3];
        v2_y[i] = vertices[i * 9 + 4];
        v2_z[i] = vertices[i * 9 + 5];

        v3_x[i] = vertices[i * 9 + 6];
        v3_y[i] = vertices[i * 9 + 7];
        v3_z[i] = vertices[i * 9 + 8];
        //printf("Vertex: %f  %f  %f\n", v1_x[i], v1_y[i], v1_z[i]);
        //printf("Vertex: %f  %f  %f\n", v2_x[i], v2_y[i], v2_z[i]);
        //printf("Vertex: %f  %f  %f\n", v3_x[i], v3_y[i], v3_z[i]);
    }


    cudaMallocManaged(&mesh->n_x, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->n_y, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->n_z, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v1_x, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v1_y, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v1_z, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v2_x, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v2_y, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v2_z, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v3_x, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v3_y, num_faces * sizeof(real));
    cudaMallocManaged(&mesh->v3_z, num_faces * sizeof(real));


    cudaMemcpy(mesh->n_x, n_x, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->n_x, n_x, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->n_y, n_y, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->n_z, n_z, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v1_x, v1_x, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v1_y, v1_y, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v1_z, v1_z, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v2_x, v2_x, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v2_y, v2_y, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v2_z, v2_z, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v3_x, v3_x, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v3_y, v3_y, num_faces * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->v3_z, v3_z, num_faces * sizeof(real), cudaMemcpyHostToDevice);

    printf("Vertex: %f  %f  %f\n", mesh->v1_x[0], mesh->v1_y[0], mesh->v1_z[0]);
    printf("Vertex: %f  %f  %f\n", mesh->v2_x[0], mesh->v2_y[0], mesh->v2_z[0]);
    printf("Vertex: %f  %f  %f\n", mesh->v3_x[0], mesh->v3_y[0], mesh->v3_z[0]);


    delete n_x;
    delete n_y;
    delete n_z;
    delete v1_x;
    delete v1_y;
    delete v1_z;
    delete v2_x;
    delete v2_y;
    delete v2_z;
    delete v3_x;
    delete v3_y;
    delete v3_z;

}




void freeMesh(Mesh* mesh){

    CHECK(cudaFree(mesh->n_x));
    CHECK(cudaFree(mesh->n_y));
    CHECK(cudaFree(mesh->n_z));
    CHECK(cudaFree(mesh->v1_x));
    CHECK(cudaFree(mesh->v1_y));
    CHECK(cudaFree(mesh->v1_z));
    CHECK(cudaFree(mesh->v2_x));
    CHECK(cudaFree(mesh->v2_y));
    CHECK(cudaFree(mesh->v2_z));
    CHECK(cudaFree(mesh->v3_x));
    CHECK(cudaFree(mesh->v3_y));
    CHECK(cudaFree(mesh->v3_z));


}