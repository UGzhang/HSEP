#include "Atom.cuh"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include "quaternions.cuh"
#include <iomanip>

// Simple output function which exports VTK files
void expertVTK(const char* filename, Atom* atom){
    int N = atom->n;

    std::ofstream os(filename);
    if (!os) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    os << "# vtk DataFile Version 4.0" << std::endl;
    os << "hesp visualization file" << std::endl;
    os << "ASCII" << std::endl;
    os<< "DATASET UNSTRUCTURED_GRID" << std::endl;
    os << "POINTS " << atom->n << " double" << std::endl;
    for(int i = 0; i < N; i++){
        os << atom->x[i] << " "<< atom->y[i]<< " " << atom->z[i] << std::endl;
    }
    os << "CELLS 0 0" << std::endl;
    os << "CELL_TYPES 0" << std::endl;
    os << "POINT_DATA " << N << std::endl;
    os << "SCALARS r double" << std::endl;
    os << "LOOKUP_TABLE default" << std::endl;
    for(int i = 0; i < N; i++){
        os << (double)atom->r[i] << std::endl;
    }
    os << "VECTORS v double" << std::endl;
    for(int i = 0; i < N; i++){
        os << atom->vx[i] << " "<< atom->vy[i]<< " " << atom->vz[i] << std::endl;
    }
    os << "VECTORS RotationAxis float" << std::endl; // Add quaternions to VTK output
    double axis[3],angle; //added for axis angle calculation
    for (int i = 0; i < N; i++)
    {
         // Convert quaternion to axis-angle
        quaternionToAxisAngle(atom->qw[i], atom->qx[i], atom->qy[i], atom->qz[i], axis, &angle);
        // Write axis-angle data to the VTK file
        os << axis[0] << " " << axis[1] << " " << axis[2] << " " << std::endl;
    }
    os << "SCALARS RotationAngle float 1" << std::endl; // Add quaternions to VTK output
    os << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < N; i++)
    {
         // Convert quaternion to axis-angle
        quaternionToAxisAngle(atom->qw[i], atom->qx[i], atom->qy[i], atom->qz[i], axis, &angle);
        // Write axis-angle data to the VTK file
        os << angle << std::endl;
    }
    os.close();
}
void exportVTKXML(const char* filename, Atom* atom) {
    int N = atom->n;

    std::ofstream os(filename);
    if (!os) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write the XML VTK header
    os << "<?xml version=\"1.0\"?>" << std::endl;
    os << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    os << "  <PolyData>" << std::endl;
    os << "    <Piece NumberOfPoints=\"" << N << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">" << std::endl;

    // Write point coordinates
    os << "      <Points>" << std::endl;
    os << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    for (int i = 0; i < N; i++) {
        os << std::setprecision(6) << atom->x[i] << " " << atom->y[i] << " " << atom->z[i] << std::endl;
    }
    os << "        </DataArray>" << std::endl;
    os << "      </Points>" << std::endl;

    // Write point data (radii)
    os << "      <PointData Scalars=\"radii\">" << std::endl;

    // Write radii
    os << "        <DataArray type=\"Float64\" Name=\"radii\" format=\"ascii\">" << std::endl;
    for (int i = 0; i < N; i++) {
        os << atom->r[i] << std::endl;
    }
    os << "        </DataArray>" << std::endl;

    // Write velocities
    os << "        <DataArray type=\"Float64\" Name=\"velocities\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    for (int i = 0; i < N; i++) {
        os << atom->vx[i] << " " << atom->vy[i] << " " << atom->vz[i] << std::endl;
    }
    os << "        </DataArray>" << std::endl;

    // Write angular velocities
    // os << "        <DataArray type=\"Float64\" Name=\"angular_velocity\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    // for (int i = 0; i < N; i++) {
    //     os << atom->wx[i] << " " << atom->wy[i] << " " << atom->wz[i] << std::endl;
    // }
    // os << "        </DataArray>" << std::endl;

    // Write quaternions
    os << "        <DataArray type=\"Float64\" Name=\"quaternion\" NumberOfComponents=\"4\" format=\"ascii\">" << std::endl;
    for (int i = 0; i < N; i++) {
        os << atom->qw[i] << " " << atom->qx[i] << " " << atom->qy[i] << " " << atom->qz[i] << std::endl;
    }
    os << "        </DataArray>" << std::endl;

    os << "      </PointData>" << std::endl;
    os << "    </Piece>" << std::endl;
    os << "  </PolyData>" << std::endl;
    os << "</VTKFile>" << std::endl;

    os.close();
}
