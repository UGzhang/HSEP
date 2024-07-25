#include "Atom.cuh"


__device__ void calculateTorque(Atom* atom, int i, real* force, real* contactPoint);
__device__ void normalizeQuaternion(real& qx, real& qy, real& qz, real& qw);
__device__ void updateQuaternion(real& qx, real& qy, real& qz, real& qw, real wx, real wy, real wz, real ts);
void quaternionToAxisAngle(double qw, double qx, double qy, double qz, double *axis, double *angle);
