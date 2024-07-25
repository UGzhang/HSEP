#include "quaternions.cuh"
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

// Calculate torque due to a collision
__device__
void calculateTorque(Atom* atom, int i, real* force, real* contactPoint){
    real* x = atom->x;
    real* y = atom->y;
    real* z = atom->z;

    real r[3] = {contactPoint[0] - x[i], contactPoint[1] - y[i], contactPoint[2] - z[i]};
    real torque[3];
   // printf("\nTorques are: %f,  %f,  %f\n", force[0], force[1], force[2]);

    torque[0] = r[1] * force[2] - r[2] * force[1];
    torque[1] = r[2] * force[0] - r[0] * force[2];
    torque[2] = r[0] * force[1] - r[1] * force[0];
    //printf("\nTorques are: %f,  %f,  %f\n", torque[0], torque[1], torque[2]);

    // Add the calculated torque to the atom's torques
    atomicAdd(&atom->tx[i], torque[0]);
    atomicAdd(&atom->ty[i], torque[1]);
    atomicAdd(&atom->tz[i], torque[2]);
}

// Normalize quaternion
__device__
void normalizeQuaternion(real& qx, real& qy, real& qz, real& qw) {
    real norm = sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (norm < 1e-10 || isnan(norm)) {
        // Handle the case where the quaternion is too small or NaN
        qx = 0.0;
        qy = 0.0;
        qz = 0.0;
        qw = 1.0;
    } else {
        qx /= norm;
        qy /= norm;
        qz /= norm;
        qw /= norm;
    }
    // // Ensure consistent representation: enforce qw >= 0
    // if (qw < 0.0) {
    //     qx = -qx;
    //     qy = -qy;
    //     qz = -qz;
    //     qw = -qw;
    // }
}

// Update quaternion based on angular velocities
__device__
void updateQuaternion(real& qx, real& qy, real& qz, real& qw, real wx, real wy, real wz, real ts) {
    // Convert angular velocity to a quaternion
    real half_ts = 0.5 * ts;
    real norm = sqrt(wx * wx + wy * wy + wz * wz);
    real theta = norm * half_ts;
    
    if (theta > 1e-10) {
        real s = sin(theta) / norm;
        real wqx = s * wx;
        real wqy = s * wy;
        real wqz = s * wz;
        real wqw = cos(theta);

        // Hamilton product of (qx, qy, qz, qw) and (wqx, wqy, wqz, wqw)
        real new_qx = qw * wqx + qx * wqw + qy * wqz - qz * wqy;
        real new_qy = qw * wqy - qx * wqz + qy * wqw + qz * wqx;
        real new_qz = qw * wqz + qx * wqy - qy * wqx + qz * wqw;
        real new_qw = qw * wqw - qx * wqx - qy * wqy - qz * wqz;

        qx += new_qx;
        qy += new_qy;
        qz += new_qz;
        qw += new_qw;

        normalizeQuaternion(qx, qy, qz, qw);
    }
}
// Function to convert quaternion to axis-angle
void quaternionToAxisAngle(double qw, double qx, double qy, double qz, double *axis, double *angle) {
    // Calculate the angle
    *angle = 2.0 * acos(qw);
    
    // Calculate the scale factor
    double s = sqrt(1.0 - qw * qw);
    
    // Avoid division by zero
    if (s < 0.001) {
        axis[0] = qx;
        axis[1] = qy;
        axis[2] = qz;
    } else {
        axis[0] = qx / s;
        axis[1] = qy / s;
        axis[2] = qz / s;
    }
}