#include "Atom.cuh"
#include "quaternions.cuh"

// Integration step with explicit Euler scheme
__global__
void integrate(Atom* atom, real ts, double domain_size){
    int N = atom->n;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *m = atom->m;
    real *vx = atom->vx;
    real *vy = atom->vy;
    real *vz = atom->vz;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;
    real *tx = atom->tx;
    real *ty = atom->ty;
    real *tz = atom->tz;
    real *wx = atom->wx;
    real *wy = atom->wy;
    real *wz = atom->wz;
    real *I = atom->I;
    real *qx = atom->qx;
    real *qy = atom->qy;
    real *qz = atom->qz;
    real *qw = atom->qw;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;
    if(index >= N)
        return;
    for (int i = index; i < N; i += gridStride) {
        real mass = 1.0 / m[i];

        real ax = fx[i] * mass;
        real ay = fy[i] * mass;
        real az = fz[i] * mass;


        // k1
        real k1_vx = ax * ts;
        real k1_vy = ay * ts;
        real k1_vz = az * ts;
        real k1_x = vx[i] * ts;
        real k1_y = vy[i] * ts;
        real k1_z = vz[i] * ts;

        // k2
        real k2_vx = (fx[i] * mass) * ts;
        real k2_vy = (fy[i] * mass) * ts;
        real k2_vz = (fz[i] * mass) * ts;
        real k2_x = (vx[i] + 0.5 * k1_vx) * ts;
        real k2_y = (vy[i] + 0.5 * k1_vy) * ts;
        real k2_z = (vz[i] + 0.5 * k1_vz) * ts;

        // k3
        real k3_vx = (fx[i] * mass) * ts;
        real k3_vy = (fy[i] * mass) * ts;
        real k3_vz = (fz[i] * mass) * ts;
        real k3_x = (vx[i] + 0.5 * k2_vx) * ts;
        real k3_y = (vy[i] + 0.5 * k2_vy) * ts;
        real k3_z = (vz[i] + 0.5 * k2_vz) * ts;

        // k4
        real k4_vx = (fx[i] * mass) * ts;
        real k4_vy = (fy[i] * mass) * ts;
        real k4_vz = (fz[i] * mass) * ts;
        real k4_x = (vx[i] + k3_vx) * ts;
        real k4_y = (vy[i] + k3_vy) * ts;
        real k4_z = (vz[i] + k3_vz) * ts;

        // Combine steps
        x[i] += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0;
        y[i] += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6.0;
        z[i] += (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6.0;

        vx[i] += (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6.0;
        vy[i] += (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6.0;
        vz[i] += (k1_vz + 2 * k2_vz + 2 * k3_vz + k4_vz) / 6.0;

        // Calculate angular accelerations
        real invInertia[3];
        invInertia[0] = 1.0 / I[i * 9 + 0];
        invInertia[1] = 1.0 / I[i * 9 + 4];
        invInertia[2] = 1.0 / I[i * 9 + 8];

        real alpha_x = tx[i] * invInertia[0];
        real alpha_y = ty[i] * invInertia[1];
        real alpha_z = tz[i] * invInertia[2];

        // Update angular velocities
        wx[i] += alpha_x * ts; 
        wy[i] += alpha_y * ts;
        wz[i] += alpha_z * ts;

        // Update quaternions
        updateQuaternion(qx[i], qy[i], qz[i], qw[i], wx[i], wy[i], wz[i], ts);
    }
}