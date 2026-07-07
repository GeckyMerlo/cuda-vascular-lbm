#pragma once

#include "particle_data.cuh"
#include "../space/space_data.cuh"

#include <math.h>

// particle_kernels.cuh
// Velocity update kernel for particle system simulation using CUDA.
// Integrator: Velocity Verlet method.
__device__ int nearestCellIdFromParticlePosition(double x, double y, double z, int nx, int ny, int nz)
{
    int ix = static_cast<int>(floor(x + 0.5));
    int iy = static_cast<int>(floor(y + 0.5));
    int iz = static_cast<int>(floor(z + 0.5));

    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
        return -1;
    }

    return iz * nx * ny + iy * nx + ix;
}

__global__ void initializeInactive(
    int* active,
    int* age,
    int* contact_count,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    active[idx] = 0;
    age[idx] = 0;
    contact_count[idx] = 0;
}

__global__ void updateVelocity(
    double *particle_x_velocities, double *particle_y_velocities, double *particle_z_velocities,
    double *x_forces_old, double *y_forces_old, double *z_forces_old,
    double *x_forces_new, double *y_forces_new, double *z_forces_new,
    const int* active,
    int num_particles,
    double time_step,
    double *mass)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_particles && active[idx])
    {
        particle_x_velocities[idx] += 0.5 * (x_forces_old[idx] + x_forces_new[idx]) / mass[idx] * time_step;
        particle_y_velocities[idx] += 0.5 * (y_forces_old[idx] + y_forces_new[idx]) / mass[idx] * time_step;
        particle_z_velocities[idx] += 0.5 * (z_forces_old[idx] + z_forces_new[idx]) / mass[idx] * time_step;
    }
}

__global__ void updatePosition(
    double *x, double *y, double *z,
    double *particle_x_velocities, double *particle_y_velocities, double *particle_z_velocities,
    double *x_forces, double *y_forces, double *z_forces,
    int* age,
    const int* active,
    int num_particles,
    double time_step,
    double *mass)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles && active[idx])
    { 
        double inv_mass = 1.0 / mass[idx];
        x[idx] += particle_x_velocities[idx] * time_step + 0.5 * x_forces[idx] * inv_mass * time_step * time_step;
        y[idx] += particle_y_velocities[idx] * time_step + 0.5 * y_forces[idx] * inv_mass * time_step * time_step;
        z[idx] += particle_z_velocities[idx] * time_step + 0.5 * z_forces[idx] * inv_mass * time_step * time_step;
        age[idx] += 1;
    }
}

__global__ void clampForces_kernel(
    double* x_forces,
    double* y_forces,
    double* z_forces,
    const int* active,
    int num_particles,
    double max_force)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !active[idx]) return;

    double fx = x_forces[idx];
    double fy = y_forces[idx];
    double fz = z_forces[idx];

    if (!isfinite(fx) || !isfinite(fy) || !isfinite(fz)) {
        x_forces[idx] = 0.0;
        y_forces[idx] = 0.0;
        z_forces[idx] = 0.0;
        return;
    }

    if (max_force <= 0.0) return;

    double mag = sqrt(fx * fx + fy * fy + fz * fz);
    if (mag > max_force && mag > 1e-16) {
        double scale = max_force / mag;
        x_forces[idx] = fx * scale;
        y_forces[idx] = fy * scale;
        z_forces[idx] = fz * scale;
    }
}

__global__ void clampVelocities_kernel(
    double* x,
    double* y,
    double* z,
    double* vx,
    double* vy,
    double* vz,
    int* active,
    int num_particles,
    double max_speed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !active[idx]) return;

    if (!isfinite(x[idx]) || !isfinite(y[idx]) || !isfinite(z[idx]) ||
        !isfinite(vx[idx]) || !isfinite(vy[idx]) || !isfinite(vz[idx])) {
        active[idx] = 0;
        vx[idx] = 0.0;
        vy[idx] = 0.0;
        vz[idx] = 0.0;
        return;
    }

    if (max_speed <= 0.0) return;

    double speed = sqrt(vx[idx] * vx[idx] + vy[idx] * vy[idx] + vz[idx] * vz[idx]);
    if (speed > max_speed && speed > 1e-16) {
        double scale = max_speed / speed;
        vx[idx] *= scale;
        vy[idx] *= scale;
        vz[idx] *= scale;
    }
}

__global__ void resetForces(
    double *x_forces, double *y_forces, double *z_forces,
    double *tx_forces, double *ty_forces, double *tz_forces,
    int* contact_count,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    {
        x_forces[idx] = 0.0;
        y_forces[idx] = 0.0;
        z_forces[idx] = 0.0;

        tx_forces[idx] = 0.0;
        ty_forces[idx] = 0.0;
        tz_forces[idx] = 0.0;
        contact_count[idx] = 0;
    }
}

__global__ void swapForces(
    double *x_forces_old, double *y_forces_old, double *z_forces_old,
    double *x_forces_new, double *y_forces_new, double *z_forces_new,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    {
        x_forces_old[idx] = x_forces_new[idx];
        y_forces_old[idx] = y_forces_new[idx];
        z_forces_old[idx] = z_forces_new[idx];

    }
}

__global__ void updateAngularVelocity(
    double* wx, double* wy, double* wz,
    const double* tx, const double* ty, const double* tz,
    const double* inertia_x, const double* inertia_y, const double* inertia_z,
    const int* active,
    int num_particles,
    double time_step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !active[idx]) return;

    wx[idx] += (inertia_x[idx] > 0.0 ? tx[idx] / inertia_x[idx] : 0.0) * time_step;
    wy[idx] += (inertia_y[idx] > 0.0 ? ty[idx] / inertia_y[idx] : 0.0) * time_step;
    wz[idx] += (inertia_z[idx] > 0.0 ? tz[idx] / inertia_z[idx] : 0.0) * time_step;
}

__global__ void updateOrientation(
    double* qw, double* qx, double* qy, double* qz,
    const double* wx, const double* wy, const double* wz,
    const int* active,
    int num_particles,
    double time_step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !active[idx]) return;

    double w0 = qw[idx];
    double x0 = qx[idx];
    double y0 = qy[idx];
    double z0 = qz[idx];
    double ox = wx[idx];
    double oy = wy[idx];
    double oz = wz[idx];

    double dq_w = -0.5 * (x0 * ox + y0 * oy + z0 * oz);
    double dq_x =  0.5 * (w0 * ox + y0 * oz - z0 * oy);
    double dq_y =  0.5 * (w0 * oy + z0 * ox - x0 * oz);
    double dq_z =  0.5 * (w0 * oz + x0 * oy - y0 * ox);

    w0 += time_step * dq_w;
    x0 += time_step * dq_x;
    y0 += time_step * dq_y;
    z0 += time_step * dq_z;

    double norm = sqrt(w0 * w0 + x0 * x0 + y0 * y0 + z0 * z0);
    if (norm > 0.0) {
        double inv = 1.0 / norm;
        qw[idx] = w0 * inv;
        qx[idx] = x0 * inv;
        qy[idx] = y0 * inv;
        qz[idx] = z0 * inv;
    } else {
        qw[idx] = 1.0;
        qx[idx] = 0.0;
        qy[idx] = 0.0;
        qz[idx] = 0.0;
    }
}

__global__ void resetCellList(
    int* cell_head,
    int* particle_next,
    int* particle_cell_id,
    int num_cells,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_cells) {
        cell_head[idx] = -1;
    }

    if (idx < num_particles) {
        particle_next[idx] = -1;
        particle_cell_id[idx] = -1;
    }
}

__global__ void buildCellList(
    int* cell_head,
    int* particle_next,
    int* particle_cell_id,
    const int* active,
    const double* x,
    const double* y,
    const double* z,
    const CellType* cell_type,
    int num_particles,
    int nx,
    int ny,
    int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !active[idx]) return;

    int cell_id = nearestCellIdFromParticlePosition(x[idx], y[idx], z[idx], nx, ny, nz);
    if (cell_id < 0 || cell_type[cell_id] == SOLID) {
        particle_cell_id[idx] = -1;
        return;
    }

    particle_cell_id[idx] = cell_id;
    int previous = atomicExch(&cell_head[cell_id], idx);
    particle_next[idx] = previous;
}

__global__ void deactivateExited(
    int* active,
    int* exited_total,
    const double* x,
    const double* y,
    const double* z,
    const CellType* cell_type,
    int num_particles,
    int nx,
    int ny,
    int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !active[idx]) return;

    if (!isfinite(x[idx]) || !isfinite(y[idx]) || !isfinite(z[idx])) {
        active[idx] = 0;
        atomicAdd(exited_total, 1);
        return;
    }

    int cell_id = nearestCellIdFromParticlePosition(x[idx], y[idx], z[idx], nx, ny, nz);
    if (cell_id < 0 || cell_type[cell_id] == OUTLET) {
        active[idx] = 0;
        atomicAdd(exited_total, 1);
    }
}
