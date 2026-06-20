// particle_kernels.cuh
// Velocity update kernel for particle system simulation using CUDA.
// Integrator: Velocity Verlet method.
__global__ void updateVelocity(
    double *particle_x_velocities, double *particle_y_velocities, double *particle_z_velocities,
    double *x_forces_old, double *y_forces_old, double *z_forces_old,
    double *x_forces_new, double *y_forces_new, double *z_forces_new,
    int num_particles,
    double time_step,
    double *mass)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_particles)
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
    int num_particles,
    double time_step,
    double *mass)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    { 
        double inv_mass = 1.0 / mass[idx];
        x[idx] += particle_x_velocities[idx] * time_step + 0.5 * x_forces[idx] * inv_mass * time_step * time_step;
        y[idx] += particle_y_velocities[idx] * time_step + 0.5 * y_forces[idx] * inv_mass * time_step * time_step;
        z[idx] += particle_z_velocities[idx] * time_step + 0.5 * z_forces[idx] * inv_mass * time_step * time_step;
    }
}

__global__ void resetForces(
    double *x_forces, double *y_forces, double *z_forces,
    double *tx_forces, double *ty_forces, double *tz_forces,
    
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
