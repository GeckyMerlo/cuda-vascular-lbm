// particle_system.cpp
#include "particle_system.hpp"
#include "particle_kernels.cuh"

#include <cuda_runtime.h>

ParticleSystem::ParticleSystem()
{
    d_particles = {};
    d_cell_list = {};
}

ParticleSystem::~ParticleSystem()
{
    free();
}

void ParticleSystem::allocate(int n)
{
    allocate(n, 0);
}

void ParticleSystem::allocate(int n, int num_cells)
{
    free();

    if (n <= 0) {
        d_particles.n = 0;
        return;
    }

    d_particles.n = n;

    cudaMalloc(reinterpret_cast<void**>(&d_particles.x), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.y), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.z), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.vx), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.vy), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.vz), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.fx), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.fy), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.fz), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.fx_old), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.fy_old), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.fz_old), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.mass), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.inertia_x), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.inertia_y), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.inertia_z), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.radius), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.shape), n * sizeof(ParticleShape));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.species), n * sizeof(ParticleSpecies));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.active), n * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.age), n * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.contact_count), n * sizeof(int));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.wx), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.wy), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.wz), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.tx), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.ty), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.tz), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.qw), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.qx), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.qy), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.qz), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.a), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.b), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.c), n * sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&d_particles.next_slot), sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.injected_total), sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.dropped_total), sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.exited_total), sizeof(int));

    cudaMemset(d_particles.x, 0, n * sizeof(double));
    cudaMemset(d_particles.y, 0, n * sizeof(double));
    cudaMemset(d_particles.z, 0, n * sizeof(double));
    cudaMemset(d_particles.vx, 0, n * sizeof(double));
    cudaMemset(d_particles.vy, 0, n * sizeof(double));
    cudaMemset(d_particles.vz, 0, n * sizeof(double));
    cudaMemset(d_particles.fx, 0, n * sizeof(double));
    cudaMemset(d_particles.fy, 0, n * sizeof(double));
    cudaMemset(d_particles.fz, 0, n * sizeof(double));
    cudaMemset(d_particles.fx_old, 0, n * sizeof(double));
    cudaMemset(d_particles.fy_old, 0, n * sizeof(double));
    cudaMemset(d_particles.fz_old, 0, n * sizeof(double));
    cudaMemset(d_particles.tx, 0, n * sizeof(double));
    cudaMemset(d_particles.ty, 0, n * sizeof(double));
    cudaMemset(d_particles.tz, 0, n * sizeof(double));

    resetCounters();

    int block = 256;
    int grid = (n + block - 1) / block;
    initializeInactive<<<grid, block>>>(
        d_particles.active,
        d_particles.age,
        d_particles.contact_count,
        n
    );

    if (num_cells > 0) {
        d_cell_list.num_cells = num_cells;
        cudaMalloc(reinterpret_cast<void**>(&d_cell_list.cell_head), num_cells * sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&d_cell_list.particle_next), n * sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&d_cell_list.particle_cell_id), n * sizeof(int));
        resetCellList();
    }
}

void ParticleSystem::initialize(ParticleData& h_particles)
{
    cudaMemcpy(d_particles.x, h_particles.x, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.y, h_particles.y, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.z, h_particles.z, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.vx, h_particles.vx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vy, h_particles.vy, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vz, h_particles.vz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.fx, h_particles.fx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.fy, h_particles.fy, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.fz, h_particles.fz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.mass, h_particles.mass, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.inertia_x, h_particles.inertia_x, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.inertia_y, h_particles.inertia_y, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.inertia_z, h_particles.inertia_z, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.radius, h_particles.radius, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.shape, h_particles.shape, d_particles.n * sizeof(ParticleShape), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.species, h_particles.species, d_particles.n * sizeof(ParticleSpecies), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.active, h_particles.active, d_particles.n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.age, h_particles.age, d_particles.n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.contact_count, h_particles.contact_count, d_particles.n * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.wx, h_particles.wx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.wy, h_particles.wy, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.wz, h_particles.wz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.tx, h_particles.tx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.ty, h_particles.ty, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.tz, h_particles.tz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.qw, h_particles.qw, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.qx, h_particles.qx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.qy, h_particles.qy, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.qz, h_particles.qz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.a, h_particles.a, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.b, h_particles.b, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.c, h_particles.c, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.fx_old, h_particles.fx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.fy_old, h_particles.fy, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.fz_old, h_particles.fz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
}

void ParticleSystem::free()
{
    cudaFree(d_cell_list.cell_head);
    cudaFree(d_cell_list.particle_next);
    cudaFree(d_cell_list.particle_cell_id);
    d_cell_list = {};

    cudaFree(d_particles.x);
    cudaFree(d_particles.y);
    cudaFree(d_particles.z);
    cudaFree(d_particles.vx);
    cudaFree(d_particles.vy);
    cudaFree(d_particles.vz);
    cudaFree(d_particles.fx);
    cudaFree(d_particles.fy);
    cudaFree(d_particles.fz);
    cudaFree(d_particles.fx_old);
    cudaFree(d_particles.fy_old);
    cudaFree(d_particles.fz_old);
    cudaFree(d_particles.mass);
    cudaFree(d_particles.inertia_x);
    cudaFree(d_particles.inertia_y);
    cudaFree(d_particles.inertia_z);
    cudaFree(d_particles.radius);
    cudaFree(d_particles.shape);
    cudaFree(d_particles.species);
    cudaFree(d_particles.active);
    cudaFree(d_particles.age);
    cudaFree(d_particles.contact_count);
    cudaFree(d_particles.qw);
    cudaFree(d_particles.qx);
    cudaFree(d_particles.qy);
    cudaFree(d_particles.qz);
    cudaFree(d_particles.wx);
    cudaFree(d_particles.wy);
    cudaFree(d_particles.wz);
    cudaFree(d_particles.tx);
    cudaFree(d_particles.ty);
    cudaFree(d_particles.tz);
    cudaFree(d_particles.a);
    cudaFree(d_particles.b);
    cudaFree(d_particles.c);
    cudaFree(d_particles.next_slot);
    cudaFree(d_particles.injected_total);
    cudaFree(d_particles.dropped_total);
    cudaFree(d_particles.exited_total);

    d_particles = {};
}

void ParticleSystem::resetCounters()
{
    if (d_particles.next_slot == nullptr) {
        return;
    }

    cudaMemset(d_particles.next_slot, 0, sizeof(int));
    cudaMemset(d_particles.injected_total, 0, sizeof(int));
    cudaMemset(d_particles.dropped_total, 0, sizeof(int));
    cudaMemset(d_particles.exited_total, 0, sizeof(int));
}

void ParticleSystem::resetForces()
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::resetForces<<<grid, block>>>(
        d_particles.fx, d_particles.fy, d_particles.fz,
        d_particles.tx, d_particles.ty, d_particles.tz,
        d_particles.contact_count,
        d_particles.n
    );
}

void ParticleSystem::updateVelocity(double dt)
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::updateVelocity<<<grid, block>>>(
        d_particles.vx, d_particles.vy, d_particles.vz,
        d_particles.fx_old, d_particles.fy_old, d_particles.fz_old,
        d_particles.fx, d_particles.fy, d_particles.fz,
        d_particles.active,
        d_particles.n,
        dt,
        d_particles.mass
    );
}

void ParticleSystem::updatePosition(double dt)
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::updatePosition<<<grid, block>>>(
        d_particles.x, d_particles.y, d_particles.z,
        d_particles.vx, d_particles.vy, d_particles.vz,
        d_particles.fx, d_particles.fy, d_particles.fz,
        d_particles.age,
        d_particles.active,
        d_particles.n,
        dt,
        d_particles.mass
    );
}

void ParticleSystem::clampForces(double max_force)
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::clampForces_kernel<<<grid, block>>>(
        d_particles.fx,
        d_particles.fy,
        d_particles.fz,
        d_particles.active,
        d_particles.n,
        max_force
    );
}

void ParticleSystem::clampVelocities(double max_speed)
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::clampVelocities_kernel<<<grid, block>>>(
        d_particles.x,
        d_particles.y,
        d_particles.z,
        d_particles.vx,
        d_particles.vy,
        d_particles.vz,
        d_particles.active,
        d_particles.n,
        max_speed
    );
}

void ParticleSystem::updateAngularVelocity(double dt)
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::updateAngularVelocity<<<grid, block>>>(
        d_particles.wx, d_particles.wy, d_particles.wz,
        d_particles.tx, d_particles.ty, d_particles.tz,
        d_particles.inertia_x, d_particles.inertia_y, d_particles.inertia_z,
        d_particles.active,
        d_particles.n,
        dt
    );
}

void ParticleSystem::updateOrientation(double dt)
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::updateOrientation<<<grid, block>>>(
        d_particles.qw, d_particles.qx, d_particles.qy, d_particles.qz,
        d_particles.wx, d_particles.wy, d_particles.wz,
        d_particles.active,
        d_particles.n,
        dt
    );
}

void ParticleSystem::swapForces()
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::swapForces<<<grid, block>>>(
        d_particles.fx_old, d_particles.fy_old, d_particles.fz_old,
        d_particles.fx, d_particles.fy, d_particles.fz,
        d_particles.n
    );
}

void ParticleSystem::deactivateExited(const CellType* d_cell_type, int nx, int ny, int nz)
{
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::deactivateExited<<<grid, block>>>(
        d_particles.active,
        d_particles.exited_total,
        d_particles.x,
        d_particles.y,
        d_particles.z,
        d_cell_type,
        d_particles.n,
        nx,
        ny,
        nz
    );
}

void ParticleSystem::resetCellList()
{
    if (d_cell_list.cell_head == nullptr || d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int max_items = d_cell_list.num_cells > d_particles.n ? d_cell_list.num_cells : d_particles.n;
    int grid = (max_items + block - 1) / block;

    ::resetCellList<<<grid, block>>>(
        d_cell_list.cell_head,
        d_cell_list.particle_next,
        d_cell_list.particle_cell_id,
        d_cell_list.num_cells,
        d_particles.n
    );
}

void ParticleSystem::buildCellList(const CellType* d_cell_type, int nx, int ny, int nz)
{
    if (d_cell_list.cell_head == nullptr || d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::buildCellList<<<grid, block>>>(
        d_cell_list.cell_head,
        d_cell_list.particle_next,
        d_cell_list.particle_cell_id,
        d_particles.active,
        d_particles.x,
        d_particles.y,
        d_particles.z,
        d_cell_type,
        d_particles.n,
        nx,
        ny,
        nz
    );
}

ParticleData& ParticleSystem::data()
{
    return d_particles;
}

const ParticleData& ParticleSystem::data() const
{
    return d_particles;
}

ParticleCellList& ParticleSystem::cellList()
{
    return d_cell_list;
}

const ParticleCellList& ParticleSystem::cellList() const
{
    return d_cell_list;
}
