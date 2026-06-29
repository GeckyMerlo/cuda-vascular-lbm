// particle_system.cpp
#include "particle_system.hpp"
#include "particle_kernels.cuh"

#include <cuda_runtime.h>

ParticleSystem::ParticleSystem() {
    d_particles = {};
}

ParticleSystem::~ParticleSystem() {
    free();
}

void ParticleSystem::allocate(int n) {
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

    cudaMalloc(reinterpret_cast<void**>(&d_particles.mass),   n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.radius), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.shape),  n * sizeof(ParticleShape));

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

    cudaMalloc(reinterpret_cast<void**>(&d_particles.fx_old), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.fy_old), n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&d_particles.fz_old), n * sizeof(double));
}

void ParticleSystem::initialize(ParticleData& h_particles) {
    cudaMemcpy(d_particles.x, h_particles.x, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.y, h_particles.y, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.z, h_particles.z, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.vx, h_particles.vx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vy, h_particles.vy, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vz, h_particles.vz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.fx, h_particles.fx, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.fy, h_particles.fy, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.fz, h_particles.fz, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particles.mass,   h_particles.mass,   d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.radius, h_particles.radius, d_particles.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.shape,  h_particles.shape,  d_particles.n * sizeof(ParticleShape), cudaMemcpyHostToDevice);

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
    

void ParticleSystem::free() {
    cudaFree(d_particles.x);
    cudaFree(d_particles.y);
    cudaFree(d_particles.z);

    cudaFree(d_particles.vx);
    cudaFree(d_particles.vy);
    cudaFree(d_particles.vz);

    cudaFree(d_particles.fx);
    cudaFree(d_particles.fy);
    cudaFree(d_particles.fz);

    cudaFree(d_particles.mass);
    cudaFree(d_particles.radius);
    cudaFree(d_particles.shape);

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

    cudaFree(d_particles.fx_old);
    cudaFree(d_particles.fy_old);
    cudaFree(d_particles.fz_old);

    d_particles = {};
}



void ParticleSystem::updateVelocity(double dt) {
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::updateVelocity<<<grid, block>>>(
        d_particles.vx, d_particles.vy, d_particles.vz,
        d_particles.fx_old, d_particles.fy_old, d_particles.fz_old,
        d_particles.fx, d_particles.fy, d_particles.fz,
        d_particles.n,
        dt,
        d_particles.mass
    );
}

void ParticleSystem::updatePosition(double dt) {
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::updatePosition<<<grid, block>>>(
        d_particles.x, d_particles.y, d_particles.z,
        d_particles.vx, d_particles.vy, d_particles.vz,
        d_particles.fx, d_particles.fy, d_particles.fz,
        d_particles.n,
        dt,
        d_particles.mass
    );
}

void ParticleSystem::resetForces() {
    if (d_particles.n <= 0) {
        return;
    }

    int block = 256;
    int grid = (d_particles.n + block - 1) / block;

    ::resetForces<<<grid, block>>>(
        d_particles.fx, d_particles.fy, d_particles.fz,
        d_particles.tx, d_particles.ty, d_particles.tz,
        d_particles.n
    );
}

void ParticleSystem::swapForces() {
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

ParticleData& ParticleSystem::data() {
    return d_particles;
}

const ParticleData& ParticleSystem::data() const {
    return d_particles;
}
