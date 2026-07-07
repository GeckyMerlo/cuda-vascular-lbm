#pragma once

#include <cuda_runtime.h>

struct FluidData {
public:
    double* density = nullptr;
    double* velocity_x = nullptr;
    double* velocity_y = nullptr;
    double* velocity_z = nullptr;
    double* force_x = nullptr;
    double* force_y = nullptr;
    double* force_z = nullptr;
    double* f = nullptr; // distribution functions
    double* f_temp = nullptr; // temporary distribution functions for streaming step
    int nx = 0, ny = 0, nz = 0;

    FluidData(int nx, int ny, int nz){
        cudaMalloc(reinterpret_cast<void**>(&density), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&velocity_x), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&velocity_y), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&velocity_z), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&force_x), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&force_y), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&force_z), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&f), nx*ny*nz*19*sizeof(double)); 
        cudaMalloc(reinterpret_cast<void**>(&f_temp), nx*ny*nz*19*sizeof(double));
        cudaMemset(force_x, 0, nx*ny*nz*sizeof(double));
        cudaMemset(force_y, 0, nx*ny*nz*sizeof(double));
        cudaMemset(force_z, 0, nx*ny*nz*sizeof(double));
        this->nx = nx;
        this->ny = ny;
        this->nz = nz;
    };

    FluidData(const FluidData&) = delete;
    FluidData& operator=(const FluidData&) = delete;
    
    ~FluidData(){
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(force_x);
        cudaFree(force_y);
        cudaFree(force_z);
        cudaFree(f);
        cudaFree(f_temp);
    };
};
