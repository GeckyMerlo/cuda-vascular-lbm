#pragma once

#include <cuda_runtime.h>

struct FluidData {
public:
    double* density = nullptr;
    double* velocity_x = nullptr;
    double* velocity_y = nullptr;
    double* velocity_z = nullptr;
    double* f = nullptr; // distribution functions
    double* f_temp = nullptr; // temporary distribution functions for streaming step
    int nx = 0, ny = 0, nz = 0;

    FluidData(int nx, int ny, int nz){
        cudaMalloc(reinterpret_cast<void**>(&density), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&velocity_x), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&velocity_y), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&velocity_z), nx*ny*nz*sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&f), nx*ny*nz*19*sizeof(double)); 
        cudaMalloc(reinterpret_cast<void**>(&f_temp), nx*ny*nz*19*sizeof(double));
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
        cudaFree(f);
        cudaFree(f_temp);
    };
};
