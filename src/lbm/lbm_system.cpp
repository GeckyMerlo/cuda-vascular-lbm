#include "lbm_system.hpp"
#include "lbm_kernels.cuh"
#include <iostream>

#include <cuda_runtime.h>

double computeTotalMassCPU(
    const std::vector<double>& h_f,
    const CellType* h_cell_type,
    int num_cells
) {
    double mass = 0.0;

    for (int id = 0; id < num_cells; id++) {
        if (h_cell_type[id] == SOLID) continue;

        for (int q = 0; q < Q; q++) {
            mass += h_f[id * Q + q];
        }
    }

    return mass;
}

LBMSystem::LBMSystem(const SpaceData& space_data, double dt, double tau)
    : dt(dt),
      tau(tau),
      omega(1.0 / tau),
      fluid(space_data.nx, space_data.ny, space_data.nz),
      space(space_data)
{
    h_f.resize(space.num_cells * Q); // only for debug

    initLBMConstants();
    initializeEquilibrium();
    computeMacroscopicVariables();
}

LBMSystem::~LBMSystem() {
    // FluidData destructor will automatically free GPU memory
}

void LBMSystem::step() {
    //copy_boundary_to_temp();
    
    collide();
    if (debug_mode) {
        cudaMemcpy(h_f.data(), fluid.f_temp, space.num_cells * Q * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "[after collide] mass = " << computeTotalMassCPU(h_f, space.h_cell_type, space.num_cells) << std::endl;
    }
    /* TEST DI STREAM
    cudaMemcpy(fluid.f_temp, fluid.f, space.num_cells * Q * sizeof(double),
           cudaMemcpyDeviceToDevice);
    */
    stream();
    if (debug_mode) {
        cudaMemcpy(h_f.data(), fluid.f, space.num_cells * Q * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "[after stream] mass = " << computeTotalMassCPU(h_f, space.h_cell_type, space.num_cells) << std::endl;
    }
    computeInlet();
    if (debug_mode) {
        cudaMemcpy(h_f.data(), fluid.f, space.num_cells * Q * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "[after inlet] mass = " << computeTotalMassCPU(h_f, space.h_cell_type, space.num_cells) << std::endl;
    }
    computeOutlet();
    if (debug_mode) {
        cudaMemcpy(h_f.data(), fluid.f, space.num_cells * Q * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "[after outlet] mass = " << computeTotalMassCPU(h_f, space.h_cell_type, space.num_cells) << std::endl;
    }
    //applyBoundaryConditions();
    computeMacroscopicVariables();
}

FluidData& LBMSystem::data() {
    return fluid;
}

const FluidData& LBMSystem::data() const {
    return fluid;
}   

void LBMSystem::collide() {
    int block = 256;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    collide_kernel<<<grid, block>>>(
        fluid.f,
        fluid.f_temp,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_cell_type,
        space.num_cells,
        omega
    );
}

void LBMSystem::stream() {
    int block = 256;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    stream_kernel<<<grid, block>>>(
        fluid.f,
        fluid.f_temp,
        space.d_cell_type,
        space.num_cells,
        space.nx,
        space.ny,
        space.nz
    );
}

void LBMSystem::applyBoundaryConditions() {
    int block = 256;

    int grid_all = (space.num_cells + block - 1) / block;
    if (grid_all <= 0) return;

    inlet_kernel<<<grid_all, block>>>(
        fluid.f,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_cell_type,
        space.num_cells,
        //1.0,      // rho0
        0.0,      // ux
        0.0,      // uy
        0.0075 ,     // uz, vessel axis in vena_cilindrica.geo, <-- value given consideri fisic velocity of 0,15 m/s, voxel 0.2 mm, delta_t 0.00001 s
        space.nx,
        space.ny,
        space.nz
    );


    int grid_outlet = (space.num_outlets + block - 1) / block;
    if (grid_outlet <= 0 || space.d_outlet_src_ids == nullptr) return;

    outlet_kernel<<<grid_outlet, block>>>(
        fluid.f,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_outlet_ids,
        space.d_outlet_src_ids,
        space.num_outlets,
        space.d_cell_type,
        space.nx,
        space.ny,
        space.nz
    );
}

void LBMSystem::computeMacroscopicVariables() {
    int block = 256;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    ::computeMacroscopicVariables_kernel<<<grid, block>>>(
        fluid.f,
        fluid.density, fluid.velocity_x, fluid.velocity_y, fluid.velocity_z,
        space.d_cell_type,
        space.num_cells
    );
}

void LBMSystem::initializeEquilibrium() {
    int block = 256;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    initializeEquilibrium_kernel<<<grid, block>>>(
        fluid.f,
        fluid.f_temp,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_cell_type,
        space.num_cells,
        1.0,
        0.0,
        0.0,
        0.0
    );
}

void LBMSystem::computeOutlet() {
    int block = 256;

    int grid_outlet = (space.num_outlets + block - 1) / block;
    if (grid_outlet <= 0 || space.d_outlet_src_ids == nullptr) return;

    outlet_kernel<<<grid_outlet, block>>>(
        fluid.f,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_outlet_ids,
        space.d_outlet_src_ids,
        space.num_outlets,
        space.d_cell_type,
        space.nx,
        space.ny,
        space.nz
    );
}

void LBMSystem::computeInlet() {
    int block = 256;

    int grid_all = (space.num_cells + block - 1) / block;
    if (grid_all <= 0) return;

    inlet_kernel<<<grid_all, block>>>(
        fluid.f,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_cell_type,
        space.num_cells,
        //1.0,      // rho0
        0.0,      // ux
        0.0,      // uy
        0.0075 ,     // uz, vessel axis in vena_cilindrica.geo, <-- value given consideri fisic velocity of 0,15 m/s, voxel 0.2 mm, delta_t 0.00001 s
        space.nx,
        space.ny,
        space.nz
    );

}

void LBMSystem::copy_boundary_to_temp() {
    int block = 256;

    int grid_all = (space.num_cells + block - 1) / block;
    if (grid_all <= 0) return;

    copy_boundary_to_temp_kernel<<<grid_all, block>>>(
        fluid.f_temp,
        fluid.f,
        space.d_cell_type,
        space.num_cells
    );
}






