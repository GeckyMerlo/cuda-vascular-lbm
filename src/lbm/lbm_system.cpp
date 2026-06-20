#include "lbm_system.hpp"
#include <algorithm> 

LBMSystem::LBMSystem(const SpaceData& space_data, double dt, double tau)
    : dt(dt),
      tau(tau),
      omega(1.0 / tau),
      fluid(space_data.nx, space_data.ny, space_data.nz),
      space(space_data)
{
    initLBMConstants();
}

LBMSystem::~LBMSystem() {
    // FluidData destructor will automatically free GPU memory
}

void LBMSystem::step() {
    computeMacroscopicVariables();
    collide();
    stream();
    applyBoundaryConditions();
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

    wall_bounce_back_kernel<<<grid_all, block>>>(
        fluid.f,
        space.d_cell_type,
        space.num_cells
    );

    inlet_kernel<<<grid_all, block>>>(
        fluid.f,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_cell_type,
        space.num_cells,
        // Inlet parameters (can be adjusted as needed) <-- TODO: Make these configurable
        1.0,      // rho0
        0.05      // u_in
    );

    int grid_outlet = (space.num_outlet_cells + block - 1) / block;

    outlet_kernel<<<grid_outlet, block>>>(
        fluid.f,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_outlet_ids,
        space.d_outlet_src_ids,
        space.num_outlet_cells
    );
}

void LBMSystem::computeMacroscopicVariables() {
    int block = 256;
    int grid = (space.num_cells + block - 1) / block;

    ::computeMacroscopicVariables_kernel<<<grid, block>>>(
        fluid.f,
        fluid.density, fluid.velocity_x, fluid.velocity_y, fluid.velocity_z,
        space.d_cell_type,
        space.num_cells
    );
}










