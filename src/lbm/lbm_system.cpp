#include "lbm_system.hpp"
#include "cuda_launch_config.cuh"
#include "lbm_kernels.cuh"
#include <iostream>

#include <cuda_runtime.h>

namespace {

template <typename Launch>
void launchMaybeProfile(
    bool enabled,
    const char* name,
    std::vector<LBMSystem::KernelTiming>& timings,
    Launch launch)
{
    if (!enabled) {
        launch();
        return;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    bool found = false;
    for (LBMSystem::KernelTiming& timing : timings) {
        if (timing.name == name) {
            timing.total_ms += elapsed_ms;
            timing.calls += 1;
            found = true;
            break;
        }
    }

    if (!found) {
        timings.push_back({name, elapsed_ms, 1});
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace

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

LBMSystem::LBMSystem(
    const SpaceData& space_data,
    double dt,
    double tau,
    OutletKernelVariant outlet_variant)
    : dt(dt),
      tau(tau),
      omega(1.0 / tau),
      outlet_variant(outlet_variant),
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

void LBMSystem::setKernelProfilingEnabled(bool enabled) {
    kernel_profiling_enabled = enabled;
}

double LBMSystem::KernelTiming::averageMs() const {
    return calls > 0 ? total_ms / static_cast<double>(calls) : 0.0;
}

const std::vector<LBMSystem::KernelTiming>& LBMSystem::kernelTimings() const {
    return kernel_timings;
}

void LBMSystem::collide() {
    int block = kCudaBlockSize;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    launchMaybeProfile(kernel_profiling_enabled, "collide_kernel", kernel_timings, [&]() {
        collide_kernel<<<grid, block>>>(
            fluid.f,
            fluid.f_temp,
            fluid.density,
            fluid.velocity_x,
            fluid.velocity_y,
            fluid.velocity_z,
            fluid.force_x,
            fluid.force_y,
            fluid.force_z,
            space.d_cell_type,
            space.num_cells,
            omega
        );
    });
}

void LBMSystem::stream() {
    int block = kCudaBlockSize;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    launchMaybeProfile(kernel_profiling_enabled, "stream_kernel", kernel_timings, [&]() {
        stream_kernel<<<grid, block>>>(
            fluid.f,
            fluid.f_temp,
            space.d_cell_type,
            space.num_cells,
            space.nx,
            space.ny,
            space.nz
        );
    });
}

void LBMSystem::applyBoundaryConditions() {
    int block = kCudaBlockSize;

    int grid_all = (space.num_cells + block - 1) / block;
    if (grid_all <= 0) return;

    launchMaybeProfile(kernel_profiling_enabled, "inlet_kernel", kernel_timings, [&]() {
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
    });


    int grid_outlet = (space.num_outlets + block - 1) / block;
    if (grid_outlet <= 0 || space.d_outlet_src_ids == nullptr) return;

    switch (outlet_variant) {
    case OutletKernelVariant::CopyAll:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_copy_all_kernel", kernel_timings, [&]() {
            outlet_copy_all_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::CopyMissing:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_copy_missing_kernel", kernel_timings, [&]() {
            outlet_copy_missing_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::EquilibriumRho1:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_equilibrium_rho1_kernel", kernel_timings, [&]() {
            outlet_equilibrium_rho1_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::ConvectiveSoft:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_convective_soft_kernel", kernel_timings, [&]() {
            outlet_convective_soft_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::ZhouHe:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_zhou_he_kernel", kernel_timings, [&]() {
            outlet_zhou_he_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    }
}

void LBMSystem::computeMacroscopicVariables() {
    int block = kCudaBlockSize;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    launchMaybeProfile(kernel_profiling_enabled, "computeMacroscopicVariables_kernel", kernel_timings, [&]() {
        ::computeMacroscopicVariables_kernel<<<grid, block>>>(
            fluid.f,
            fluid.density, fluid.velocity_x, fluid.velocity_y, fluid.velocity_z,
            fluid.force_x, fluid.force_y, fluid.force_z,
            space.d_cell_type,
            space.num_cells
        );
    });
}

void LBMSystem::initializeEquilibrium() {
    int block = kCudaBlockSize;
    int grid = (space.num_cells + block - 1) / block;
    if (grid <= 0) return;

    launchMaybeProfile(kernel_profiling_enabled, "initializeEquilibrium_kernel", kernel_timings, [&]() {
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
    });
}

void LBMSystem::computeOutlet() {
    int block = kCudaBlockSize;

    int grid_outlet = (space.num_outlets + block - 1) / block;
    if (grid_outlet <= 0 || space.d_outlet_src_ids == nullptr) return;

    switch (outlet_variant) {
    case OutletKernelVariant::CopyAll:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_copy_all_kernel", kernel_timings, [&]() {
            outlet_copy_all_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::CopyMissing:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_copy_missing_kernel", kernel_timings, [&]() {
            outlet_copy_missing_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::EquilibriumRho1:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_equilibrium_rho1_kernel", kernel_timings, [&]() {
            outlet_equilibrium_rho1_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::ConvectiveSoft:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_convective_soft_kernel", kernel_timings, [&]() {
            outlet_convective_soft_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    case OutletKernelVariant::ZhouHe:
        launchMaybeProfile(kernel_profiling_enabled, "outlet_zhou_he_kernel", kernel_timings, [&]() {
            outlet_zhou_he_kernel<<<grid_outlet, block>>>(
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
                space.nz);
        });
        break;
    }
}

void LBMSystem::computeInlet() {
    int block = kCudaBlockSize;

    int grid_all = (space.num_cells + block - 1) / block;
    if (grid_all <= 0) return;

    launchMaybeProfile(kernel_profiling_enabled, "inlet_kernel", kernel_timings, [&]() {
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
    });

}

void LBMSystem::copy_boundary_to_temp() {
    int block = kCudaBlockSize;

    int grid_all = (space.num_cells + block - 1) / block;
    if (grid_all <= 0) return;

    launchMaybeProfile(kernel_profiling_enabled, "copy_boundary_to_temp_kernel", kernel_timings, [&]() {
        copy_boundary_to_temp_kernel<<<grid_all, block>>>(
            fluid.f_temp,
            fluid.f,
            space.d_cell_type,
            space.num_cells
        );
    });
}


