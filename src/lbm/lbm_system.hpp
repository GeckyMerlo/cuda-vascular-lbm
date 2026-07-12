#pragma once

#include <vector>

#include "fluid_data.cuh"
#include "../space/space_data.cuh"
#include "lbm_constants.cuh"

enum class OutletKernelVariant {
    CopyAll,
    CopyMissing,
    EquilibriumRho1,
    ConvectiveSoft,
    ZhouHe
};

class LBMSystem {
public:
    LBMSystem(
        const SpaceData& space_data,
        double dt,
        double tau,
        OutletKernelVariant outlet_variant = OutletKernelVariant::ZhouHe);
    ~LBMSystem();

    LBMSystem(const LBMSystem&) = delete;
    LBMSystem& operator=(const LBMSystem&) = delete;

    void step();

    FluidData& data();
    const FluidData& data() const;

private:
    double dt;
    double tau;
    double omega;
    OutletKernelVariant outlet_variant;

    FluidData fluid;
    const SpaceData& space;

    void collide();
    void stream();
    void applyBoundaryConditions();
    void computeMacroscopicVariables();
    void initializeEquilibrium();
    void computeInlet();
    void computeOutlet();
    void copy_boundary_to_temp();

    bool debug_mode = false; // Set to true to enable debug mode, false to disable
    std::vector<double> h_f;
};
