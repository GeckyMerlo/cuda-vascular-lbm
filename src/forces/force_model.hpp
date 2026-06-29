#pragma once

#include "../particles/particle_system.hpp"
#include "../lbm/fluid_data.cuh"

class ForceModel {
public:
    void computeParticleForces(ParticleData& p, double dt);
    void computeWallForces(ParticleData& p);
    void computeFluidForces(ParticleData& p, FluidData& fluid);
};
