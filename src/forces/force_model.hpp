#pragma once

#include "../particles/particle_system.hpp"
#include "../lbm/fluid_data.cuh"
#include "../space/space_data.cuh"

struct ParticleForceParameters {
    double tau = 0.8;
    double contact_stiffness = 0.05;
    double contact_damping = 0.02;
    double friction = 0.2;
    double wall_stiffness = 0.08;
    double wall_damping = 0.02;
};

class ForceModel {
public:
    void resetFluidForces(FluidData& fluid, int num_cells);
    void injectParticles(
        ParticleData& p,
        const SpaceData& space,
        const FluidData& fluid,
        int step,
        int rbc_count,
        int platelet_count,
        int leukocyte_count);
    void computeFluidForces(ParticleData& p, FluidData& fluid, const SpaceData& space, const ParticleForceParameters& params);
    void computeParticleForces(ParticleData& p, ParticleCellList& cells, const SpaceData& space, const ParticleForceParameters& params);
    void computeWallForces(ParticleData& p, const SpaceData& space, const ParticleForceParameters& params);

    // Compatibility no-ops for older call sites.
    void computeParticleForces(ParticleData& p, double dt);
    void computeWallForces(ParticleData& p);
    void computeFluidForces(ParticleData& p, FluidData& fluid);
};
