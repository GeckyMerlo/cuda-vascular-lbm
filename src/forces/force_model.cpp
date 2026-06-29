#include "force_model.hpp"

void ForceModel::computeParticleForces(ParticleData& p, double dt)
{
    (void)p;
    (void)dt;
}

void ForceModel::computeWallForces(ParticleData& p)
{
    (void)p;
}

void ForceModel::computeFluidForces(ParticleData& p, FluidData& fluid)
{
    (void)p;
    (void)fluid;
}
