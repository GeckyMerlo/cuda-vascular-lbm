// particle_system.hpp
#pragma once
#include "particle_data.cuh"

class ParticleSystem {
public:
    ParticleSystem();
    ~ParticleSystem();

    void allocate(int n);
    void initialize(ParticleData& h_particles);
    void free();

    void resetForces();
    void updateVelocity(double dt);
    void updatePosition(double dt);
    void swapForces();

    ParticleData& data();
    const ParticleData& data() const;

private:
    ParticleData d_particles;
};