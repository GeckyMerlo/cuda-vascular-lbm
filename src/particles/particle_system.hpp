// particle_system.hpp
#pragma once
#include "particle_data.cuh"
#include "../space/space_data.cuh"

class ParticleSystem {
public:
    ParticleSystem();
    ~ParticleSystem();

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

    void allocate(int n);
    void allocate(int n, int num_cells);
    void initialize(ParticleData& h_particles);
    void free();

    void resetCounters();
    void resetForces();
    void updateVelocity(double dt);
    void updatePosition(double dt);
    void clampForces(double max_force);
    void clampVelocities(double max_speed);
    void updateAngularVelocity(double dt);
    void updateOrientation(double dt);
    void swapForces();
    void deactivateExited(const CellType* d_cell_type, int nx, int ny, int nz);
    void resetCellList();
    void buildCellList(const CellType* d_cell_type, int nx, int ny, int nz);

    ParticleData& data();
    const ParticleData& data() const;
    ParticleCellList& cellList();
    const ParticleCellList& cellList() const;

private:
    ParticleData d_particles;
    ParticleCellList d_cell_list;
};
