#pragma once

enum ParticleShape : int {
    SPHERE = 0,
    ELLIPSOID = 1
};

enum ParticleSpecies : int {
    RBC = 0,
    PLATELET = 1,
    LEUKOCYTE = 2
};

struct ParticleCellList {
    int num_cells = 0;
    int* cell_head = nullptr;
    int* particle_next = nullptr;
    int* particle_cell_id = nullptr;
};

struct ParticleData {
    int n = 0; // fixed particle capacity

    // common data
    double* x = nullptr;  double* y = nullptr;  double* z = nullptr; //position
    double* vx = nullptr; double* vy = nullptr; double* vz = nullptr; //velocity
    double* fx = nullptr; double* fy = nullptr; double* fz = nullptr; // force
    double* fx_old = nullptr; double* fy_old = nullptr; double* fz_old = nullptr; // force at previous time step

    double* mass = nullptr;
    double* inertia_x = nullptr; double* inertia_y = nullptr; double* inertia_z = nullptr;
    ParticleShape* shape = nullptr;
    ParticleSpecies* species = nullptr;
    int* active = nullptr;
    int* age = nullptr;
    int* contact_count = nullptr;

    // sphere only
    double* radius = nullptr; 

    // rotation
    double* wx = nullptr; double* wy = nullptr; double* wz = nullptr; // angular velocity
    double* tx = nullptr; double* ty = nullptr; double* tz = nullptr; // torque moment
    double* qw = nullptr; double* qx = nullptr; double* qy = nullptr; double* qz = nullptr; // quaternion (orientation)

    // ellipsoid only
    double* a = nullptr; double* b = nullptr; double* c = nullptr; // semi-axes lengths

    // append-only injection counters on device
    int* next_slot = nullptr;
    int* injected_total = nullptr;
    int* dropped_total = nullptr;
    int* exited_total = nullptr;
};
