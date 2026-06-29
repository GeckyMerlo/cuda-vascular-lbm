#pragma once

enum ParticleShape : int {
    SPHERE = 0,
    ELLIPSOID = 1
};

struct ParticleData {
    int n = 0; // number of particles

    // common data
    double* x = nullptr;  double* y = nullptr;  double* z = nullptr; //position
    double* vx = nullptr; double* vy = nullptr; double* vz = nullptr; //velocity
    double* fx = nullptr; double* fy = nullptr; double* fz = nullptr; // force
    double* fx_old = nullptr; double* fy_old = nullptr; double* fz_old = nullptr; // force at previous time step

    double* mass = nullptr;
    ParticleShape* shape = nullptr;

    // sphere only
    double* radius = nullptr; 

    // rotation
    double* wx = nullptr; double* wy = nullptr; double* wz = nullptr; // angular velocity
    double* tx = nullptr; double* ty = nullptr; double* tz = nullptr; // torque moment
    double* qw = nullptr; double* qx = nullptr; double* qy = nullptr; double* qz = nullptr; // quaternion (orientation)

    // ellipsoid only
    double* a = nullptr; double* b = nullptr; double* c = nullptr; // semi-axes lengths
};
