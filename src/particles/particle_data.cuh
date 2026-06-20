enum ParticleShape : int {
    SPHERE = 0,
    ELLIPSOID = 1
};

struct ParticleData {
    int n; // number of particles

    // common data
    double* x;  double* y;  double* z; //position
    double* vx; double* vy; double* vz; //velocity
    double* fx; double* fy; double* fz; // force
    double* fx_old; double* fy_old; double* fz_old; // force at previous time step

    double* mass;
    ParticleShape* shape;

    // sphere only
    double* radius; 

    // rotation
    double* wx; double* wy; double* wz; // angular velocity
    double* tx; double* ty; double* tz; // torque moment
    double* qw; double* qx; double* qy; double* qz; // quaternion (orientation)

    // ellipsoid only
    double* a; double* b; double* c; // semi-axes lengths
};