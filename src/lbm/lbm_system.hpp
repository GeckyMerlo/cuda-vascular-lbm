#include "fluid_data.cuh"
#include "../space/space_data.cuh"
#include "lbm_kernels.cuh"
#include "lbm_constants.cuh"

class LBMSystem {
public:
    LBMSystem(const SpaceData& space_data, double dt, double tau);
    ~LBMSystem();

    void step();

    FluidData& data();
    const FluidData& data() const;

private:
    double dt;
    double tau;
    double omega;

    FluidData fluid;
    const SpaceData& space;

    void collide();
    void stream();
    void applyBoundaryConditions();
    void computeMacroscopicVariables();

};

