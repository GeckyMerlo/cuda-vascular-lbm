#include "lbm_constants.cuh"

__constant__ int d_cx[Q];
__constant__ int d_cy[Q];
__constant__ int d_cz[Q];
__constant__ double d_w[Q];
__constant__ int d_opposite[Q];

void initLBMConstants() {
    const int h_cx[Q] = {
        0,
        1, -1,  0,  0,  0,  0,
        1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0
    };

    const int h_cy[Q] = {
        0,
        0,  0,  1, -1,  0,  0,
        1, -1, -1,  1,  0,  0,  0,  0,  1, -1,  1, -1
    };

    const int h_cz[Q] = {
        0,
        0,  0,  0,  0,  1, -1,
        0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1
    };

    const double h_w[Q] = {
        1.0 / 3.0,

        1.0 / 18.0, 1.0 / 18.0,
        1.0 / 18.0, 1.0 / 18.0,
        1.0 / 18.0, 1.0 / 18.0,

        1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0
    };

    const int h_opposite[Q] = {
        0,
        2, 1,
        4, 3,
        6, 5,
        8, 7,
        10, 9,
        12, 11,
        14, 13,
        16, 15,
        18, 17
    };

    // Copy to constant memory
    cudaMemcpyToSymbol(d_cx, h_cx, sizeof(h_cx));
    cudaMemcpyToSymbol(d_cy, h_cy, sizeof(h_cy));
    cudaMemcpyToSymbol(d_cz, h_cz, sizeof(h_cz));
    cudaMemcpyToSymbol(d_w, h_w, sizeof(h_w));
    cudaMemcpyToSymbol(d_opposite, h_opposite, sizeof(h_opposite));
}