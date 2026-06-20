// lbm_constants.cuh
#pragma once

constexpr int Q = 19;

extern __constant__ int d_cx[Q];
extern __constant__ int d_cy[Q];
extern __constant__ int d_cz[Q];
extern __constant__ double d_w[Q];
extern __constant__ int d_opposite[Q];

void initLBMConstants();