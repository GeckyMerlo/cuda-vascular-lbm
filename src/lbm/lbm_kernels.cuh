__global__ void collide_kernel(
    double* f,
    double* f_temp,
    const double* rho,
    const double* ux,
    const double* uy,
    const double* uz,
    const int* cell_type,
    int num_cells,
    double omega
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    if (cell_type[id] == SOLID) return;

    double local_rho = rho[id];
    double local_ux  = ux[id];
    double local_uy  = uy[id];
    double local_uz  = uz[id];

    double u2 = local_ux*local_ux +
                local_uy*local_uy +
                local_uz*local_uz;

    for (int q = 0; q < 19; q++) {
        double cu = d_cx[q]*local_ux +
                    d_cy[q]*local_uy +
                    d_cz[q]*local_uz;

        double feq = d_w[q] * local_rho *
            (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);

        int idx = id * 19 + q;

        f_temp[idx] = f[idx] - omega * (f[idx] - feq);
    }
}

__global__ void computeMacroscopicVariables_kernel(
    const double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const int* cell_type,
    int num_cells
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    if (cell_type[id] == SOLID) return;

    double local_rho = 0.0;
    double local_ux = 0.0;
    double local_uy = 0.0;
    double local_uz = 0.0;

    for (int q = 0; q < 19; q++) {
        int idx = id * 19 + q;
        double f_val = f[idx];
        local_rho += f_val;
        local_ux += f_val * d_cx[q];
        local_uy += f_val * d_cy[q];
        local_uz += f_val * d_cz[q];
    }

    rho[id] = local_rho;
    if (local_rho > 0.0) {
        ux[id] = local_ux / local_rho;
        uy[id] = local_uy / local_rho;
        uz[id] = local_uz / local_rho;
    }else{
        ux[id] = 0.0;
        uy[id] = 0.0;
        uz[id] = 0.0;
    }
}

// streaming pull model, fluid cells pull from neighbors, solid cells keep their distribution functions unchanged
__global__ void stream_kernel(
    double* f,
    const double* f_temp,
    const int* cell_type,
    int num_cells,
    int nx, int ny, int nz
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    int z = id / (nx * ny);
    int y = (id % (nx * ny)) / nx;
    int x = id % nx;

    for (int q = 0; q < 19; q++) {
        // Compute source cell coordinates for every direction q
        int src_x = (x - d_cx[q] + nx) % nx;
        int src_y = (y - d_cy[q] + ny) % ny;
        int src_z = (z - d_cz[q] + nz) % nz;

        int src_id = src_z * (nx * ny) + src_y * nx + src_x;

        if (cell_type[src_id] != SOLID) {
            f[id * 19 + q] = f_temp[src_id * 19 + q];
        } else {
            f[id * 19 + q] = f_temp[id * 19 + q];
        }
    }
}

__global__ void wall_bounce_back_kernel(
    double* f,
    const int* cell_type,
    int num_cells
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    if (cell_type[id] == SOLID) {
        double tmp[19];

        // Load current distribution functions into temporary array
        for (int q = 0; q < 19; q++) {
            tmp[q] = f[id * 19 + q];
        }

        // Bounce-back: swap distribution functions with their opposites
        for (int q = 0; q < 19; q++) {
            int opposite_q = d_opposite[q];
            f[id * 19 + q] = tmp[opposite_q];
        }
    }else if (cell_type[id] == INLET) {

    }else if (cell_type[id] == OUTLET) {

    }
}

__global__ void inlet_kernel(
    double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const int* cell_type,
    int num_cells,
    double rho0,
    double u_in
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    if (cell_type[id] != INLET) return;

    rho[id] = rho0;
    ux[id] = u_in;
    uy[id] = 0.0;
    uz[id] = 0.0;

    double u2 = u_in * u_in;

    for (int q = 0; q < 19; q++) {
        double cu = d_cx[q] * u_in;

        double feq = d_w[q] * rho0 *
            (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);

        f[id * 19 + q] = feq;
    }
}

__global__ void outlet_kernel(
    double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const int* outlet_ids,
    const int* outlet_src_ids,
    int num_outlet_cells
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_outlet_cells) return;

    int id     = outlet_ids[k];
    int src_id = outlet_src_ids[k];

    for (int q = 0; q < 19; q++) {
        f[id * 19 + q] = f[src_id * 19 + q];
    }

    rho[id] = rho[src_id];
    ux[id]  = ux[src_id];
    uy[id]  = uy[src_id];
    uz[id]  = uz[src_id];
}

