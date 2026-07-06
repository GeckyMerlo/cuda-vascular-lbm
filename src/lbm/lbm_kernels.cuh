#pragma once

#include "lbm_constants.cuh"
#include "../space/space_data.cuh"

__device__ double feq_q(int q, double rho, double ux, double uy, double uz) {
    double cu = d_cx[q]*ux + d_cy[q]*uy + d_cz[q]*uz;
    double u2 = ux*ux + uy*uy + uz*uz;

    return d_w[q] * rho *
        (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
}

__global__ void initializeEquilibrium_kernel(
    double* f,
    double* f_temp,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const CellType* cell_type,
    int num_cells,
    double rho0,
    double ux0,
    double uy0,
    double uz0
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    if (cell_type[id] == SOLID) {
        rho[id] = 0.0;
        ux[id] = 0.0;
        uy[id] = 0.0;
        uz[id] = 0.0;

        for (int q = 0; q < Q; q++) {
            f[id * Q + q] = 0.0;
            f_temp[id * Q + q] = 0.0;
        }
        return;
    }

    rho[id] = rho0;
    ux[id] = ux0;
    uy[id] = uy0;
    uz[id] = uz0;

    double u2 = ux0*ux0 + uy0*uy0 + uz0*uz0;

    for (int q = 0; q < Q; q++) {
        double cu = d_cx[q]*ux0 + d_cy[q]*uy0 + d_cz[q]*uz0;
        double feq = d_w[q] * rho0 *
            (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);

        f[id * Q + q] = feq;
        f_temp[id * Q + q] = feq;
    }
}

__global__ void collide_kernel(
    double* f,
    double* f_temp,
    const double* rho,
    const double* ux,
    const double* uy,
    const double* uz,
    const CellType* cell_type,
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

    for (int q = 0; q < Q; q++) {
        double cu = d_cx[q]*local_ux +
                    d_cy[q]*local_uy +
                    d_cz[q]*local_uz;

        double feq = d_w[q] * local_rho *
            (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);

        int idx = id * Q + q;

        f_temp[idx] = f[idx] - omega * (f[idx] - feq);
    }
}

__global__ void computeMacroscopicVariables_kernel(
    const double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const CellType* cell_type,
    int num_cells
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    if (cell_type[id] == SOLID) return;

    double local_rho = 0.0;
    double local_ux = 0.0;
    double local_uy = 0.0;
    double local_uz = 0.0;

    for (int q = 0; q < Q; q++) {
        int idx = id * Q + q;
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
    const CellType* cell_type,
    int num_cells,
    int nx, int ny, int nz
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    //if (cell_type[id] != FLUID) return;
    if (cell_type[id] == SOLID) return;

    int z = id / (nx * ny);
    int y = (id % (nx * ny)) / nx;
    int x = id % nx;

    for (int q = 0; q < Q; q++) {
        // Compute source cell coordinates for every direction q
        int src_x = x - d_cx[q];
        int src_y = y - d_cy[q];
        int src_z = z - d_cz[q];

        if (src_x < 0 || src_x >= nx ||
            src_y < 0 || src_y >= ny ||
            src_z < 0 || src_z >= nz) {
            f[id * Q + q] = f_temp[id * Q + d_opposite[q]];
            continue;
        }

        int src_id = src_z * (nx * ny) + src_y * nx + src_x;

        if (cell_type[src_id] == SOLID) {
            // wall bounce-back
            f[id * Q + q] = f_temp[id * Q + d_opposite[q]];
        } else {
            // pull da FLUID, INLET, OUTLET
            f[id * Q + q] = f_temp[src_id * Q + q];
        }
    }
}

__global__ void wall_bounce_back_kernel(
    double* f,
    const CellType* cell_type,
    int num_cells
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    if (cell_type[id] == SOLID) {
        double tmp[Q];

        // Load current distribution functions into temporary array
        for (int q = 0; q < Q; q++) {
            tmp[q] = f[id * Q + q];
        }

        // Bounce-back: swap distribution functions with their opposites
        for (int q = 0; q < Q; q++) {
            int opposite_q = d_opposite[q];
            f[id * Q + q] = tmp[opposite_q];
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
    const CellType* cell_type,
    int num_cells,
    double rho0,
    double ux_in,
    double uy_in,
    double uz_in,
    int nx, int ny, int nz
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    if (cell_type[id] != INLET) return;

    rho[id] = rho0;
    ux[id]  = ux_in;
    uy[id]  = uy_in;
    uz[id]  = uz_in;

    int z = id / (nx * ny);
    int y = (id % (nx * ny)) / nx;
    int x = id % nx;

    for (int q = 1; q < Q; q++) {

        int src_x = x - d_cx[q];
        int src_y = y - d_cy[q];
        int src_z = z - d_cz[q];

        bool missing =
            src_x < 0 || src_x >= nx ||
            src_y < 0 || src_y >= ny ||
            src_z < 0 || src_z >= nz;

        if (!missing) {
            int src_id = src_z * (nx * ny) + src_y * nx + src_x;
            missing = (cell_type[src_id] == SOLID);
        }

        if (!missing) continue;

        int opp = d_opposite[q];

        double cu = d_cx[q]*ux_in + d_cy[q]*uy_in + d_cz[q]*uz_in;

        // Velocity bounce-back / Nguyen-Ladd style correction
        f[id*Q + q] = f[id*Q + opp] + 6.0 * d_w[q] * rho0 * cu;
    }
}

/*
__global__ void outlet_kernel(
    double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const int* outlet_ids,
    const int* outlet_src_ids,
    int num_outlet_cells,
    CellType* cell_type,
    int nx, int ny, int nz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_outlet_cells) return;

    int id     = outlet_ids[k];
    int src_id = outlet_src_ids[k];

    rho[id] = rho[src_id];
    ux[id]  = ux[src_id];
    uy[id]  = uy[src_id];
    uz[id]  = uz[src_id];

    //double u2 = ux[id]*ux[id] + uy[id]*uy[id] + uz[id]*uz[id];
    if (cell_type[src_id] != FLUID) return;
    
    for (int q = 0; q < Q; q++) {
        /*double cu = d_cx[q]*ux[id] + d_cy[q]*uy[id] + d_cz[q]*uz[id];

        f[id * Q + q] = d_w[q] * rho[id] *
            (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2); 
        f[id * Q + q] = f[src_id * Q + q];
    }
}
*/

/*
__global__ void outlet_kernel(
    double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const int* outlet_ids,
    const int* outlet_src_ids,
    int num_outlet_cells,
    const CellType* cell_type,
    int nx, int ny, int nz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_outlet_cells) return;

    int id     = outlet_ids[k];
    int src_id = outlet_src_ids[k];

    if (src_id < 0 || cell_type[src_id] == SOLID) return;

    rho[id] = rho[src_id];
    ux[id]  = ux[src_id];
    uy[id]  = uy[src_id];
    uz[id]  = uz[src_id];

    int z = id / (nx * ny);
    int y = (id % (nx * ny)) / nx;
    int x = id % nx;

    for (int q = 0; q < Q; q++) {
        int src_x = x - d_cx[q];
        int src_y = y - d_cy[q];
        int src_z = z - d_cz[q];

        bool missing =
            src_x < 0 || src_x >= nx ||
            src_y < 0 || src_y >= ny ||
            src_z < 0 || src_z >= nz;

        if (!missing) {
            int pull_id = src_z * (nx * ny) + src_y * nx + src_x;
            missing = (cell_type[pull_id] == SOLID);
        }

        if (missing) {
            f[id * Q + q] = f[src_id * Q + q];
        }
    }
}
*/

/*
__global__ void outlet_kernel(
    double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const int* outlet_ids,
    const int* outlet_src_ids,
    int num_outlet_cells,
    const CellType* cell_type,
    int nx, int ny, int nz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_outlet_cells) return;

    int id     = outlet_ids[k];
    int src_id = outlet_src_ids[k];

    if (src_id < 0 || cell_type[src_id] == SOLID) return;

    double rho_out = 1.0;

    double ux_out = ux[src_id];
    double uy_out = uy[src_id];
    double uz_out = uz[src_id];

    rho[id] = rho_out;
    ux[id]  = ux_out;
    uy[id]  = uy_out;
    uz[id]  = uz_out;

    double u2 = ux_out*ux_out + uy_out*uy_out + uz_out*uz_out;

    // unknown populations at z_max outlet: cz < 0
    int qs[5] = {6, 12, 13, 16, 17};

    for (int i = 0; i < 5; i++) {
        int q = qs[i];

        double cu = d_cx[q]*ux_out + d_cy[q]*uy_out + d_cz[q]*uz_out;

        f[id * Q + q] = d_w[q] * rho_out *
            (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
    }
}
*/
__global__ void outlet_kernel(
    double* f,
    double* rho,
    double* ux,
    double* uy,
    double* uz,
    const int* outlet_ids,
    const int* outlet_src_ids,
    int num_outlet_cells,
    const CellType* cell_type,
    int nx, int ny, int nz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_outlet_cells) return;

    int id     = outlet_ids[k];
    int src_id = outlet_src_ids[k];

    if (src_id < 0 || cell_type[src_id] == SOLID) return;

    double alpha = 0.1;      // convective strength
    double beta  = 0.05;     // density correction strength
    double rho_target = 1.0;

    // 1. Convective extrapolation
    for (int q = 0; q < Q; q++) {
        double f_old    = f[id     * Q + q];
        double f_inside = f[src_id * Q + q];

        f[id * Q + q] = (1.0 - alpha) * f_old + alpha * f_inside;
    }

    // 2. Compute current outlet density
    double rho_now = 0.0;
    for (int q = 0; q < Q; q++) {
        rho_now += f[id * Q + q];
    }

    // 3. Soft correction toward rho = 1
    double delta_rho = rho_target - rho_now;

    for (int q = 0; q < Q; q++) {
        f[id * Q + q] += beta * d_w[q] * delta_rho;
    }

    // 4. Recompute macroscopic variables
    double r = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;

    for (int q = 0; q < Q; q++) {
        double fq = f[id * Q + q];

        r  += fq;
        vx += fq * d_cx[q];
        vy += fq * d_cy[q];
        vz += fq * d_cz[q];
    }

    rho[id] = r;

    if (r > 1e-12) {
        ux[id] = vx / r;
        uy[id] = vy / r;
        uz[id] = vz / r;
    } else {
        ux[id] = 0.0;
        uy[id] = 0.0;
        uz[id] = 0.0;
    }
}

__global__ void copy_boundary_to_temp_kernel(
    double* f_temp,
    const double* f,
    const CellType* cell_type,
    int num_cells
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    if (cell_type[id] != FLUID) {
        for (int q = 0; q < Q; q++) {
            f_temp[id * Q + q] = f[id * Q + q];
        }
    }
}
