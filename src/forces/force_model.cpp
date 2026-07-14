#include "force_model.hpp"
#include "cuda_launch_config.cuh"

#include <cuda_runtime.h>

#include <math.h>

namespace {

constexpr double PI = 3.14159265358979323846;

__device__ int cellId(int x, int y, int z, int nx, int ny)
{
    return z * nx * ny + y * nx + x;
}

__device__ bool decodeCellId(int id, int nx, int ny, int nz, int& x, int& y, int& z)
{
    if (id < 0 || id >= nx * ny * nz) {
        return false;
    }

    z = id / (nx * ny);
    y = (id % (nx * ny)) / nx;
    x = id % nx;
    return true;
}

__device__ int nearestCellId(double x, double y, double z, int nx, int ny, int nz)
{
    int ix = static_cast<int>(floor(x + 0.5));
    int iy = static_cast<int>(floor(y + 0.5));
    int iz = static_cast<int>(floor(z + 0.5));

    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
        return -1;
    }

    return cellId(ix, iy, iz, nx, ny);
}

__device__ void cross(
    double ax, double ay, double az,
    double bx, double by, double bz,
    double& cx, double& cy, double& cz)
{
    cx = ay * bz - az * by;
    cy = az * bx - ax * bz;
    cz = ax * by - ay * bx;
}

__device__ void speciesDefaults(
    ParticleSpecies species,
    ParticleShape& shape,
    double& radius,
    double& mass,
    double& a,
    double& b,
    double& c,
    double& ix,
    double& iy,
    double& iz)
{
    if (species == RBC) {
        shape = ELLIPSOID;
        radius = 0.45;
        mass = 1.0;
        a = 0.65;
        b = 0.45;
        c = 0.18;
        ix = 0.2 * mass * (b * b + c * c);
        iy = 0.2 * mass * (a * a + c * c);
        iz = 0.2 * mass * (a * a + b * b);
        return;
    }

    if (species == PLATELET) {
        shape = SPHERE;
        radius = 0.25;
        mass = 0.25;
    } else {
        shape = SPHERE;
        radius = 0.60;
        mass = 2.0;
    }

    a = radius;
    b = radius;
    c = radius;
    ix = 0.4 * mass * radius * radius;
    iy = ix;
    iz = ix;
}

__global__ void resetFluidForces_kernel(double* fx, double* fy, double* fz, int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    fx[id] = 0.0;
    fy[id] = 0.0;
    fz[id] = 0.0;
}

__global__ void injectParticles_kernel(
    ParticleData p,
    const int* inlet_ids,
    int num_inlets,
    const double* fluid_ux,
    const double* fluid_uy,
    const double* fluid_uz,
    const double* normals_x,
    const double* normals_y,
    const double* normals_z,
    int nx,
    int ny,
    int nz,
    int step,
    int rbc_count,
    int platelet_count,
    int leukocyte_count)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rbc_count + platelet_count + leukocyte_count;
    if (local >= total) return;

    if (num_inlets <= 0) {
        atomicAdd(p.dropped_total, 1);
        return;
    }

    ParticleSpecies particle_species = RBC;
    if (local >= rbc_count + platelet_count) {
        particle_species = LEUKOCYTE;
    } else if (local >= rbc_count) {
        particle_species = PLATELET;
    }

    int slot = atomicAdd(p.next_slot, 1);
    if (slot >= p.n) {
        atomicAdd(p.dropped_total, 1);
        return;
    }

    unsigned int seed =
        static_cast<unsigned int>(step) * 1315423911u +
        static_cast<unsigned int>(local) * 2654435761u;
    int inlet_id = inlet_ids[seed % static_cast<unsigned int>(num_inlets)];

    int cx = 0, cy = 0, cz = 0;
    if (!decodeCellId(inlet_id, nx, ny, nz, cx, cy, cz)) {
        p.active[slot] = 0;
        atomicAdd(p.dropped_total, 1);
        return;
    }

    double nxn = normals_x != nullptr ? normals_x[inlet_id] : 0.0;
    double nyn = normals_y != nullptr ? normals_y[inlet_id] : 0.0;
    double nzn = normals_z != nullptr ? normals_z[inlet_id] : -1.0;
    double inside_x = -nxn;
    double inside_y = -nyn;
    double inside_z = -nzn;
    double inside_norm = sqrt(inside_x * inside_x + inside_y * inside_y + inside_z * inside_z);
    if (inside_norm <= 1e-12) {
        inside_x = 0.0;
        inside_y = 0.0;
        inside_z = 1.0;
    } else {
        inside_x /= inside_norm;
        inside_y /= inside_norm;
        inside_z /= inside_norm;
    }

    ParticleShape shape = SPHERE;
    double radius = 0.0, mass = 1.0, a = 0.0, b = 0.0, c = 0.0;
    double inertia_x = 1.0, inertia_y = 1.0, inertia_z = 1.0;
    speciesDefaults(particle_species, shape, radius, mass, a, b, c, inertia_x, inertia_y, inertia_z);

    p.x[slot] = static_cast<double>(cx) + inside_x * (radius + 0.75);
    p.y[slot] = static_cast<double>(cy) + inside_y * (radius + 0.75);
    p.z[slot] = static_cast<double>(cz) + inside_z * (radius + 0.75);
    p.vx[slot] = fluid_ux[inlet_id];
    p.vy[slot] = fluid_uy[inlet_id];
    p.vz[slot] = fluid_uz[inlet_id];
    p.fx[slot] = 0.0;
    p.fy[slot] = 0.0;
    p.fz[slot] = 0.0;
    p.fx_old[slot] = 0.0;
    p.fy_old[slot] = 0.0;
    p.fz_old[slot] = 0.0;
    p.tx[slot] = 0.0;
    p.ty[slot] = 0.0;
    p.tz[slot] = 0.0;
    p.wx[slot] = 0.0;
    p.wy[slot] = 0.0;
    p.wz[slot] = 0.0;
    p.qw[slot] = 1.0;
    p.qx[slot] = 0.0;
    p.qy[slot] = 0.0;
    p.qz[slot] = 0.0;
    p.radius[slot] = radius;
    p.mass[slot] = mass;
    p.inertia_x[slot] = inertia_x;
    p.inertia_y[slot] = inertia_y;
    p.inertia_z[slot] = inertia_z;
    p.shape[slot] = shape;
    p.species[slot] = particle_species;
    p.a[slot] = a;
    p.b[slot] = b;
    p.c[slot] = c;
    p.age[slot] = 0;
    p.contact_count[slot] = 0;
    p.active[slot] = 1;

    atomicAdd(p.injected_total, 1);
}

__device__ double interpolateFluid(
    const double* density,
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    const SpaceData space,
    double px,
    double py,
    double pz,
    double weights[8],
    int ids[8],
    double& rho,
    double& ux,
    double& uy,
    double& uz)
{
    int x0 = static_cast<int>(floor(px));
    int y0 = static_cast<int>(floor(py));
    int z0 = static_cast<int>(floor(pz));
    double fx = px - static_cast<double>(x0);
    double fy = py - static_cast<double>(y0);
    double fz = pz - static_cast<double>(z0);

    rho = 0.0;
    ux = 0.0;
    uy = 0.0;
    uz = 0.0;
    double total_weight = 0.0;

    int k = 0;
    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                int x = x0 + dx;
                int y = y0 + dy;
                int z = z0 + dz;
                double wx = dx == 0 ? 1.0 - fx : fx;
                double wy = dy == 0 ? 1.0 - fy : fy;
                double wz = dz == 0 ? 1.0 - fz : fz;
                double w = wx * wy * wz;

                ids[k] = -1;
                weights[k] = 0.0;

                if (x >= 0 && x < space.nx &&
                    y >= 0 && y < space.ny &&
                    z >= 0 && z < space.nz) {
                    int id = cellId(x, y, z, space.nx, space.ny);
                    if (space.d_cell_type[id] != SOLID && w > 0.0) {
                        ids[k] = id;
                        weights[k] = w;
                        total_weight += w;
                    }
                }

                ++k;
            }
        }
    }

    if (total_weight <= 1e-12) {
        return 0.0;
    }

    for (int i = 0; i < 8; ++i) {
        if (ids[i] < 0) continue;

        weights[i] /= total_weight;
        int id = ids[i];
        double w = weights[i];
        rho += density[id] * w;
        ux += velocity_x[id] * w;
        uy += velocity_y[id] * w;
        uz += velocity_z[id] * w;
    }

    return total_weight;
}

__global__ void computeDragAndFluidReaction_kernel(
    ParticleData p,
    const double* density,
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    double* force_x,
    double* force_y,
    double* force_z,
    SpaceData space,
    double tau,
    double max_particle_force,
    double fluid_reaction_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.n || !p.active[idx]) return;

    double weights[8];
    int ids[8];
    double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
    double valid_weight = interpolateFluid(
        density,
        velocity_x,
        velocity_y,
        velocity_z,
        space,
        p.x[idx],
        p.y[idx],
        p.z[idx],
        weights,
        ids,
        rho,
        ux,
        uy,
        uz);

    if (valid_weight <= 0.0 || rho <= 0.0) {
        return;
    }

    double mu = rho * (tau - 0.5) / 3.0;
    if (mu < 0.0) {
        mu = 0.0;
    }

    double drag_scale = 6.0 * PI * mu * p.radius[idx];
    double drag_x = drag_scale * (ux - p.vx[idx]);
    double drag_y = drag_scale * (uy - p.vy[idx]);
    double drag_z = drag_scale * (uz - p.vz[idx]);

    if (!isfinite(drag_x) || !isfinite(drag_y) || !isfinite(drag_z)) {
        p.active[idx] = 0;
        return;
    }

    if (max_particle_force > 0.0) {
        double drag_mag = sqrt(drag_x * drag_x + drag_y * drag_y + drag_z * drag_z);
        if (drag_mag > max_particle_force && drag_mag > 1e-16) {
            double scale = max_particle_force / drag_mag;
            drag_x *= scale;
            drag_y *= scale;
            drag_z *= scale;
        }
    }

    p.fx[idx] += drag_x;
    p.fy[idx] += drag_y;
    p.fz[idx] += drag_z;

    double reaction_scale = fluid_reaction_scale > 0.0 ? fluid_reaction_scale : 0.0;
    for (int i = 0; i < 8; ++i) {
        int id = ids[i];
        if (id < 0) continue;

        double w = weights[i] * reaction_scale;
        atomicAdd(&force_x[id], -drag_x * w);
        atomicAdd(&force_y[id], -drag_y * w);
        atomicAdd(&force_z[id], -drag_z * w);
    }
}

__global__ void computeContactForces_kernel(
    ParticleData p,
    ParticleCellList cells,
    const SpaceData space,
    double stiffness,
    double damping,
    double friction)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n || !p.active[i]) return;

    int center_id = cells.particle_cell_id[i];
    if (center_id < 0) return;

    int cx = 0, cy = 0, cz = 0;
    if (!decodeCellId(center_id, space.nx, space.ny, space.nz, cx, cy, cz)) return;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                int ny = cy + dy;
                int nz = cz + dz;
                if (nx < 0 || nx >= space.nx ||
                    ny < 0 || ny >= space.ny ||
                    nz < 0 || nz >= space.nz) {
                    continue;
                }

                int neighbor_id = cellId(nx, ny, nz, space.nx, space.ny);
                int j = cells.cell_head[neighbor_id];
                while (j >= 0) {
                    if (j > i && p.active[j]) {
                        double rx = p.x[i] - p.x[j];
                        double ry = p.y[i] - p.y[j];
                        double rz = p.z[i] - p.z[j];
                        double dist2 = rx * rx + ry * ry + rz * rz;
                        double min_dist = p.radius[i] + p.radius[j];

                        if (dist2 > 1e-16 && dist2 < min_dist * min_dist) {
                            double dist = sqrt(dist2);
                            double inv_dist = 1.0 / dist;
                            double nxn = rx * inv_dist;
                            double nyn = ry * inv_dist;
                            double nzn = rz * inv_dist;
                            double overlap = min_dist - dist;

                            double rvx = p.vx[i] - p.vx[j];
                            double rvy = p.vy[i] - p.vy[j];
                            double rvz = p.vz[i] - p.vz[j];
                            double vn = rvx * nxn + rvy * nyn + rvz * nzn;
                            double normal_mag = stiffness * overlap - damping * vn;
                            if (normal_mag < 0.0) normal_mag = 0.0;

                            double tvx = rvx - vn * nxn;
                            double tvy = rvy - vn * nyn;
                            double tvz = rvz - vn * nzn;
                            double tx = -damping * tvx;
                            double ty = -damping * tvy;
                            double tz = -damping * tvz;
                            double tangent_mag = sqrt(tx * tx + ty * ty + tz * tz);
                            double tangent_cap = friction * normal_mag;
                            if (tangent_mag > tangent_cap && tangent_mag > 1e-16) {
                                double scale = tangent_cap / tangent_mag;
                                tx *= scale;
                                ty *= scale;
                                tz *= scale;
                            }

                            double fx = normal_mag * nxn + tx;
                            double fy = normal_mag * nyn + ty;
                            double fz = normal_mag * nzn + tz;

                            atomicAdd(&p.fx[i], fx);
                            atomicAdd(&p.fy[i], fy);
                            atomicAdd(&p.fz[i], fz);
                            atomicAdd(&p.fx[j], -fx);
                            atomicAdd(&p.fy[j], -fy);
                            atomicAdd(&p.fz[j], -fz);

                            double rix = -nxn * p.radius[i];
                            double riy = -nyn * p.radius[i];
                            double riz = -nzn * p.radius[i];
                            double rjx = nxn * p.radius[j];
                            double rjy = nyn * p.radius[j];
                            double rjz = nzn * p.radius[j];
                            double torque_x = 0.0, torque_y = 0.0, torque_z = 0.0;
                            cross(rix, riy, riz, tx, ty, tz, torque_x, torque_y, torque_z);
                            atomicAdd(&p.tx[i], torque_x);
                            atomicAdd(&p.ty[i], torque_y);
                            atomicAdd(&p.tz[i], torque_z);
                            cross(rjx, rjy, rjz, -tx, -ty, -tz, torque_x, torque_y, torque_z);
                            atomicAdd(&p.tx[j], torque_x);
                            atomicAdd(&p.ty[j], torque_y);
                            atomicAdd(&p.tz[j], torque_z);

                            atomicAdd(&p.contact_count[i], 1);
                            atomicAdd(&p.contact_count[j], 1);
                        }
                    }

                    j = cells.particle_next[j];
                }
            }
        }
    }
}

__global__ void computeWallForces_kernel(
    ParticleData p,
    const SpaceData space,
    double stiffness,
    double damping)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.n || !p.active[idx]) return;

    int cid = nearestCellId(p.x[idx], p.y[idx], p.z[idx], space.nx, space.ny, space.nz);
    if (cid < 0) return;

    int cx = 0, cy = 0, cz = 0;
    if (!decodeCellId(cid, space.nx, space.ny, space.nz, cx, cy, cz)) return;

    const int dirs[6][3] = {
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0},
        { 0,  0,  1}, { 0,  0, -1}
    };

    for (int d = 0; d < 6; ++d) {
        int sx = cx + dirs[d][0];
        int sy = cy + dirs[d][1];
        int sz = cz + dirs[d][2];
        bool solid_neighbor =
            sx < 0 || sx >= space.nx ||
            sy < 0 || sy >= space.ny ||
            sz < 0 || sz >= space.nz;

        if (!solid_neighbor) {
            int sid = cellId(sx, sy, sz, space.nx, space.ny);
            solid_neighbor = space.d_cell_type[sid] == SOLID;
        }

        if (!solid_neighbor) continue;

        double nxn = -static_cast<double>(dirs[d][0]);
        double nyn = -static_cast<double>(dirs[d][1]);
        double nzn = -static_cast<double>(dirs[d][2]);
        double center =
            dirs[d][0] != 0 ? static_cast<double>(cx) :
            dirs[d][1] != 0 ? static_cast<double>(cy) :
                              static_cast<double>(cz);
        double pos =
            dirs[d][0] != 0 ? p.x[idx] :
            dirs[d][1] != 0 ? p.y[idx] :
                              p.z[idx];
        double plane = center + 0.5 * static_cast<double>(dirs[d][0] + dirs[d][1] + dirs[d][2]);
        double signed_distance = (plane - pos) * static_cast<double>(dirs[d][0] + dirs[d][1] + dirs[d][2]);
        double overlap = p.radius[idx] - signed_distance;

        if (overlap <= 0.0) continue;

        double vn = p.vx[idx] * nxn + p.vy[idx] * nyn + p.vz[idx] * nzn;
        double mag = stiffness * overlap - damping * vn;
        if (mag < 0.0) mag = 0.0;

        p.fx[idx] += mag * nxn;
        p.fy[idx] += mag * nyn;
        p.fz[idx] += mag * nzn;
    }
}

} // namespace

void ForceModel::resetFluidForces(FluidData& fluid, int num_cells)
{
    int block = kCudaBlockSize;
    int grid = (num_cells + block - 1) / block;
    if (grid <= 0) return;

    resetFluidForces_kernel<<<grid, block>>>(
        fluid.force_x,
        fluid.force_y,
        fluid.force_z,
        num_cells
    );
}

void ForceModel::injectParticles(
    ParticleData& p,
    const SpaceData& space,
    const FluidData& fluid,
    int step,
    int rbc_count,
    int platelet_count,
    int leukocyte_count)
{
    int total = rbc_count + platelet_count + leukocyte_count;
    if (p.n <= 0 || total <= 0) {
        return;
    }

    int block = kCudaBlockSize;
    int grid = (total + block - 1) / block;

    injectParticles_kernel<<<grid, block>>>(
        p,
        space.d_inlet_ids,
        space.num_inlets,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        space.d_normals_x,
        space.d_normals_y,
        space.d_normals_z,
        space.nx,
        space.ny,
        space.nz,
        step,
        rbc_count,
        platelet_count,
        leukocyte_count
    );
}

void ForceModel::computeFluidForces(
    ParticleData& p,
    FluidData& fluid,
    const SpaceData& space,
    const ParticleForceParameters& params)
{
    if (p.n <= 0) {
        return;
    }

    int block = kCudaBlockSize;
    int grid = (p.n + block - 1) / block;

    computeDragAndFluidReaction_kernel<<<grid, block>>>(
        p,
        fluid.density,
        fluid.velocity_x,
        fluid.velocity_y,
        fluid.velocity_z,
        fluid.force_x,
        fluid.force_y,
        fluid.force_z,
        space,
        params.tau,
        params.max_particle_force,
        params.fluid_reaction_scale
    );
}

void ForceModel::computeParticleForces(
    ParticleData& p,
    ParticleCellList& cells,
    const SpaceData& space,
    const ParticleForceParameters& params)
{
    if (p.n <= 0 || cells.cell_head == nullptr) {
        return;
    }

    int block = kCudaBlockSize;
    int grid = (p.n + block - 1) / block;

    computeContactForces_kernel<<<grid, block>>>(
        p,
        cells,
        space,
        params.contact_stiffness,
        params.contact_damping,
        params.friction
    );
}

void ForceModel::computeWallForces(
    ParticleData& p,
    const SpaceData& space,
    const ParticleForceParameters& params)
{
    if (p.n <= 0) {
        return;
    }

    int block = kCudaBlockSize;
    int grid = (p.n + block - 1) / block;

    computeWallForces_kernel<<<grid, block>>>(
        p,
        space,
        params.wall_stiffness,
        params.wall_damping
    );
}

void ForceModel::computeParticleForces(ParticleData& p, double dt)
{
    (void)p;
    (void)dt;
}

void ForceModel::computeWallForces(ParticleData& p)
{
    (void)p;
}

void ForceModel::computeFluidForces(ParticleData& p, FluidData& fluid)
{
    (void)p;
    (void)fluid;
}
