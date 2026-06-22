#include "space_system.hpp"
#include "space_utilities.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

SpaceSystem::SpaceSystem() = default;

SpaceSystem::~SpaceSystem()
{
    delete[] space_data.h_cell_type;

    cudaFree(space_data.d_cell_type);
    cudaFree(space_data.d_inlet_ids);
    cudaFree(space_data.d_outlet_cells);
}

const SpaceData& SpaceSystem::data() const
{
    return space_data;
}

void SpaceSystem::initialize(const char* mesh_file, double dx, double dt)
{
    space_data.dx = dx;
    space_data.dt = dt;

    classifyCellsFromMesh(mesh_file, dx);
    buildOutletSrc();
}

void SpaceSystem::classifyCellsFromMesh(const char* mesh_file, double dx)
{
    MeshData mesh = readMsh41Ascii(mesh_file);

    Vec3 min_p{1e30, 1e30, 1e30};
    Vec3 max_p{-1e30, -1e30, -1e30};

    for (const auto& p : mesh.nodes) {
        min_p.x = std::min(min_p.x, p.x);
        min_p.y = std::min(min_p.y, p.y);
        min_p.z = std::min(min_p.z, p.z);

        max_p.x = std::max(max_p.x, p.x);
        max_p.y = std::max(max_p.y, p.y);
        max_p.z = std::max(max_p.z, p.z);
    }

    Vec3 mesh_center{
        0.5 * (min_p.x + max_p.x),
        0.5 * (min_p.y + max_p.y),
        0.5 * (min_p.z + max_p.z)
    };

    double padding = dx;

    space_data.origin = {
        min_p.x - padding,
        min_p.y - padding,
        min_p.z - padding
    };

    space_data.lx = (max_p.x - min_p.x) + 2.0 * padding;
    space_data.ly = (max_p.y - min_p.y) + 2.0 * padding;
    space_data.lz = (max_p.z - min_p.z) + 2.0 * padding;

    space_data.nx = static_cast<int>(std::ceil(space_data.lx / dx));
    space_data.ny = static_cast<int>(std::ceil(space_data.ly / dx));
    space_data.nz = static_cast<int>(std::ceil(space_data.lz / dx));

    space_data.num_cells =
        space_data.nx * space_data.ny * space_data.nz;

    space_data.h_cell_type = new CellType[space_data.num_cells];

    space_data.h_inlet_ids.clear();
    space_data.h_outlet_cells.clear();
    space_data.num_inlet_cells = 0;
    space_data.num_outlet_cells = 0;

    double wall_threshold   = 0.75 * dx;
    double inlet_threshold  = 1.25 * dx;
    double outlet_threshold = 1.25 * dx;

    for (int k = 0; k < space_data.nz; k++) {
        for (int j = 0; j < space_data.ny; j++) {
            for (int i = 0; i < space_data.nx; i++) {

                int id = i + space_data.nx * (j + space_data.ny * k);

                Vec3 p{
                    space_data.origin.x + (i + 0.5) * dx,
                    space_data.origin.y + (j + 0.5) * dx,
                    space_data.origin.z + (k + 0.5) * dx
                };

                bool inside = pointInsideMesh(p, mesh);

                if (!inside) {
                    space_data.h_cell_type[id] = SOLID;
                    continue;
                }

                if (nearTaggedSurface(p, mesh.inlet_triangles, mesh, inlet_threshold)) {
                    space_data.h_cell_type[id] = INLET;
                    space_data.h_inlet_ids.push_back(id);
                }
                else if (nearTaggedSurface(p, mesh.outlet_triangles, mesh, outlet_threshold)) {
                    space_data.h_cell_type[id] = OUTLET;

                    Vec3 n = nearestTaggedSurfaceNormal(
                        p,
                        mesh.outlet_triangles,
                        mesh,
                        mesh_center
                    );

                    OutletCell outlet;
                    outlet.id = id;
                    outlet.neighbor_inside = -1;
                    outlet.nx = n.x;
                    outlet.ny = n.y;
                    outlet.nz = n.z;

                    space_data.h_outlet_cells.push_back(outlet);
                }
                else if (nearTaggedSurface(p, mesh.wall_triangles, mesh, wall_threshold)) {
                    space_data.h_cell_type[id] = SOLID;
                }
                else {
                    space_data.h_cell_type[id] = FLUID;
                }
            }
        }
    }

    space_data.num_inlet_cells =
        static_cast<int>(space_data.h_inlet_ids.size());

    space_data.num_outlet_cells =
        static_cast<int>(space_data.h_outlet_cells.size());

    cudaMalloc(
        &space_data.d_cell_type,
        space_data.num_cells * sizeof(CellType)
    );

    cudaMemcpy(
        space_data.d_cell_type,
        space_data.h_cell_type,
        space_data.num_cells * sizeof(CellType),
        cudaMemcpyHostToDevice
    );

    cudaMalloc(
        &space_data.d_inlet_ids,
        space_data.num_inlet_cells * sizeof(int)
    );

    cudaMemcpy(
        space_data.d_inlet_ids,
        space_data.h_inlet_ids.data(),
        space_data.num_inlet_cells * sizeof(int),
        cudaMemcpyHostToDevice
    );
}


int SpaceSystem::cellIdFromPosition(const Vec3& p) const
{
    int i = static_cast<int>(
        std::floor((p.x - space_data.origin.x) / space_data.dx)
    );

    int j = static_cast<int>(
        std::floor((p.y - space_data.origin.y) / space_data.dx)
    );

    int k = static_cast<int>(
        std::floor((p.z - space_data.origin.z) / space_data.dx)
    );

    if (i < 0 || i >= space_data.nx ||
        j < 0 || j >= space_data.ny ||
        k < 0 || k >= space_data.nz)
        return -1;

    return i + space_data.nx * (j + space_data.ny * k);
}

Vec3 SpaceSystem::cellCenter(int id) const
{
    int i = id % space_data.nx;
    int j = (id / space_data.nx) % space_data.ny;
    int k = id / (space_data.nx * space_data.ny);

    return {
        space_data.origin.x + (i + 0.5) * space_data.dx,
        space_data.origin.y + (j + 0.5) * space_data.dx,
        space_data.origin.z + (k + 0.5) * space_data.dx
    };
}

int SpaceSystem::findNearestFluidNeighbor(int id) const
{
    const int nx = space_data.nx;
    const int ny = space_data.ny;
    const int nz = space_data.nz;

    int x = id % nx;
    int y = (id / nx) % ny;
    int z = id / (nx * ny);

    const int dirs[26][3] = {
        {-1,-1,-1}, {0,-1,-1}, {1,-1,-1},
        {-1, 0,-1}, {0, 0,-1}, {1, 0,-1},
        {-1, 1,-1}, {0, 1,-1}, {1, 1,-1},

        {-1,-1, 0}, {0,-1, 0}, {1,-1, 0},
        {-1, 0, 0},             {1, 0, 0},
        {-1, 1, 0}, {0, 1, 0}, {1, 1, 0},

        {-1,-1, 1}, {0,-1, 1}, {1,-1, 1},
        {-1, 0, 1}, {0, 0, 1}, {1, 0, 1},
        {-1, 1, 1}, {0, 1, 1}, {1, 1, 1}
    };

    for (const auto& d : dirs) {
        int xn = x + d[0];
        int yn = y + d[1];
        int zn = z + d[2];

        if (xn < 0 || xn >= nx ||
            yn < 0 || yn >= ny ||
            zn < 0 || zn >= nz)
            continue;

        int nid = xn + nx * (yn + ny * zn);

        if (space_data.h_cell_type[nid] == FLUID)
            return nid;
    }

    return -1;
}

void SpaceSystem::buildOutletSrc()
{
    for (auto& outlet : space_data.h_outlet_cells) {

        Vec3 p = cellCenter(outlet.id);

        Vec3 inside_p{
            p.x - space_data.dx * outlet.nx,
            p.y - space_data.dx * outlet.ny,
            p.z - space_data.dx * outlet.nz
        };

        int src_id = cellIdFromPosition(inside_p);

        if (src_id >= 0 && space_data.h_cell_type[src_id] == FLUID) {
            outlet.neighbor_inside = src_id;
        }
        else {
            outlet.neighbor_inside = findNearestFluidNeighbor(outlet.id);
        }

        if (outlet.neighbor_inside == -1) {
            std::cerr << "Warning: outlet cell "
                      << outlet.id
                      << " has no neighboring FLUID cell\n";
        }
    }

    cudaMalloc(
        &space_data.d_outlet_cells,
        space_data.num_outlet_cells * sizeof(OutletCell)
    );

    cudaMemcpy(
        space_data.d_outlet_cells,
        space_data.h_outlet_cells.data(),
        space_data.num_outlet_cells * sizeof(OutletCell),
        cudaMemcpyHostToDevice
    );
}

