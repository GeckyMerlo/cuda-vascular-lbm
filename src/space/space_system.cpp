#include "space_system.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace {

bool cudaOk(cudaError_t result, const char* operation)
{
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    return false;
}

bool readBytes(std::ifstream& file, void* dst, std::size_t bytes)
{
    file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return file.good();
}

template <typename T>
bool readValue(std::ifstream& file, T& value)
{
    return readBytes(file, &value, sizeof(T));
}

bool isValidCellId(int id, const SpaceData& space)
{
    return id >= 0 && id < space.num_cells;
}

int cellId(int x, int y, int z, const SpaceData& space)
{
    return z * space.nx * space.ny + y * space.nx + x;
}

bool decodeCellId(int id, const SpaceData& space, int& x, int& y, int& z)
{
    if (!isValidCellId(id, space)) {
        return false;
    }

    const int xy = space.nx * space.ny;
    z = id / xy;
    y = (id % xy) / space.nx;
    x = id % space.nx;
    return true;
}

void freeHost(void* ptr)
{
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

void freeDevice(void* ptr)
{
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

bool allocateAndCopyIds(
    int*& h_ids,
    int*& d_ids,
    const std::vector<int>& ids,
    const char* name)
{
    h_ids = nullptr;
    d_ids = nullptr;

    if (ids.empty()) {
        return true;
    }

    const std::size_t bytes = ids.size() * sizeof(int);

    if (!cudaOk(cudaMallocHost(reinterpret_cast<void**>(&h_ids), bytes), name)) {
        return false;
    }

    std::copy(ids.begin(), ids.end(), h_ids);

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_ids), bytes), name)) {
        return false;
    }

    return cudaOk(cudaMemcpy(d_ids, h_ids, bytes, cudaMemcpyHostToDevice), name);
}

bool inferInteriorDirection(
    int id,
    const SpaceData& space,
    bool fluid_only,
    double& dir_x,
    double& dir_y,
    double& dir_z)
{
    int x = 0;
    int y = 0;
    int z = 0;
    if (!decodeCellId(id, space, x, y, z)) {
        return false;
    }

    const CellType boundary_type = space.h_cell_type[id];
    const int dirs[][3] = {
        {0, 0, -1},
        {0, 0, 1},
        {-1, 0, 0},
        {1, 0, 0},
        {0, -1, 0},
        {0, 1, 0},
    };

    dir_x = 0.0;
    dir_y = 0.0;
    dir_z = 0.0;

    for (const auto& d : dirs) {
        const int nx = x + d[0];
        const int ny = y + d[1];
        const int nz = z + d[2];
        if (nx < 0 || nx >= space.nx ||
            ny < 0 || ny >= space.ny ||
            nz < 0 || nz >= space.nz) {
            continue;
        }

        const int neighbor_id = cellId(nx, ny, nz, space);
        const CellType neighbor_type = space.h_cell_type[neighbor_id];
        if (fluid_only) {
            if (neighbor_type != FLUID) {
                continue;
            }
        } else if (neighbor_type == SOLID || neighbor_type == boundary_type) {
            continue;
        }

        dir_x += static_cast<double>(d[0]);
        dir_y += static_cast<double>(d[1]);
        dir_z += static_cast<double>(d[2]);
    }

    const double norm = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    if (norm <= 1e-12) {
        return false;
    }

    dir_x /= norm;
    dir_y /= norm;
    dir_z /= norm;
    return true;
}

void repairBoundaryNormals(SpaceData& space)
{
    if (space.h_normals == nullptr) {
        return;
    }

    for (int id = 0; id < space.num_cells; ++id) {
        const CellType type = space.h_cell_type[id];
        if (type != INLET && type != OUTLET) {
            continue;
        }

        double inside_x = 0.0;
        double inside_y = 0.0;
        double inside_z = 0.0;
        bool has_inside = inferInteriorDirection(id, space, true, inside_x, inside_y, inside_z);
        if (!has_inside) {
            has_inside = inferInteriorDirection(id, space, false, inside_x, inside_y, inside_z);
        }

        double normal_x = space.h_normals[3 * id];
        double normal_y = space.h_normals[3 * id + 1];
        double normal_z = space.h_normals[3 * id + 2];
        const double normal_norm =
            std::sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);

        if (normal_norm > 1e-12) {
            normal_x /= normal_norm;
            normal_y /= normal_norm;
            normal_z /= normal_norm;

            if (has_inside &&
                normal_x * inside_x + normal_y * inside_y + normal_z * inside_z > 0.0) {
                normal_x = -normal_x;
                normal_y = -normal_y;
                normal_z = -normal_z;
            }
        } else if (has_inside) {
            normal_x = -inside_x;
            normal_y = -inside_y;
            normal_z = -inside_z;
        }

        space.h_normals[3 * id] = normal_x;
        space.h_normals[3 * id + 1] = normal_y;
        space.h_normals[3 * id + 2] = normal_z;
    }
}

int findOutletSourceId(int outlet_id, const SpaceData& space)
{
    const int xy = space.nx * space.ny;
    const int z = outlet_id / xy;
    const int y = (outlet_id % xy) / space.nx;
    const int x = outlet_id % space.nx;

    const int candidates[][3] = {
        {x, y, z - 1},
        {x, y, z + 1},
        {x - 1, y, z},
        {x + 1, y, z},
        {x, y - 1, z},
        {x, y + 1, z},
    };

    double inward_x = 0.0;
    double inward_y = 0.0;
    double inward_z = 0.0;
    bool has_normal = false;

    if (space.h_normals != nullptr) {
        const double normal_x = space.h_normals[3 * outlet_id];
        const double normal_y = space.h_normals[3 * outlet_id + 1];
        const double normal_z = space.h_normals[3 * outlet_id + 2];
        const double normal_norm =
            std::sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);

        if (normal_norm > 1e-12) {
            inward_x = -normal_x / normal_norm;
            inward_y = -normal_y / normal_norm;
            inward_z = -normal_z / normal_norm;
            has_normal = true;
        }
    }

    auto chooseCandidate = [&](bool fluid_only) {
        int best_id = -1;
        double best_score = -std::numeric_limits<double>::infinity();

        for (const auto& c : candidates) {
            if (c[0] < 0 || c[0] >= space.nx ||
                c[1] < 0 || c[1] >= space.ny ||
                c[2] < 0 || c[2] >= space.nz) {
                continue;
            }

            const int id = cellId(c[0], c[1], c[2], space);
            if (fluid_only) {
                if (space.h_cell_type[id] != FLUID) {
                    continue;
                }
            } else if (space.h_cell_type[id] == SOLID ||
                       space.h_cell_type[id] == OUTLET) {
                continue;
            }

            double score = 0.0;
            if (has_normal) {
                score =
                    static_cast<double>(c[0] - x) * inward_x +
                    static_cast<double>(c[1] - y) * inward_y +
                    static_cast<double>(c[2] - z) * inward_z;
            }

            if (score > best_score) {
                best_score = score;
                best_id = id;
            }
        }

        return best_id;
    };

    const int normal_aligned_fluid = chooseCandidate(true);
    if (normal_aligned_fluid >= 0) {
        return normal_aligned_fluid;
    }

    const int normal_aligned_open = chooseCandidate(false);
    if (normal_aligned_open >= 0) {
        return normal_aligned_open;
    }

    for (const auto& c : candidates) {
        if (c[0] < 0 || c[0] >= space.nx ||
            c[1] < 0 || c[1] >= space.ny ||
            c[2] < 0 || c[2] >= space.nz) {
            continue;
        }

        const int id = cellId(c[0], c[1], c[2], space);
        if (space.h_cell_type[id] == FLUID) {
            return id;
        }
    }

    for (const auto& c : candidates) {
        if (c[0] < 0 || c[0] >= space.nx ||
            c[1] < 0 || c[1] >= space.ny ||
            c[2] < 0 || c[2] >= space.nz) {
            continue;
        }

        const int id = cellId(c[0], c[1], c[2], space);
        if (space.h_cell_type[id] != SOLID && space.h_cell_type[id] != OUTLET) {
            return id;
        }
    }

    return outlet_id;
}

} // namespace

SpaceSystem::SpaceSystem() = default;

SpaceSystem::~SpaceSystem()
{
    freeHost(space_data.h_cell_type);
    freeHost(space_data.h_inlet_ids);
    freeHost(space_data.h_outlet_ids);
    freeHost(space_data.h_outlet_src_ids);

    freeDevice(space_data.d_cell_type);
    freeDevice(space_data.d_inlet_ids);
    freeDevice(space_data.d_outlet_ids);
    freeDevice(space_data.d_outlet_src_ids);

    freeHost(space_data.h_normals);
    freeDevice(space_data.d_normals_x);
    freeDevice(space_data.d_normals_y);
    freeDevice(space_data.d_normals_z);
}

const SpaceData& SpaceSystem::data() const
{
    return space_data;
}

bool SpaceSystem::initialize(const char* mesh_file, double dt)
{
    if (!loadVoxelDomain(mesh_file)) {
        std::cerr << "Failed to load voxel domain from mesh file: " << mesh_file << '\n';
        return false;
    }

    space_data.dt = dt;
    return true;
}

bool SpaceSystem::loadVoxelDomain(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        return false;
    }

    if (!readValue(file, space_data.nx) ||
        !readValue(file, space_data.ny) ||
        !readValue(file, space_data.nz) ||
        !readValue(file, space_data.dx) ||
        !readValue(file, space_data.dy) ||
        !readValue(file, space_data.dz) ||
        !readValue(file, space_data.x0) ||
        !readValue(file, space_data.y0) ||
        !readValue(file, space_data.z0)) {
        return false;
    }

    if (space_data.nx <= 0 || space_data.ny <= 0 || space_data.nz <= 0) {
        std::cerr << "Voxel domain has invalid dimensions\n";
        return false;
    }

    const std::int64_t num_cells =
        static_cast<std::int64_t>(space_data.nx) *
        static_cast<std::int64_t>(space_data.ny) *
        static_cast<std::int64_t>(space_data.nz);

    if (num_cells > std::numeric_limits<int>::max()) {
        std::cerr << "Voxel domain is too large for int cell indexing\n";
        return false;
    }

    space_data.num_cells = static_cast<int>(num_cells);

    const std::size_t cell_bytes =
        static_cast<std::size_t>(space_data.num_cells) * sizeof(CellType);

    if (!cudaOk(cudaMallocHost(reinterpret_cast<void**>(&space_data.h_cell_type), cell_bytes),
                "cudaMallocHost(h_cell_type)")) {
        return false;
    }

    if (!readBytes(file, space_data.h_cell_type, cell_bytes)) {
        return false;
    }

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&space_data.d_cell_type), cell_bytes),
                "cudaMalloc(d_cell_type)") ||
        !cudaOk(cudaMemcpy(space_data.d_cell_type,
                           space_data.h_cell_type,
                           cell_bytes,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy(d_cell_type)")) {
        return false;
    }

    const std::size_t normal_component_bytes =
        static_cast<std::size_t>(space_data.num_cells) * sizeof(double);

    const std::size_t normal_vector_bytes =
        static_cast<std::size_t>(space_data.num_cells) * 3 * sizeof(double);

    if (!cudaOk(cudaMallocHost(reinterpret_cast<void**>(&space_data.h_normals),
                               normal_vector_bytes),
                "cudaMallocHost(h_normals)")) {
        return false;
    }

    if (!readBytes(file, space_data.h_normals, normal_vector_bytes)) {
        return false;
    }

    repairBoundaryNormals(space_data);

    std::vector<double> normals_x(static_cast<std::size_t>(space_data.num_cells));
    std::vector<double> normals_y(static_cast<std::size_t>(space_data.num_cells));
    std::vector<double> normals_z(static_cast<std::size_t>(space_data.num_cells));

    for (int i = 0; i < space_data.num_cells; ++i) {
        std::size_t id = static_cast<std::size_t>(i);
        normals_x[id] = space_data.h_normals[id * 3 + 0];
        normals_y[id] = space_data.h_normals[id * 3 + 1];
        normals_z[id] = space_data.h_normals[id * 3 + 2];
    }

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&space_data.d_normals_x),
                           normal_component_bytes),
                "cudaMalloc(d_normals_x)") ||
        !cudaOk(cudaMemcpy(space_data.d_normals_x,
                           normals_x.data(),
                           normal_component_bytes,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy(d_normals_x)") ||
        !cudaOk(cudaMalloc(reinterpret_cast<void**>(&space_data.d_normals_y),
                           normal_component_bytes),
                "cudaMalloc(d_normals_y)") ||
        !cudaOk(cudaMemcpy(space_data.d_normals_y,
                           normals_y.data(),
                           normal_component_bytes,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy(d_normals_y)") ||
        !cudaOk(cudaMalloc(reinterpret_cast<void**>(&space_data.d_normals_z),
                           normal_component_bytes),
                "cudaMalloc(d_normals_z)") ||
        !cudaOk(cudaMemcpy(space_data.d_normals_z,
                           normals_z.data(),
                           normal_component_bytes,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy(d_normals_z)")) {
        return false;
    }

    int explicit_inlet_count = 0;
    if (readValue(file, explicit_inlet_count)) {
        if (explicit_inlet_count < 0) {
            std::cerr << "Voxel domain has a negative inlet count\n";
            return false;
        }

        std::vector<int> inlet_ids(static_cast<std::size_t>(explicit_inlet_count));
        if (!inlet_ids.empty() &&
            !readBytes(file, inlet_ids.data(), inlet_ids.size() * sizeof(int))) {
            return false;
        }

        int explicit_outlet_count = 0;
        if (!readValue(file, explicit_outlet_count)) {
            return false;
        }
        if (explicit_outlet_count < 0) {
            std::cerr << "Voxel domain has a negative outlet count\n";
            return false;
        }

        std::vector<int> outlet_ids(static_cast<std::size_t>(explicit_outlet_count));
        if (!outlet_ids.empty() &&
            !readBytes(file, outlet_ids.data(), outlet_ids.size() * sizeof(int))) {
            return false;
        }

        space_data.num_inlets = static_cast<int>(inlet_ids.size());
        space_data.num_outlets = static_cast<int>(outlet_ids.size());

        if (!allocateAndCopyIds(space_data.h_inlet_ids,
                                space_data.d_inlet_ids,
                                inlet_ids,
                                "inlet ids") ||
            !allocateAndCopyIds(space_data.h_outlet_ids,
                                space_data.d_outlet_ids,
                                outlet_ids,
                                "outlet ids")) {
            return false;
        }
    } else {
        file.clear();
        if (!buildBoundaryIdLists()) {
            return false;
        }
    }

    if (!buildOutletSourceIds()) {
        return false;
    }

    return true;
}

bool SpaceSystem::buildBoundaryIdLists()
{
    std::vector<int> inlet_ids;
    std::vector<int> outlet_ids;

    inlet_ids.reserve(static_cast<std::size_t>(space_data.num_cells / 100));
    outlet_ids.reserve(static_cast<std::size_t>(space_data.num_cells / 100));

    for (int id = 0; id < space_data.num_cells; ++id) {
        if (space_data.h_cell_type[id] == INLET) {
            inlet_ids.push_back(id);
        } else if (space_data.h_cell_type[id] == OUTLET) {
            outlet_ids.push_back(id);
        }
    }

    space_data.num_inlets = static_cast<int>(inlet_ids.size());
    space_data.num_outlets = static_cast<int>(outlet_ids.size());

    return allocateAndCopyIds(space_data.h_inlet_ids,
                              space_data.d_inlet_ids,
                              inlet_ids,
                              "inlet ids") &&
           allocateAndCopyIds(space_data.h_outlet_ids,
                              space_data.d_outlet_ids,
                              outlet_ids,
                              "outlet ids");
}

bool SpaceSystem::buildOutletSourceIds()
{

    if (space_data.num_outlets <= 0) {
        return true;
    }

    std::vector<int> source_ids(static_cast<std::size_t>(space_data.num_outlets));
    for (int i = 0; i < space_data.num_outlets; ++i) {
        const int outlet_id = space_data.h_outlet_ids[i];
        source_ids[static_cast<std::size_t>(i)] =
            isValidCellId(outlet_id, space_data) ? findOutletSourceId(outlet_id, space_data)
                                                 : outlet_id;
    }

    return allocateAndCopyIds(space_data.h_outlet_src_ids,
                              space_data.d_outlet_src_ids,
                              source_ids,
                              "outlet source ids");
}
