#include "space_system.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

    double *normals_x = new double[space_data.num_cells];
    double *normals_y = new double[space_data.num_cells];
    double *normals_z = new double[space_data.num_cells];

    for (int i = 0; i < space_data.num_cells; ++i) {
        std::size_t id = static_cast<std::size_t>(i);
        normals_x[id] = space_data.h_normals[id * 3 + 0];
        normals_y[id] = space_data.h_normals[id * 3 + 1];
        normals_z[id] = space_data.h_normals[id * 3 + 2];
    }

    cudaMalloc(reinterpret_cast<void**>(&space_data.d_normals_x), normal_component_bytes);
    cudaMemcpy(space_data.d_normals_x, normals_x, normal_component_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&space_data.d_normals_y), normal_component_bytes);
    cudaMemcpy(space_data.d_normals_y, normals_y, normal_component_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&space_data.d_normals_z), normal_component_bytes);
    cudaMemcpy(space_data.d_normals_z, normals_z, normal_component_bytes, cudaMemcpyHostToDevice);

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
