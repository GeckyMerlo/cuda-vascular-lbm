#include "space_system.hpp"
#include "space_utilities.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>


SpaceSystem::SpaceSystem() = default;

SpaceSystem::~SpaceSystem()
{
    delete[] space_data.h_cell_type;
    delete[] space_data.h_inlet_ids;
    delete[] space_data.h_outlet_ids;

    cudaFree(space_data.d_cell_type);
    cudaFree(space_data.d_inlet_ids);
    cudaFree(space_data.d_outlet_ids);
}

const SpaceData& SpaceSystem::data() const
{
    return space_data;
}

void SpaceSystem::initialize(const char* mesh_file, double dt)
{
    // Load the voxel domain from the mesh file
    if (!loadVoxelDomain(mesh_file)) {
        std::cerr << "Failed to load voxel domain from mesh file: " << mesh_file << std::endl;
        return;
    }

    // Set the temporal step
    space_data.dt = dt;

    // Additional initialization steps can be added here
}

bool SpaceSystem::loadVoxelDomain(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        return false;
    }

    file.read(reinterpret_cast<char*>(&space_data.nx), sizeof(int));
    file.read(reinterpret_cast<char*>(&space_data.ny), sizeof(int));
    file.read(reinterpret_cast<char*>(&space_data.nz), sizeof(int));

    file.read(reinterpret_cast<char*>(&space_data.dx), sizeof(double));
    file.read(reinterpret_cast<char*>(&space_data.dy), sizeof(double));
    file.read(reinterpret_cast<char*>(&space_data.dz), sizeof(double));

    file.read(reinterpret_cast<char*>(&space_data.x0), sizeof(double));
    file.read(reinterpret_cast<char*>(&space_data.y0), sizeof(double));
    file.read(reinterpret_cast<char*>(&space_data.z0), sizeof(double));

    space_data.num_cells =
        static_cast<size_t>(space_data.nx) *
        static_cast<size_t>(space_data.ny) *
        static_cast<size_t>(space_data.nz);

    cudaMallocHost(&space_data.h_cell_type,
                   space_data.num_cells * sizeof(CellType));

    file.read(reinterpret_cast<char*>(space_data.h_cell_type),
              space_data.num_cells * sizeof(CellType));

    cudaMalloc(&space_data.d_cell_type,
               space_data.num_cells * sizeof(CellType));

    cudaMemcpy(space_data.d_cell_type,
               space_data.h_cell_type,
               space_data.num_cells * sizeof(CellType),
               cudaMemcpyHostToDevice);

    file.read(reinterpret_cast<char*>(&space_data.num_inlets), sizeof(int));

    cudaMallocHost(&space_data.h_inlet_ids,
                   space_data.num_inlets * sizeof(int));

    file.read(reinterpret_cast<char*>(space_data.h_inlet_ids),
              space_data.num_inlets * sizeof(int));

    cudaMalloc(&space_data.d_inlet_ids,
               space_data.num_inlets * sizeof(int));

    cudaMemcpy(space_data.d_inlet_ids,
               space_data.h_inlet_ids,
               space_data.num_inlets * sizeof(int),
               cudaMemcpyHostToDevice);

    file.read(reinterpret_cast<char*>(&space_data.num_outlets), sizeof(int));

    cudaMallocHost(&space_data.h_outlet_ids,
                   space_data.num_outlets * sizeof(int));

    file.read(reinterpret_cast<char*>(space_data.h_outlet_ids),
              space_data.num_outlets * sizeof(int));

    cudaMalloc(&space_data.d_outlet_ids,
               space_data.num_outlets * sizeof(int));

    cudaMemcpy(space_data.d_outlet_ids,
               space_data.h_outlet_ids,
               space_data.num_outlets * sizeof(int),
               cudaMemcpyHostToDevice);

    return file.good();
}