#pragma once

#include "space_data.cuh"

class SpaceSystem {
public: 
    SpaceSystem();
    ~SpaceSystem();

    SpaceSystem(const SpaceSystem&) = delete;
    SpaceSystem& operator=(const SpaceSystem&) = delete;

    bool initialize(const char* mesh_file, double dt);
    
    const SpaceData& data() const;

    inline int idx(int i, int j, int k)
    {
        return k * space_data.nx * space_data.ny + j * space_data.nx + i;
    }

private:
    SpaceData space_data;
    
    bool loadVoxelDomain(const char* filename);
    bool buildBoundaryIdLists();
    bool buildOutletSourceIds();
    
};
