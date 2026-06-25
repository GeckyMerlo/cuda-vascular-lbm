#include "space_data.cuh"
#include "../msh_utils/mesh_reader.hpp"

/* Space System 
TODO: Implement space system functionality, initialize space data from the mesh file.
*/

class SpaceSystem {
public: 
    SpaceSystem();
    ~SpaceSystem();

    void initialize(const char* mesh_file,
        double dt); // temporal step
    
    const SpaceData& data() const;

    inline int idx(int i, int j, int k)
    {
        return k * space_data.nx * space_data.ny + j * space_data.nx + i;
    }

private:
    SpaceData space_data;
    
    bool loadVoxelDomain(const char* filename);
    
};