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
        double dx, // spatial step
        double dt); // temporal step
    
    const SpaceData& data() const;

private:
    SpaceData space_data;
    
    void classifyCellsFromMesh(const char* mesh_file, double dx);
    void buildOutletSrc();
    int cellIdFromPosition(const Vec3& p) const;
    Vec3 cellCenter(int id) const;
    int findNearestFluidNeighbor(int id) const;
};