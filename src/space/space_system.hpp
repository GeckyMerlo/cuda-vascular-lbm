#include "space_data.cuh"

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
    
    void allocateHost();
    void classifyCellsFromMesh(const char* mesh_file);
    void detectInletOutlet();
    void buildOutletSrc();
    void copyToDevice();
};