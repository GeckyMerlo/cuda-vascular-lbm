enum CellType : int {
    FLUID  = 0,
    SOLID  = 1,
    INLET  = 2,
    OUTLET = 3
};

struct SpaceData {
    int nx, ny, nz;        // number of cells in each dimension
    double dx, dy, dz;         // spatial steps

    // boundary masks / flags
    CellType* h_cell_type;      // CPU: FLUID, SOLID, INLET, OUTLET
    CellType* d_cell_type;      // GPU copy

    double x0, y0, z0;

    int num_cells;

    int* h_inlet_ids;
    int* h_outlet_ids;
    int* d_inlet_ids;
    int* d_outlet_ids;

    int num_inlets;
    int num_outlets;

    double dt; // temporal step
};