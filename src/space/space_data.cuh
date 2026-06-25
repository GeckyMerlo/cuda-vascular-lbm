enum CellType : int {
    FLUID  = 0,
    SOLID  = 1,
    INLET  = 2,
    OUTLET = 3
};

struct OutletCell {
    int id;
    int neighbor_inside;
    double nx, ny, nz;
};

struct SpaceData {
    int nx, ny, nz;        // number of cells in each dimension
    double dx, dt;         // spatial/temporal step
    double lx, ly, lz;     // physical dimensions of the domain
    int num_cells;

    // boundary masks / flags
    CellType* h_cell_type;      // CPU: FLUID, SOLID, INLET, OUTLET
    CellType* d_cell_type;      // GPU copy

    // inlet/outlet info
    int* h_inlet_ids;
    OutletCell* h_outlet_cells;

    int num_inlet_cells = 0;
    int num_outlet_cells = 0;

    Vec3 origin;

    int* d_inlet_ids;
    OutletCell* d_outlet_cells;
};