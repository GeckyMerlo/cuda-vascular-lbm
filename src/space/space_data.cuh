enum CellType : int {
    FLUID  = 0,
    SOLID  = 1,
    INLET  = 2,
    OUTLET = 3
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
    int* h_outlet_ids;
    int* h_outlet_src_ids;
    
    int num_inlet_cells;
    int num_outlet_cells;

    int* d_inlet_ids;
    int* d_outlet_ids;
    int* d_outlet_src_ids;

};