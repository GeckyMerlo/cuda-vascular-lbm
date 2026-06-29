#pragma once

enum CellType : int {
    FLUID  = 0,
    SOLID  = 1,
    INLET  = 2,
    OUTLET = 3
};

struct SpaceData {
    int nx = 0, ny = 0, nz = 0;        // number of cells in each dimension
    double dx = 0.0, dy = 0.0, dz = 0.0;         // spatial steps

    // boundary masks / flags
    CellType* h_cell_type = nullptr;      // CPU: FLUID, SOLID, INLET, OUTLET
    CellType* d_cell_type = nullptr;      // GPU copy

    double x0 = 0.0, y0 = 0.0, z0 = 0.0;

    int num_cells = 0;

    int* h_inlet_ids = nullptr;
    int* h_outlet_ids = nullptr;
    int* h_outlet_src_ids = nullptr;
    int* d_inlet_ids = nullptr;
    int* d_outlet_ids = nullptr;
    int* d_outlet_src_ids = nullptr;

    int num_inlets = 0;
    int num_outlets = 0;
    int num_outlet_cells = 0;

    double dt = 0.0; // temporal step
};
