#pragma once

#include <vector>
#include <string>

enum class BoundaryType {
    UNKNOWN = 0,
    WALL,
    INLET,
    OUTLET
};

struct Vec3 {
    double x, y, z;
};

struct BoundaryTriangle {
    int v1, v2, v3;              // node indices
    int physical_tag;            // Gmsh physical tag
    BoundaryType type;           // WALL / INLET / OUTLET
};

struct MeshData {
    std::vector<Vec3> nodes;

    std::vector<BoundaryTriangle> triangles;

    std::vector<BoundaryTriangle> wall_triangles;
    std::vector<BoundaryTriangle> inlet_triangles;
    std::vector<BoundaryTriangle> outlet_triangles;
};