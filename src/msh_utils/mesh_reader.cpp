#include "mesh_reader.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

static BoundaryType boundaryTypeFromName(const std::string& name)
{
    std::string s = name;

    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (s.find("wall") != std::string::npos)
        return BoundaryType::WALL;

    if (s.find("inlet") != std::string::npos)
        return BoundaryType::INLET;

    if (s.find("outlet") != std::string::npos)
        return BoundaryType::OUTLET;

    return BoundaryType::UNKNOWN;
}

static int nodesPerElement(int elementType)
{
    switch (elementType) {
        case 1: return 2;  // line
        case 2: return 3;  // triangle
        case 3: return 4;  // quadrangle
        case 4: return 4;  // tetrahedron
        case 5: return 8;  // hexahedron
        default:
            return 0;
    }
}

MeshData readMsh41Ascii(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Cannot open msh file: " + filename);

    MeshData mesh;

    std::unordered_map<int, BoundaryType> physicalTagToType;
    std::unordered_map<int, int> surfaceEntityToPhysicalTag;

    std::string token;

    while (file >> token) {

        if (token == "$MeshFormat") {
            double version;
            int fileType;
            int dataSize;

            file >> version >> fileType >> dataSize;

            if (version < 4.0 || version >= 5.0)
                throw std::runtime_error("Only Gmsh 4.x ASCII is supported");

            if (fileType != 0)
                throw std::runtime_error("Only ASCII .msh files are supported");

            file >> token; // $EndMeshFormat
        }

        else if (token == "$PhysicalNames") {
            int nPhysicalNames;
            file >> nPhysicalNames;

            for (int i = 0; i < nPhysicalNames; i++) {
                int dim;
                int tag;
                std::string name;

                file >> dim >> tag >> std::ws;
                std::getline(file, name);

                // remove quotes
                name.erase(std::remove(name.begin(), name.end(), '"'), name.end());

                if (dim == 2) {
                    physicalTagToType[tag] = boundaryTypeFromName(name);
                }
            }

            file >> token; // $EndPhysicalNames
        }

        else if (token == "$Entities") {
            int numPoints, numCurves, numSurfaces, numVolumes;
            file >> numPoints >> numCurves >> numSurfaces >> numVolumes;

            // points
            for (int i = 0; i < numPoints; i++) {
                std::string line;
                std::getline(file >> std::ws, line);
            }

            // curves
            for (int i = 0; i < numCurves; i++) {
                std::string line;
                std::getline(file >> std::ws, line);
            }

            // surfaces
            for (int i = 0; i < numSurfaces; i++) {
                std::string line;
                std::getline(file >> std::ws, line);

                std::stringstream ss(line);

                int entityTag;
                double minX, minY, minZ, maxX, maxY, maxZ;
                int numPhysicalTags;

                ss >> entityTag
                   >> minX >> minY >> minZ
                   >> maxX >> maxY >> maxZ
                   >> numPhysicalTags;

                if (numPhysicalTags > 0) {
                    int physicalTag;
                    ss >> physicalTag;

                    surfaceEntityToPhysicalTag[entityTag] = physicalTag;
                }
            }

            // volumes
            for (int i = 0; i < numVolumes; i++) {
                std::string line;
                std::getline(file >> std::ws, line);
            }

            file >> token; // $EndEntities
        }

        else if (token == "$Nodes") {
            std::size_t numEntityBlocks;
            std::size_t numNodes;
            std::size_t minNodeTag;
            std::size_t maxNodeTag;

            file >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;

            mesh.nodes.resize(maxNodeTag + 1);

            for (std::size_t b = 0; b < numEntityBlocks; b++) {
                int entityDim;
                int entityTag;
                int parametric;
                std::size_t numNodesInBlock;

                file >> entityDim >> entityTag >> parametric >> numNodesInBlock;

                std::vector<std::size_t> nodeTags(numNodesInBlock);

                for (std::size_t i = 0; i < numNodesInBlock; i++)
                    file >> nodeTags[i];

                for (std::size_t i = 0; i < numNodesInBlock; i++) {
                    double x, y, z;
                    file >> x >> y >> z;

                    mesh.nodes[nodeTags[i]] = {x, y, z};
                }
            }

            file >> token; // $EndNodes
        }

        else if (token == "$Elements") {
            std::size_t numEntityBlocks;
            std::size_t numElements;
            std::size_t minElementTag;
            std::size_t maxElementTag;

            file >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;

            for (std::size_t b = 0; b < numEntityBlocks; b++) {
                int entityDim;
                int entityTag;
                int elementType;
                std::size_t numElementsInBlock;

                file >> entityDim >> entityTag >> elementType >> numElementsInBlock;

                int npe = nodesPerElement(elementType);

                if (npe == 0) {
                    throw std::runtime_error("Unsupported Gmsh element type: " +
                                             std::to_string(elementType));
                }

                for (std::size_t e = 0; e < numElementsInBlock; e++) {
                    std::size_t elementTag;
                    file >> elementTag;

                    std::vector<std::size_t> elemNodes(npe);

                    for (int n = 0; n < npe; n++)
                        file >> elemNodes[n];

                    // surface triangle
                    if (entityDim == 2 && elementType == 2) {
                        int physicalTag = 0;

                        auto it = surfaceEntityToPhysicalTag.find(entityTag);

                        if (it != surfaceEntityToPhysicalTag.end())
                            physicalTag = it->second;

                        BoundaryType type = BoundaryType::UNKNOWN;

                        auto itType = physicalTagToType.find(physicalTag);

                        if (itType != physicalTagToType.end())
                            type = itType->second;

                        BoundaryTriangle tri{
                            static_cast<int>(elemNodes[0]),
                            static_cast<int>(elemNodes[1]),
                            static_cast<int>(elemNodes[2]),
                            physicalTag,
                            type
                        };

                        mesh.triangles.push_back(tri);

                        if (type == BoundaryType::WALL)
                            mesh.wall_triangles.push_back(tri);
                        else if (type == BoundaryType::INLET)
                            mesh.inlet_triangles.push_back(tri);
                        else if (type == BoundaryType::OUTLET)
                            mesh.outlet_triangles.push_back(tri);
                    }
                }
            }

            file >> token; // $EndElements
        }
    }

    std::cout << "Loaded mesh: " << filename << "\n";
    std::cout << "Nodes: " << mesh.nodes.size() << "\n";
    std::cout << "Surface triangles: " << mesh.triangles.size() << "\n";
    std::cout << "Wall triangles: " << mesh.wall_triangles.size() << "\n";
    std::cout << "Inlet triangles: " << mesh.inlet_triangles.size() << "\n";
    std::cout << "Outlet triangles: " << mesh.outlet_triangles.size() << "\n";

    return mesh;
}