#include "lbm/lbm_system.hpp"
#include "space/space_system.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FluidStats {
    int active_cells = 0;
    double avg_density = 0.0;
    double avg_ux = 0.0;
    double avg_uy = 0.0;
    double avg_uz = 0.0;
    double avg_speed = 0.0;
    double max_speed = 0.0;
};

bool cudaOk(cudaError_t result, const char* operation)
{
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    return false;
}

int parseIntArg(char** argv, int argc, int index, int default_value)
{
    if (argc <= index) {
        return default_value;
    }

    return std::stoi(argv[index]);
}

double parseDoubleArg(char** argv, int argc, int index, double default_value)
{
    if (argc <= index) {
        return default_value;
    }

    return std::stod(argv[index]);
}

FluidStats computeFluidStats(const SpaceData& space, const FluidData& fluid)
{
    std::vector<double> rho(static_cast<std::size_t>(space.num_cells));
    std::vector<double> ux(static_cast<std::size_t>(space.num_cells));
    std::vector<double> uy(static_cast<std::size_t>(space.num_cells));
    std::vector<double> uz(static_cast<std::size_t>(space.num_cells));

    const std::size_t bytes = static_cast<std::size_t>(space.num_cells) * sizeof(double);

    if (!cudaOk(cudaMemcpy(rho.data(), fluid.density, bytes, cudaMemcpyDeviceToHost),
                "copy density") ||
        !cudaOk(cudaMemcpy(ux.data(), fluid.velocity_x, bytes, cudaMemcpyDeviceToHost),
                "copy velocity_x") ||
        !cudaOk(cudaMemcpy(uy.data(), fluid.velocity_y, bytes, cudaMemcpyDeviceToHost),
                "copy velocity_y") ||
        !cudaOk(cudaMemcpy(uz.data(), fluid.velocity_z, bytes, cudaMemcpyDeviceToHost),
                "copy velocity_z")) {
        throw std::runtime_error("Failed to copy fluid statistics from GPU");
    }

    FluidStats stats;

    for (int id = 0; id < space.num_cells; ++id) {
        if (space.h_cell_type[id] == SOLID) {
            continue;
        }

        const double speed = std::sqrt(
            ux[static_cast<std::size_t>(id)] * ux[static_cast<std::size_t>(id)] +
            uy[static_cast<std::size_t>(id)] * uy[static_cast<std::size_t>(id)] +
            uz[static_cast<std::size_t>(id)] * uz[static_cast<std::size_t>(id)]);

        stats.active_cells += 1;
        stats.avg_density += rho[static_cast<std::size_t>(id)];
        stats.avg_ux += ux[static_cast<std::size_t>(id)];
        stats.avg_uy += uy[static_cast<std::size_t>(id)];
        stats.avg_uz += uz[static_cast<std::size_t>(id)];
        stats.avg_speed += speed;
        stats.max_speed = std::max(stats.max_speed, speed);
    }

    if (stats.active_cells > 0) {
        const double inv_cells = 1.0 / static_cast<double>(stats.active_cells);
        stats.avg_density *= inv_cells;
        stats.avg_ux *= inv_cells;
        stats.avg_uy *= inv_cells;
        stats.avg_uz *= inv_cells;
        stats.avg_speed *= inv_cells;
    }

    return stats;
}

void writeStats(std::ofstream& output, int step, const FluidStats& stats)
{
    output
        << step << ','
        << stats.active_cells << ','
        << stats.avg_density << ','
        << stats.avg_ux << ','
        << stats.avg_uy << ','
        << stats.avg_uz << ','
        << stats.avg_speed << ','
        << stats.max_speed << '\n';
}

std::string paddedStep(int step)
{
    std::string s = std::to_string(step);
    return std::string(6 - std::min<int>(6, s.size()), '0') + s;
}

void writeFluidVTI(
    const std::string& filename,
    const SpaceData& space,
    const FluidData& fluid

) {
    std::vector<double> rho(space.num_cells);
    std::vector<double> ux(space.num_cells);
    std::vector<double> uy(space.num_cells);
    std::vector<double> uz(space.num_cells);

    const std::size_t bytes =
        static_cast<std::size_t>(space.num_cells) * sizeof(double);
    if (!cudaOk(cudaMemcpy(rho.data(), fluid.density, bytes, cudaMemcpyDeviceToHost),
                "copy density for VTI") ||
        !cudaOk(cudaMemcpy(ux.data(), fluid.velocity_x, bytes, cudaMemcpyDeviceToHost),
                "copy velocity_x for VTI") ||
        !cudaOk(cudaMemcpy(uy.data(), fluid.velocity_y, bytes, cudaMemcpyDeviceToHost),
                "copy velocity_y for VTI") ||
        !cudaOk(cudaMemcpy(uz.data(), fluid.velocity_z, bytes, cudaMemcpyDeviceToHost),
                "copy velocity_z for VTI")) {
        throw std::runtime_error("Failed to copy fluid data for VTI");
    }
    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error("Cannot open VTI file: " + filename);
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"0 " << space.nx
        << " 0 " << space.ny
        << " 0 " << space.nz
        << "\" Origin=\"0 0 0\" Spacing=\""
        << space.dx << " " << space.dx << " " << space.dx << "\">\n";
    out << "    <Piece Extent=\"0 " << space.nx
        << " 0 " << space.ny
        << " 0 " << space.nz << "\">\n";
    out << "      <CellData Scalars=\"density\" Vectors=\"velocity\">\n";
    out << "        <DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        out << rho[i] << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float64\" Name=\"speed\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        const double speed = std::sqrt(ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i]);
        out << speed << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"cell_type\" format=\"ascii\">\n";

    for (int i = 0; i < space.num_cells; ++i) {
        out << static_cast<int>(space.h_cell_type[i]) << " ";
    }

    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float64\" Name=\"velocity\" "
        << "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        out << ux[i] << " " << uy[i] << " " << uz[i] << " ";
    }
    out << "\n        </DataArray>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

} // namespace

int main(int argc, char** argv)
{
    const std::string mesh_file = argc > 1 ? argv[1] : "msh/voxel_domain.bin";
    const int steps = parseIntArg(argv, argc, 2, 200);
    const int output_interval = parseIntArg(argv, argc, 3, 20);
    const double tau = parseDoubleArg(argv, argc, 4, 0.8);
    const double dt = 1.0;

    if (steps < 0) {
        std::cerr << "steps must be non-negative\n";
        return 1;
    }

    if (output_interval <= 0) {
        std::cerr << "output_interval must be positive\n";
        return 1;
    }

    if (tau <= 0.5) {
        std::cerr << "tau must be greater than 0.5 for a stable BGK LBM run\n";
        return 1;
    }

    if (!cudaOk(cudaSetDevice(0), "cudaSetDevice")) {
        return 1;
    }

    SpaceSystem space;
    if (!space.initialize(mesh_file.c_str(), dt)) {
        return 1;
    }

    const SpaceData& domain = space.data();

    std::cout
        << "Loaded domain: "
        << domain.nx << " x " << domain.ny << " x " << domain.nz
        << " cells, inlets=" << domain.num_inlets
        << ", outlets=" << domain.num_outlets << '\n';

    LBMSystem lbm(domain, dt, tau);

    if (!cudaOk(cudaDeviceSynchronize(), "initial LBM synchronization")) {
        return 1;
    }

    std::filesystem::create_directories("output");
    std::ofstream stats_file("output/lbm_stats.csv");
    if (!stats_file) {
        std::cerr << "Failed to open output/lbm_stats.csv\n";
        return 1;
    }

    stats_file << "step,active_cells,avg_density,avg_ux,avg_uy,avg_uz,avg_speed,max_speed\n";
    writeStats(stats_file, 0, computeFluidStats(domain, lbm.data()));
    writeFluidVTI("output/fluid_000000.vti", domain, lbm.data());

    for (int step = 1; step <= steps; ++step) {
        lbm.step();

        if (!cudaOk(cudaGetLastError(), "LBM kernel launch") ||
            !cudaOk(cudaDeviceSynchronize(), "LBM step synchronization")) {
            return 1;
        }

        if (step % output_interval == 0 || step == steps) {
            const FluidStats stats = computeFluidStats(domain, lbm.data());
            writeStats(stats_file, step, stats);

            const std::string filename = "output/fluid_" + paddedStep(step) + ".vti";
            writeFluidVTI(filename, domain, lbm.data());

            std::cout
                << "step " << step
                << " avg_uz=" << stats.avg_uz
                << " max_speed=" << stats.max_speed << '\n';
        }
    }

    std::cout << "Wrote output/lbm_stats.csv\n";
    return 0;
}
