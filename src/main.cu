#include "forces/force_model.hpp"
#include "lbm/lbm_system.hpp"
#include "particles/particle_system.hpp"
#include "space/space_system.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FluidStats {
    int active_cells = 0;
    int finite_cells = 0;
    int nonfinite_cells = 0;
    double avg_density = 0.0;
    double min_density = 0.0;
    double max_density = 0.0;
    double avg_ux = 0.0;
    double avg_uy = 0.0;
    double avg_uz = 0.0;
    double avg_speed = 0.0;
    double max_speed = 0.0;
    double avg_density_inlet = 0.0;
    double avg_density_outlet = 0.0;
    double avg_uz_inlet = 0.0;
    double avg_uz_outlet = 0.0;
    double mass_flux_in = 0.0;
    double mass_flux_out = 0.0;
    int inlet_cells = 0;
    int outlet_cells = 0;
};

struct ParticleStats {
    int active_particles = 0;
    int finite_particles = 0;
    int nonfinite_particles = 0;
    int injected_total = 0;
    int dropped_total = 0;
    int exited_total = 0;
    double avg_particle_speed = 0.0;
    double total_reaction_x = 0.0;
    double total_reaction_y = 0.0;
    double total_reaction_z = 0.0;
    double total_reaction_mag = 0.0;
};

struct ParticleConfig {
    int max_particles = 0;
    double rbc_rate = 1.0;
    double platelet_rate = 0.0;
    double leukocyte_rate = 0.0;
    int particle_output_interval = 0;
    double contact_stiffness = 0.05;
    double contact_damping = 0.02;
    double friction = 0.2;
    double wall_stiffness = 0.08;
    double wall_damping = 0.02;
};

struct RuntimeConfig {
    std::string mesh_file = "msh/voxel_domain.bin";
    int steps = 200;
    int output_interval = 20;
    double tau = 0.8;
    ParticleConfig particles;
};

struct InjectionAccumulator {
    double rbc = 0.0;
    double platelet = 0.0;
    double leukocyte = 0.0;
};

bool cudaOk(cudaError_t result, const char* operation)
{
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    return false;
}

bool startsWithDashDash(const std::string& value)
{
    return value.rfind("--", 0) == 0;
}

std::string optionValue(const std::string& arg, char** argv, int argc, int& i)
{
    const std::size_t equals = arg.find('=');
    if (equals != std::string::npos) {
        return arg.substr(equals + 1);
    }

    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for option: " + arg);
    }

    ++i;
    return argv[i];
}

RuntimeConfig parseArgs(int argc, char** argv)
{
    RuntimeConfig config;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (!startsWithDashDash(arg)) {
            positional.push_back(arg);
            continue;
        }

        std::string name = arg;
        const std::size_t equals = name.find('=');
        if (equals != std::string::npos) {
            name = name.substr(0, equals);
        }

        const std::string value = optionValue(arg, argv, argc, i);

        if (name == "--max-particles") {
            config.particles.max_particles = std::stoi(value);
        } else if (name == "--rbc-rate") {
            config.particles.rbc_rate = std::stod(value);
        } else if (name == "--platelet-rate") {
            config.particles.platelet_rate = std::stod(value);
        } else if (name == "--leukocyte-rate") {
            config.particles.leukocyte_rate = std::stod(value);
        } else if (name == "--particle-output-interval") {
            config.particles.particle_output_interval = std::stoi(value);
        } else if (name == "--contact-stiffness") {
            config.particles.contact_stiffness = std::stod(value);
        } else if (name == "--contact-damping") {
            config.particles.contact_damping = std::stod(value);
        } else if (name == "--friction") {
            config.particles.friction = std::stod(value);
        } else if (name == "--wall-stiffness") {
            config.particles.wall_stiffness = std::stod(value);
        } else if (name == "--wall-damping") {
            config.particles.wall_damping = std::stod(value);
        } else {
            throw std::runtime_error("Unknown option: " + name);
        }
    }

    if (positional.size() > 0) config.mesh_file = positional[0];
    if (positional.size() > 1) config.steps = std::stoi(positional[1]);
    if (positional.size() > 2) config.output_interval = std::stoi(positional[2]);
    if (positional.size() > 3) config.tau = std::stod(positional[3]);
    if (positional.size() > 4) {
        throw std::runtime_error("Too many positional arguments");
    }

    if (config.particles.particle_output_interval <= 0) {
        config.particles.particle_output_interval = config.output_interval;
    }

    return config;
}

int injectionCount(double rate, double& accumulator)
{
    if (rate <= 0.0) {
        return 0;
    }

    accumulator += rate;
    int count = static_cast<int>(std::floor(accumulator));
    accumulator -= static_cast<double>(count);
    return count;
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
    double min_density = std::numeric_limits<double>::infinity();
    double max_density = -std::numeric_limits<double>::infinity();

    for (int id = 0; id < space.num_cells; ++id) {
        if (space.h_cell_type[id] == SOLID) {
            continue;
        }

        stats.active_cells += 1;

        if (!std::isfinite(rho[id]) ||
            !std::isfinite(ux[id]) ||
            !std::isfinite(uy[id]) ||
            !std::isfinite(uz[id])) {
            stats.nonfinite_cells += 1;
            continue;
        }

        const double speed = std::sqrt(ux[id] * ux[id] + uy[id] * uy[id] + uz[id] * uz[id]);

        stats.finite_cells += 1;
        stats.avg_density += rho[id];
        min_density = std::min(min_density, rho[id]);
        max_density = std::max(max_density, rho[id]);
        stats.avg_ux += ux[id];
        stats.avg_uy += uy[id];
        stats.avg_uz += uz[id];
        stats.avg_speed += speed;
        stats.max_speed = std::max(stats.max_speed, speed);

        const auto type = space.h_cell_type[id];
        const double normal_dot_u =
            ux[id] * space.h_normals[3 * id] +
            uy[id] * space.h_normals[3 * id + 1] +
            uz[id] * space.h_normals[3 * id + 2];

        if (type == INLET) {
            const double inward_normal_speed = -normal_dot_u;
            stats.inlet_cells++;
            stats.avg_density_inlet += rho[id];
            stats.avg_uz_inlet += inward_normal_speed;
            stats.mass_flux_in += rho[id] * inward_normal_speed;
        }

        if (type == OUTLET) {
            stats.outlet_cells++;
            stats.avg_density_outlet += rho[id];
            stats.avg_uz_outlet += normal_dot_u;
            stats.mass_flux_out += rho[id] * normal_dot_u;
        }
    }

    if (stats.finite_cells > 0) {
        const double inv_cells = 1.0 / static_cast<double>(stats.finite_cells);
        stats.avg_density *= inv_cells;
        stats.avg_ux *= inv_cells;
        stats.avg_uy *= inv_cells;
        stats.avg_uz *= inv_cells;
        stats.avg_speed *= inv_cells;
        stats.min_density = min_density;
        stats.max_density = max_density;
    }

    if (stats.inlet_cells > 0) {
        stats.avg_density_inlet /= stats.inlet_cells;
        stats.avg_uz_inlet /= stats.inlet_cells;
    }

    if (stats.outlet_cells > 0) {
        stats.avg_density_outlet /= stats.outlet_cells;
        stats.avg_uz_outlet /= stats.outlet_cells;
    }

    return stats;
}

ParticleStats computeParticleStats(const ParticleData& particles, const FluidData& fluid, int num_cells)
{
    ParticleStats stats;
    if (particles.n <= 0) {
        return stats;
    }

    std::vector<int> active(static_cast<std::size_t>(particles.n));
    std::vector<double> vx(static_cast<std::size_t>(particles.n));
    std::vector<double> vy(static_cast<std::size_t>(particles.n));
    std::vector<double> vz(static_cast<std::size_t>(particles.n));

    if (!cudaOk(cudaMemcpy(active.data(), particles.active, active.size() * sizeof(int), cudaMemcpyDeviceToHost),
                "copy particle active flags") ||
        !cudaOk(cudaMemcpy(vx.data(), particles.vx, vx.size() * sizeof(double), cudaMemcpyDeviceToHost),
                "copy particle vx") ||
        !cudaOk(cudaMemcpy(vy.data(), particles.vy, vy.size() * sizeof(double), cudaMemcpyDeviceToHost),
                "copy particle vy") ||
        !cudaOk(cudaMemcpy(vz.data(), particles.vz, vz.size() * sizeof(double), cudaMemcpyDeviceToHost),
                "copy particle vz")) {
        throw std::runtime_error("Failed to copy particle statistics from GPU");
    }

    cudaMemcpy(&stats.injected_total, particles.injected_total, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&stats.dropped_total, particles.dropped_total, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&stats.exited_total, particles.exited_total, sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < particles.n; ++i) {
        if (!active[static_cast<std::size_t>(i)]) {
            continue;
        }

        stats.active_particles++;
        if (!std::isfinite(vx[static_cast<std::size_t>(i)]) ||
            !std::isfinite(vy[static_cast<std::size_t>(i)]) ||
            !std::isfinite(vz[static_cast<std::size_t>(i)])) {
            stats.nonfinite_particles++;
            continue;
        }

        stats.finite_particles++;
        stats.avg_particle_speed += std::sqrt(
            vx[static_cast<std::size_t>(i)] * vx[static_cast<std::size_t>(i)] +
            vy[static_cast<std::size_t>(i)] * vy[static_cast<std::size_t>(i)] +
            vz[static_cast<std::size_t>(i)] * vz[static_cast<std::size_t>(i)]);
    }

    if (stats.finite_particles > 0) {
        stats.avg_particle_speed /= static_cast<double>(stats.finite_particles);
    }

    std::vector<double> fx(static_cast<std::size_t>(num_cells));
    std::vector<double> fy(static_cast<std::size_t>(num_cells));
    std::vector<double> fz(static_cast<std::size_t>(num_cells));
    const std::size_t force_bytes = static_cast<std::size_t>(num_cells) * sizeof(double);

    if (!cudaOk(cudaMemcpy(fx.data(), fluid.force_x, force_bytes, cudaMemcpyDeviceToHost),
                "copy fluid force_x") ||
        !cudaOk(cudaMemcpy(fy.data(), fluid.force_y, force_bytes, cudaMemcpyDeviceToHost),
                "copy fluid force_y") ||
        !cudaOk(cudaMemcpy(fz.data(), fluid.force_z, force_bytes, cudaMemcpyDeviceToHost),
                "copy fluid force_z")) {
        throw std::runtime_error("Failed to copy fluid reaction statistics from GPU");
    }

    for (int id = 0; id < num_cells; ++id) {
        stats.total_reaction_x += fx[static_cast<std::size_t>(id)];
        stats.total_reaction_y += fy[static_cast<std::size_t>(id)];
        stats.total_reaction_z += fz[static_cast<std::size_t>(id)];
    }
    stats.total_reaction_mag = std::sqrt(
        stats.total_reaction_x * stats.total_reaction_x +
        stats.total_reaction_y * stats.total_reaction_y +
        stats.total_reaction_z * stats.total_reaction_z);

    return stats;
}

void writeStats(std::ofstream& output, int step, const FluidStats& fluid, const ParticleStats& particles)
{
    output
        << step << ','
        << fluid.active_cells << ','
        << fluid.finite_cells << ','
        << fluid.nonfinite_cells << ','
        << fluid.avg_density << ','
        << fluid.min_density << ','
        << fluid.max_density << ','
        << fluid.avg_ux << ','
        << fluid.avg_uy << ','
        << fluid.avg_uz << ','
        << fluid.avg_speed << ','
        << fluid.max_speed << ','
        << fluid.avg_density_inlet << ','
        << fluid.avg_density_outlet << ','
        << fluid.avg_uz_inlet << ','
        << fluid.avg_uz_outlet << ','
        << fluid.mass_flux_in << ','
        << fluid.mass_flux_out << ','
        << particles.active_particles << ','
        << particles.finite_particles << ','
        << particles.nonfinite_particles << ','
        << particles.injected_total << ','
        << particles.dropped_total << ','
        << particles.exited_total << ','
        << particles.avg_particle_speed << ','
        << particles.total_reaction_x << ','
        << particles.total_reaction_y << ','
        << particles.total_reaction_z << ','
        << particles.total_reaction_mag << '\n';
}

std::string paddedStep(int step)
{
    std::string s = std::to_string(step);
    return std::string(6 - std::min<int>(6, s.size()), '0') + s;
}

void writeFluidVTI(const std::string& filename, const SpaceData& space, const FluidData& fluid)
{
    std::vector<double> rho(space.num_cells);
    std::vector<double> ux(space.num_cells);
    std::vector<double> uy(space.num_cells);
    std::vector<double> uz(space.num_cells);

    const std::size_t bytes = static_cast<std::size_t>(space.num_cells) * sizeof(double);
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

    auto finiteOrZero = [](double value) {
        return std::isfinite(value) ? value : 0.0;
    };

    auto finiteCell = [&](int i) {
        return std::isfinite(rho[i]) &&
               std::isfinite(ux[i]) &&
               std::isfinite(uy[i]) &&
               std::isfinite(uz[i]);
    };

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"0 " << space.nx
        << " 0 " << space.ny
        << " 0 " << space.nz
        << "\" Origin=\""
        << space.x0 << " " << space.y0 << " " << space.z0
        << "\" Spacing=\""
        << space.dx << " " << space.dy << " " << space.dz << "\">\n";
    out << "    <Piece Extent=\"0 " << space.nx
        << " 0 " << space.ny
        << " 0 " << space.nz << "\">\n";
    out << "      <CellData Scalars=\"density\" Vectors=\"velocity\">\n";
    out << "        <DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        out << finiteOrZero(rho[i]) << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float64\" Name=\"speed\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        const double vx = finiteOrZero(ux[i]);
        const double vy = finiteOrZero(uy[i]);
        const double vz = finiteOrZero(uz[i]);
        const double speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        out << speed << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"finite\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        out << (finiteCell(i) ? 1 : 0) << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"cell_type\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        out << static_cast<int>(space.h_cell_type[i]) << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < space.num_cells; ++i) {
        out << finiteOrZero(ux[i]) << " "
            << finiteOrZero(uy[i]) << " "
            << finiteOrZero(uz[i]) << " ";
    }
    out << "\n        </DataArray>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

void writeParticleVTP(const std::string& filename, const SpaceData& space, const ParticleData& particles)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot open VTP file: " + filename);
    }

    if (particles.n <= 0) {
        out << "<?xml version=\"1.0\"?>\n";
        out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << "  <PolyData><Piece NumberOfPoints=\"0\" NumberOfVerts=\"0\"/></PolyData>\n";
        out << "</VTKFile>\n";
        return;
    }

    const std::size_t n = static_cast<std::size_t>(particles.n);
    std::vector<int> active(n), species(n), contact_count(n);
    std::vector<double> x(n), y(n), z(n), vx(n), vy(n), vz(n), fx(n), fy(n), fz(n);
    std::vector<double> wx(n), wy(n), wz(n), qw(n), qx(n), qy(n), qz(n), radius(n), a(n), b(n), c(n);

    cudaMemcpy(active.data(), particles.active, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(species.data(), particles.species, n * sizeof(ParticleSpecies), cudaMemcpyDeviceToHost);
    cudaMemcpy(contact_count.data(), particles.contact_count, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(x.data(), particles.x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), particles.y, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(z.data(), particles.z, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx.data(), particles.vx, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy.data(), particles.vy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz.data(), particles.vz, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fx.data(), particles.fx, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), particles.fy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), particles.fz, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(wx.data(), particles.wx, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(wy.data(), particles.wy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(wz.data(), particles.wz, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(qw.data(), particles.qw, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(qx.data(), particles.qx, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(qy.data(), particles.qy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(qz.data(), particles.qz, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(radius.data(), particles.radius, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(a.data(), particles.a, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), particles.b, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(c.data(), particles.c, n * sizeof(double), cudaMemcpyDeviceToHost);

    auto finiteOrZero = [](double value) {
        return std::isfinite(value) ? value : 0.0;
    };

    std::vector<int> ids;
    ids.reserve(n);
    for (int i = 0; i < particles.n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        if (active[idx] &&
            std::isfinite(x[idx]) &&
            std::isfinite(y[idx]) &&
            std::isfinite(z[idx])) {
            ids.push_back(i);
        }
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << ids.size() << "\" NumberOfVerts=\"" << ids.size() << "\">\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int id : ids) {
        const std::size_t i = static_cast<std::size_t>(id);
        out << space.x0 + (finiteOrZero(x[i]) + 0.5) * space.dx << " "
            << space.y0 + (finiteOrZero(y[i]) + 0.5) * space.dy << " "
            << space.z0 + (finiteOrZero(z[i]) + 0.5) * space.dz << " ";
    }
    out << "\n        </DataArray>\n";
    out << "      </Points>\n";
    out << "      <Verts>\n";
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < ids.size(); ++i) out << i << " ";
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < ids.size(); ++i) out << i + 1 << " ";
    out << "\n        </DataArray>\n";
    out << "      </Verts>\n";
    out << "      <PointData>\n";

    auto writeScalarInt = [&](const char* name, const std::vector<int>& values) {
        out << "        <DataArray type=\"Int32\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (int id : ids) out << values[static_cast<std::size_t>(id)] << " ";
        out << "\n        </DataArray>\n";
    };
    auto writeScalarDouble = [&](const char* name, const std::vector<double>& values) {
        out << "        <DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (int id : ids) out << finiteOrZero(values[static_cast<std::size_t>(id)]) << " ";
        out << "\n        </DataArray>\n";
    };
    auto writeVector = [&](const char* name, const std::vector<double>& vxv, const std::vector<double>& vyv, const std::vector<double>& vzv) {
        out << "        <DataArray type=\"Float64\" Name=\"" << name << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int id : ids) {
            const std::size_t i = static_cast<std::size_t>(id);
            out << finiteOrZero(vxv[i]) << " "
                << finiteOrZero(vyv[i]) << " "
                << finiteOrZero(vzv[i]) << " ";
        }
        out << "\n        </DataArray>\n";
    };

    writeScalarInt("species", species);
    writeScalarInt("active", active);
    writeScalarInt("contact_count", contact_count);
    writeScalarDouble("radius", radius);
    writeVector("velocity", vx, vy, vz);
    writeVector("force", fx, fy, fz);
    writeVector("angular_velocity", wx, wy, wz);
    writeVector("axes", a, b, c);
    out << "        <DataArray type=\"Float64\" Name=\"quaternion\" NumberOfComponents=\"4\" format=\"ascii\">\n";
    for (int id : ids) {
        const std::size_t i = static_cast<std::size_t>(id);
        out << finiteOrZero(qw[i]) << " "
            << finiteOrZero(qx[i]) << " "
            << finiteOrZero(qy[i]) << " "
            << finiteOrZero(qz[i]) << " ";
    }
    out << "\n        </DataArray>\n";
    out << "      </PointData>\n";
    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";
}

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config;
    try {
        config = parseArgs(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    const double dt = 1.0;

    if (config.steps < 0) {
        std::cerr << "steps must be non-negative\n";
        return 1;
    }

    if (config.output_interval <= 0) {
        std::cerr << "output_interval must be positive\n";
        return 1;
    }

    if (config.tau <= 0.5) {
        std::cerr << "tau must be greater than 0.5 for a stable BGK LBM run\n";
        return 1;
    }

    if (config.particles.max_particles < 0) {
        std::cerr << "--max-particles must be non-negative\n";
        return 1;
    }

    if (!cudaOk(cudaSetDevice(0), "cudaSetDevice")) {
        return 1;
    }

    SpaceSystem space;
    if (!space.initialize(config.mesh_file.c_str(), dt)) {
        return 1;
    }

    const SpaceData& domain = space.data();

    std::cout
        << "Loaded domain: "
        << domain.nx << " x " << domain.ny << " x " << domain.nz
        << " cells, inlets=" << domain.num_inlets
        << ", outlets=" << domain.num_outlets << '\n';

    LBMSystem lbm(domain, dt, config.tau);
    ForceModel force_model;
    ParticleSystem particle_system;
    const bool particles_enabled = config.particles.max_particles > 0;

    ParticleForceParameters force_params;
    force_params.tau = config.tau;
    force_params.contact_stiffness = config.particles.contact_stiffness;
    force_params.contact_damping = config.particles.contact_damping;
    force_params.friction = config.particles.friction;
    force_params.wall_stiffness = config.particles.wall_stiffness;
    force_params.wall_damping = config.particles.wall_damping;

    if (particles_enabled) {
        particle_system.allocate(config.particles.max_particles, domain.num_cells);
        std::cout
            << "Particles enabled: capacity=" << config.particles.max_particles
            << ", rbc_rate=" << config.particles.rbc_rate
            << ", platelet_rate=" << config.particles.platelet_rate
            << ", leukocyte_rate=" << config.particles.leukocyte_rate << '\n';
    }

    if (!cudaOk(cudaDeviceSynchronize(), "initial synchronization")) {
        return 1;
    }

    std::filesystem::create_directories("output");
    std::ofstream stats_file("output/lbm_stats.csv");
    if (!stats_file) {
        std::cerr << "Failed to open output/lbm_stats.csv\n";
        return 1;
    }

    stats_file
        << "step,active_cells,finite_cells,nonfinite_cells,"
        << "avg_density,min_density,max_density,avg_ux,avg_uy,avg_uz,avg_speed,max_speed,"
        << "avg_density_inlet,avg_density_outlet,avg_uz_inlet,avg_uz_outlet,"
        << "mass_flux_in,mass_flux_out,"
        << "active_particles,finite_particles,nonfinite_particles,"
        << "injected_total,dropped_total,exited_total,avg_particle_speed,"
        << "total_reaction_x,total_reaction_y,total_reaction_z,total_reaction_mag\n";

    writeStats(
        stats_file,
        0,
        computeFluidStats(domain, lbm.data()),
        particles_enabled ? computeParticleStats(particle_system.data(), lbm.data(), domain.num_cells)
                          : ParticleStats{});
    writeFluidVTI("output/fluid_000000.vti", domain, lbm.data());
    if (particles_enabled) {
        writeParticleVTP("output/particles_000000.vtp", domain, particle_system.data());
    }

    InjectionAccumulator injection;

    for (int step = 1; step <= config.steps; ++step) {
        if (particles_enabled) {
            const int rbc_count = injectionCount(config.particles.rbc_rate, injection.rbc);
            const int platelet_count = injectionCount(config.particles.platelet_rate, injection.platelet);
            const int leukocyte_count = injectionCount(config.particles.leukocyte_rate, injection.leukocyte);

            force_model.injectParticles(
                particle_system.data(),
                domain,
                lbm.data(),
                step,
                rbc_count,
                platelet_count,
                leukocyte_count);

            particle_system.resetForces();
            force_model.resetFluidForces(lbm.data(), domain.num_cells);
            particle_system.resetCellList();
            particle_system.buildCellList(domain.d_cell_type, domain.nx, domain.ny, domain.nz);
            force_model.computeFluidForces(particle_system.data(), lbm.data(), domain, force_params);
            force_model.computeParticleForces(particle_system.data(), particle_system.cellList(), domain, force_params);
            force_model.computeWallForces(particle_system.data(), domain, force_params);
            particle_system.updateVelocity(dt);
            particle_system.updateAngularVelocity(dt);
            particle_system.updatePosition(dt);
            particle_system.updateOrientation(dt);
            particle_system.deactivateExited(domain.d_cell_type, domain.nx, domain.ny, domain.nz);
            particle_system.swapForces();
        }

        lbm.step();

        if (!cudaOk(cudaGetLastError(), "simulation kernel launch") ||
            !cudaOk(cudaDeviceSynchronize(), "simulation step synchronization")) {
            return 1;
        }

        const bool fluid_output_due = step % config.output_interval == 0 || step == config.steps;
        const bool particle_output_due =
            particles_enabled &&
            (step % config.particles.particle_output_interval == 0 || step == config.steps);

        if (fluid_output_due) {
            const FluidStats fluid_stats = computeFluidStats(domain, lbm.data());
            const ParticleStats particle_stats =
                particles_enabled ? computeParticleStats(particle_system.data(), lbm.data(), domain.num_cells)
                                  : ParticleStats{};
            writeStats(stats_file, step, fluid_stats, particle_stats);

            const std::string filename = "output/fluid_" + paddedStep(step) + ".vti";
            writeFluidVTI(filename, domain, lbm.data());

            std::cout
                << "step " << step
                << " avg_uz=" << fluid_stats.avg_uz
                << " max_speed=" << fluid_stats.max_speed;
            if (fluid_stats.nonfinite_cells > 0) {
                std::cout << " nonfinite_cells=" << fluid_stats.nonfinite_cells;
            }
            if (particles_enabled) {
                std::cout << " active_particles=" << particle_stats.active_particles;
                if (particle_stats.nonfinite_particles > 0) {
                    std::cout << " nonfinite_particles=" << particle_stats.nonfinite_particles;
                }
            }
            std::cout << '\n';
        }

        if (particle_output_due) {
            const std::string filename = "output/particles_" + paddedStep(step) + ".vtp";
            writeParticleVTP(filename, domain, particle_system.data());
        }
    }

    std::cout << "Wrote output/lbm_stats.csv\n";
    return 0;
}
