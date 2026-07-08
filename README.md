# Vascular LBM Blood-Particle Simulation

This project simulates blood flow in a vessel-scale domain using a CUDA
Lattice Boltzmann Method (LBM) fluid solver with optional rigid blood particles.
The current model is mesoscopic: it does not try to resolve all microscopic
cell membrane physics, but it represents blood cells as rigid particles that
move through and react back on the flow.

The default executable still runs a fluid-only simulation. Particle coupling is
enabled only when `--max-particles` is greater than zero.

## What The Simulation Models

### Mesoscopic Blood Flow

The model is intended for vessels whose diameter is larger than the particle
diameters. In that regime, red blood cells, platelets, and leukocytes can be
approximated as rigid bodies carried by the fluid, while the fluid itself is
solved on a fixed lattice.

The project currently represents:

- Fluid: D3Q19 Lattice Boltzmann Method.
- Red blood cells: rigid ellipsoid state for orientation/output, effective
  sphere radius for drag and contact in this first implementation.
- Platelets: small spheres.
- Leukocytes: larger spheres.
- Vessel geometry: voxelized domain loaded from `msh/*.bin`.

### Lattice Units

Most solver quantities are in lattice units:

- Grid cells are indexed as `(x, y, z)` lattice coordinates.
- One simulation step uses `dt = 1`.
- Particle position, velocity, force, torque, radius, mass, and inertia are
  stored in lattice units.
- Physical coordinates are only reconstructed for output using the voxel
  domain origin and spacing.

This keeps the numerical coupling simple and consistent with the LBM solver.

### Voxel Domain And Cell Types

The fluid domain is stored in a custom binary voxel file. Each voxel is one of:

- `FLUID = 0`
- `SOLID = 1`
- `INLET = 2`
- `OUTLET = 3`

The bundled primary domain is:

- `msh/voxel_domain.bin`
- Generated from `msh/vena_cilindrica.geo`
- Straight cylindrical vessel aligned with the z axis

The binary file stores:

- Grid dimensions.
- Grid spacing.
- Physical origin.
- Cell type mask.
- Boundary normals.
- Explicit inlet and outlet cell ID lists.

### D3Q19 LBM Fluid Solver

The fluid solver uses a D3Q19 lattice:

- `D3`: three spatial dimensions.
- `Q19`: nineteen lattice velocity directions per cell.

Each fluid step performs:

1. Collision with a BGK relaxation model.
2. Streaming along the D3Q19 directions.
3. Inlet boundary update.
4. Outlet boundary update.
5. Macroscopic reconstruction of density and velocity.

The equilibrium distribution is the standard second-order LBM equilibrium:

```text
f_eq = w_i * rho * (1 + 3(c_i.u) + 4.5(c_i.u)^2 - 1.5(u.u))
```

The relaxation parameter is `tau`. It must be greater than `0.5`; values too
close to `0.5` are usually less stable.

### Boundary Conditions

The current boundary conditions are simple and tuned for the bundled straight
z-aligned cylindrical vessel:

- Solid walls use bounce-back behavior during streaming.
- Inlet assumes a z-min inlet and imposes a positive z velocity.
- Outlet copies distributions from a neighboring interior source cell, giving a
  simple zero-gradient outflow.

Important limitation: the voxel file stores normals, but the active LBM inlet
velocity kernel is still axis-specific. Arbitrary vascular inlet orientation is
not fully generalized yet.

### Particle Species

Particles are stored in structure-of-arrays form on the GPU.

Each particle has:

- Species: `RBC`, `PLATELET`, or `LEUKOCYTE`.
- Active flag.
- Position and velocity.
- Force and previous force.
- Mass and moment of inertia.
- Effective radius.
- Shape enum.
- Quaternion orientation.
- Angular velocity.
- Torque.
- Ellipsoid axes `a`, `b`, `c`.
- Contact count diagnostic.

Default species parameters are defined in `src/forces/force_model.cpp`.

### Inlet Injection

Particles are not seeded throughout the whole domain. Instead, they are injected
through inlet cells during the simulation.

Injection behavior:

- The particle system preallocates a fixed capacity.
- Each step requests a configurable number of RBCs, platelets, and leukocytes.
- Fractional injection rates accumulate over time.
- Inlet cells are selected deterministically from the domain inlet ID list.
- Initial particle velocity is copied from the local fluid velocity.
- Particles are marked inactive when they leave through outlet cells or move
  outside the domain.

If the fixed capacity is full, additional injection attempts increment the
`dropped_total` statistic.

### Fluid-To-Particle Interpolation

The particle drag model needs local fluid velocity at particle centers. The
solver computes that with trilinear interpolation:

- The 8 neighboring lattice cells are sampled around the particle position.
- Solid cells are ignored.
- Weights are renormalized over the non-solid neighbors.
- Interpolated density and velocity are used by the drag model.

### Drag Force And Fluid Reaction

Particle drag uses a low-Reynolds Stokes-style model:

```text
F_drag = 6 * pi * mu * r * (u_fluid - v_particle)
```

Where:

- `mu = rho * (tau - 0.5) / 3` in lattice units.
- `r` is the particle effective radius.
- `u_fluid` is the interpolated fluid velocity.
- `v_particle` is the particle velocity.

The equal and opposite reaction force is deposited back into the fluid force
arrays using the same interpolation weights:

```text
F_fluid = -F_drag
```

The LBM collision step includes this body force using a Guo-style forcing term.
Macroscopic velocity reconstruction includes the half-force correction:

```text
u = (sum_i f_i c_i + 0.5 F) / rho
```

### Cell Lists

Particle-particle contact is accelerated with a GPU cell list:

- `cell_head[num_cells]`: head of each voxel's linked list.
- `particle_next[max_particles]`: next particle in the same cell.
- `particle_cell_id[max_particles]`: current cell of each particle.

Each step:

1. Cell heads are reset.
2. Active particles are assigned to their current voxel.
3. Contact checks traverse the particle's own cell and the 26 neighboring cells.

### Particle Contact

The first implementation uses sphere contact for all species:

- RBCs still carry ellipsoid axes and orientation for visualization.
- Contact uses the effective sphere radius.
- Overlap produces a spring-damper normal force.
- Tangential damping is capped by a friction coefficient.
- Tangential contact contributes torque.

This is intentionally simpler than true ellipsoid contact. It provides useful
collision behavior while keeping the model tractable.

### Rotation And Orientation

Particles have angular velocity and quaternion orientation.

Each coupled step:

1. Torque updates angular velocity.
2. Angular velocity updates the quaternion.
3. The quaternion is normalized.

RBC orientation can be visualized from the quaternion and ellipsoid axes, even
though contact and drag still use an effective radius.

### Wall Handling

Particles receive a minimal wall repulsion near solid neighboring cells. This
is a support mechanism to keep particles inside the vessel. It is not a full
wall-contact or lubrication model.

### Output

The solver writes:

- `output/fluid_XXXXXX.vti`: VTK ImageData fluid fields.
- `output/particles_XXXXXX.vtp`: VTK PolyData particle points, when particles
  are enabled.
- `output/lbm_stats.csv`: time-series fluid and particle diagnostics.

When `--warmup-steps` is greater than zero, the warmup advances the solver before
any output files or CSV rows are created. The first recorded `step = 0` output is
the post-warmup state.

Fluid fields include:

- Density.
- Speed.
- Finite-value mask for detecting invalid density or velocity cells.
- Cell type.
- Velocity vector.

Particle fields include:

- Species.
- Active flag.
- Contact count.
- Radius.
- Speed.
- Velocity.
- Force.
- Angular velocity.
- Axes.
- Quaternion.

## Repository Layout

```text
.
|-- CMakeLists.txt
|-- bibliography.txt
|-- msh/
|   |-- voxel_domain.bin
|   |-- cilindric_vessel_stenosis30_voxel_domain.bin
|   |-- vena_cilindrica.geo
|   `-- centering.py
`-- src/
    |-- main.cu
    |-- lbm/
    |-- particles/
    |-- forces/
    |-- space/
    |-- msh_utils/
    `-- mesh_preprocessing.py
```

Main subsystems:

- `src/main.cu`: CLI, simulation loop, statistics, output.
- `src/lbm`: D3Q19 LBM constants, fluid data, kernels, and solver wrapper.
- `src/particles`: particle data, cell lists, integration kernels.
- `src/forces`: injection, drag, fluid reaction, contact, wall forces.
- `src/space`: custom voxel-domain loading.
- `src/mesh_preprocessing.py`: optional Gmsh/trimesh voxelization pipeline.

## Requirements

### Required To Build And Run

- CMake 3.24 or newer.
- A C++17 compiler supported by CUDA.
- NVIDIA CUDA Toolkit with `nvcc`.
- NVIDIA GPU compatible with the configured CUDA architectures.
- Optional for the wrapper workflow: GNU Make or another compatible `make`.

The CMake project enables CUDA and C++:

```cmake
project(vascular_lbm LANGUAGES CXX CUDA)
```

By default, `CMakeLists.txt` targets CUDA architectures:

```text
60 70 75 80 86
```

Adjust `CMAKE_CUDA_ARCHITECTURES` if your GPU needs a different architecture.

### Optional For Mesh Preprocessing

To regenerate the voxel domain from `.geo` geometry:

- Python 3.
- `gmsh`
- `trimesh`
- `numpy`
- `tqdm`
- `rtree` or another trimesh nearest-query backend if required by your local
  trimesh installation.

Example:

```powershell
python -m pip install gmsh trimesh numpy tqdm rtree
```

### Optional For Viewing Results

- ParaView is recommended for `.vti` and `.vtp` files.
- Any spreadsheet or plotting tool can inspect `output/lbm_stats.csv`.

## Building

Use a fresh out-of-source build directory. The checked-in `build/` directory may
contain stale cache files from another machine, so prefer creating a new build
folder.

### Makefile Wrapper

The repository includes a top-level `Makefile` with editable variables for the
mesh path, number of steps, output interval, warmup length, `tau`, particle
capacity, injection rates, and contact parameters.

Build:

```powershell
make build
```

Run with the hardcoded particle-enabled arguments in `Makefile`:

```powershell
make run
```

Run the same mesh as a fluid-only baseline:

```powershell
make run-fluid
```

Print the exact particle run command before executing it:

```powershell
make print-args
```

Clean generated build/output folders:

```powershell
make clean
make clean-output
```

Before running `make run`, edit these variables near the top of `Makefile`:

```make
MESH_FILE := msh/voxel_domain.bin
STEPS := 50000
OUTPUT_INTERVAL := 200
WARMUP_STEPS := 0
TAU := 0.8
MAX_PARTICLES := 10000
RBC_RATE := 0.05
PLATELET_RATE := 0.02
LEUKOCYTE_RATE := 0.02
PARTICLE_SUBSTEPS := 4
MAX_PARTICLE_FORCE := 0.02
MAX_PARTICLE_SPEED := 0.05
```

### Windows PowerShell

```powershell
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release
```

If CMake cannot find CUDA, verify:

```powershell
nvcc --version
```

If `nvcc` is not found, install the CUDA Toolkit or add it to `PATH`.

If your GPU architecture is not in the default list:

```powershell
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --config Release
```

### Linux Or WSL With CUDA

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

### Expected Executable

The target is named:

```text
vascular_lbm
```

Depending on platform and generator, it will usually be located at:

```text
build-cuda/vascular_lbm
build-cuda/Release/vascular_lbm.exe
```

## Running

General syntax:

```text
vascular_lbm [mesh_file] [steps] [output_interval] [tau] [options]
```

Positional arguments:

| Argument | Default | Meaning |
|---|---:|---|
| `mesh_file` | `msh/voxel_domain.bin` | Voxel domain to load |
| `steps` | `200` | Number of recorded simulation steps after any warmup |
| `output_interval` | `20` | Fluid/stat output interval |
| `tau` | `0.8` | BGK relaxation time, must be `> 0.5` |

### Runtime Options

| Option | Default | Meaning |
|---|---:|---|
| `--warmup-steps` | `0` | Steps to run before output begins. No VTI, VTP, or CSV rows are written during warmup, and output numbering starts at `0` after warmup. |

### Fluid-Only Run

Particles are disabled by default.

Windows:

```powershell
.\build-cuda\vascular_lbm.exe msh\voxel_domain.bin 200 20 0.8
```

Linux:

```bash
./build-cuda/vascular_lbm msh/voxel_domain.bin 200 20 0.8
```

Run a 1000-step warmup before recording 5000 steps:

```bash
./build-cuda/vascular_lbm msh/voxel_domain.bin 5000 100 0.8 --warmup-steps 1000
```

### Particle-Coupled Run

Enable particles with `--max-particles`.

```powershell
.\build-cuda\vascular_lbm.exe msh\voxel_domain.bin 300 20 0.8 `
  --max-particles 5000 `
  --rbc-rate 2 `
  --platelet-rate 0.2 `
  --leukocyte-rate 0.02
```

The same command in one line:

```powershell
.\build-cuda\vascular_lbm.exe msh\voxel_domain.bin 300 20 0.8 --max-particles 5000 --rbc-rate 2 --platelet-rate 0.2 --leukocyte-rate 0.02
```

### Particle Options

| Option | Default | Meaning |
|---|---:|---|
| `--max-particles` | `0` | Fixed particle capacity. Particles disabled at `0`. |
| `--rbc-rate` | `1.0` | RBC injection attempts per step. |
| `--platelet-rate` | `0.0` | Platelet injection attempts per step. |
| `--leukocyte-rate` | `0.0` | Leukocyte injection attempts per step. |
| `--particle-output-interval` | fluid interval | Particle VTP output interval. |
| `--particle-substeps` | `4` | Particle integration substeps per LBM step. Fluid reaction is averaged over these substeps. |
| `--contact-stiffness` | `0.02` | Sphere-contact spring stiffness. |
| `--contact-damping` | `0.04` | Contact damping. |
| `--friction` | `0.2` | Tangential friction cap coefficient. |
| `--wall-stiffness` | `0.03` | Minimal wall repulsion stiffness. |
| `--wall-damping` | `0.04` | Minimal wall repulsion damping. |
| `--max-particle-force` | `0.02` | Per-particle force cap before integration. Set `0` to disable. |
| `--max-particle-speed` | `0.05` | Per-particle speed cap after velocity updates. Set `0` to disable. |

Long options can be written either as:

```text
--max-particles 5000
```

or:

```text
--max-particles=5000
```

## Regenerating The Voxel Domain

The bundled voxel file is already present. Regeneration is optional.

The preprocessing script currently targets:

```text
msh/vena_cilindrica.geo
```

and writes:

```text
msh/voxel_domain.bin
```

Run:

```powershell
python src\mesh_preprocessing.py
```

The script:

1. Opens the Gmsh geometry.
2. Generates a 3D mesh.
3. Extracts inlet, outlet, and wall surfaces.
4. Builds a trimesh surface.
5. Voxelizes the domain.
6. Marks fluid, solid, inlet, and outlet cells.
7. Writes the custom binary voxel file.

## Viewing Results In ParaView

### Fluid

1. Open ParaView.
2. Open the generated fluid files:

   ```text
   output/fluid_*.vti
   ```

3. ParaView should detect them as a time series.
4. Click `Apply`.
5. Useful views:
   - Color by `speed` to inspect flow magnitude.
   - Color by `density` to inspect stability/compressibility artifacts.
   - Color by `cell_type` to inspect domain masks.
   - Use `Glyph` or `Stream Tracer` on the `velocity` vector.

### Particles

Particles are written only when `--max-particles > 0`.

1. Open:

   ```text
   output/particles_*.vtp
   ```

2. Click `Apply`.
3. Useful views:
   - Color by `species`.
   - Color by `contact_count`.
   - Use `Glyph` with `radius` as scale information if desired.
   - Display vectors for `velocity`, `force`, or `angular_velocity`.

Species IDs:

| ID | Species |
|---:|---|
| `0` | RBC |
| `1` | Platelet |
| `2` | Leukocyte |

### Statistics CSV

Open:

```text
output/lbm_stats.csv
```

Important columns:

- `finite_cells`
- `nonfinite_cells`
- `avg_density`
- `min_density`
- `max_density`
- `avg_speed`
- `max_speed`
- `mass_flux_in`
- `mass_flux_out`
- `active_particles`
- `finite_particles`
- `nonfinite_particles`
- `injected_total`
- `dropped_total`
- `exited_total`
- `avg_particle_speed`
- `total_reaction_x/y/z`

The reaction force columns are useful for checking whether particles are feeding
momentum back into the fluid.

## Suggested Smoke Tests

### Fluid Only

Run a short simulation:

```powershell
.\build-cuda\vascular_lbm.exe msh\voxel_domain.bin 20 10 0.8
```

Expected:

- `output/lbm_stats.csv` exists.
- `output/fluid_000000.vti`, `output/fluid_000010.vti`, and
  `output/fluid_000020.vti` exist.
- Console prints finite `avg_uz` and `max_speed`.

### Low-Rate Particle Coupling

Run:

```powershell
.\build-cuda\vascular_lbm.exe msh\voxel_domain.bin 50 10 0.8 --max-particles 200 --rbc-rate 1 --platelet-rate 0.1 --leukocyte-rate 0.02
```

Expected:

- Fluid output still appears.
- `output/particles_*.vtp` appears.
- `active_particles` in `lbm_stats.csv` increases.
- `injected_total` increases.
- No NaN or infinite values appear in printed statistics.

### Capacity Limit

Run with intentionally low capacity:

```powershell
.\build-cuda\vascular_lbm.exe msh\voxel_domain.bin 50 10 0.8 --max-particles 5 --rbc-rate 3
```

Expected:

- `dropped_total` increases after capacity is exhausted.
- The program continues running.

## Troubleshooting

### `CMAKE_CUDA_COMPILER-NOTFOUND`

CMake did not find the CUDA compiler.

Check:

```powershell
nvcc --version
```

If that fails, install the NVIDIA CUDA Toolkit and ensure the CUDA `bin`
directory is in `PATH`.

### `ninja: error: loading 'build.ninja'`

The build directory is stale or was not configured with Ninja.

Create a fresh build directory:

```powershell
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release
```

### `tau must be greater than 0.5`

The BGK LBM viscosity is tied to:

```text
nu = (tau - 0.5) / 3
```

`tau <= 0.5` is invalid for this solver.

### Particles Do Not Appear

Check:

- Did you pass `--max-particles` greater than zero?
- Is `--rbc-rate`, `--platelet-rate`, or `--leukocyte-rate` greater than zero?
- Does the voxel domain report nonzero inlet cells?
- Are you opening `output/particles_*.vtp` in ParaView?

### Particle Injection Stops

The particle buffer has fixed capacity. Once full, injection attempts are
dropped and counted in `dropped_total`.

Increase:

```text
--max-particles
```

or reduce injection rates.

### Simulation Becomes Unstable

Try:

- Increase `tau`, for example from `0.8` to `1.0`.
- Reduce injection rates.
- Reduce `--contact-stiffness`.
- Increase output frequency to inspect early divergence.
- Start with fluid-only mode and then enable particles.

## Current Limitations

- The LBM inlet velocity kernel is still z-axis specific.
- The outlet is a simple zero-gradient extrapolation, not a fully developed
  pressure or traction boundary condition.
- RBCs use ellipsoid orientation for state/output, but drag and contact use an
  effective sphere radius.
- Wall interaction is a minimal repulsion, not a full wall-contact model.
- Particle parameters are hard-coded defaults in the force model rather than a
  separate material-parameter file.
- There is no automated test target yet.

## References And Project Notes

The modeling intent and source links are summarized in:

```text
bibliography.txt
```

The implementation follows the project direction there:

- Mesoscopic rather than microscopic simulation.
- D3Q19 LBM fluid.
- Rigid particle approximation for blood cells.
- Stokes-style drag coupling.
- Particle contact and force reaction back into the fluid.
