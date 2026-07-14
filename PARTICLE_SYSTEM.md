# Particle System in the Simplified Blood-Vessel Simulation

## 1. Scope and model level

The simulation combines a D3Q19 lattice Boltzmann (LBM) fluid with discrete,
rigid particles representing red blood cells (RBCs), platelets, and leukocytes.
The coupling is **unresolved**: a particle is smaller than or comparable to a
lattice cell, the LBM does not resolve its surface, and the particle does not
exclude fluid volume. Its mechanical interaction with the fluid is reduced to
a pointwise drag force and an equal-and-opposite body force on the lattice.

All particle quantities are expressed in lattice units. One LBM step has
`dt = 1`; particle motion is subdivided into four substeps by default.

## 2. Force model

The total translational force accumulated on particle `i` is

```text
F_i = F_drag,i + sum_j F_contact,ij + F_wall,i
```

There is no gravity, buoyancy, Brownian force, lift, lubrication force,
deformation model, or adhesion model.

### 2.1 Fluid drag

The local fluid density and velocity are first interpolated to the particle
center. The particle then receives a Stokes-type drag force:

```text
mu     = rho_f * (tau - 0.5) / 3
F_drag = 6 * pi * mu * r * (u_f - v_p)
```

Here `tau` is the BGK relaxation time, `r` is the particle's effective spherical
radius, `u_f` is the interpolated fluid velocity, and `v_p` is the particle
velocity. Thus the drag is linear in slip velocity and assumes a low-Reynolds-
number spherical response even for an RBC.

If `max_particle_force > 0`, the drag magnitude is capped before it is applied.
The complete accumulated force (drag, contacts, and wall force together) is
capped again immediately before integration. The executable default is `0.02`;
setting the value to zero disables both caps.

### 2.2 Particle-particle contact

Every particle is treated as a sphere for contact. Two particles interact when

```text
delta = r_i + r_j - |x_i - x_j| > 0.
```

With the unit normal `n` pointing from particle `j` to particle `i`, relative
translational velocity `v_rel = v_i - v_j`, and
`v_n = dot(v_rel, n)`, the normal spring-dashpot force is

```text
F_n = max(0, k_c * delta - c_c * v_n).
```

The tangential relative velocity and trial damping force are

```text
v_t       = v_rel - v_n * n
F_t,trial = -c_c * v_t.
```

The tangential magnitude is limited by a Coulomb-style cap,
`|F_t| <= friction * F_n`. Equal and opposite forces are atomically accumulated
on the pair. The tangential force also produces torque about each spherical
contact point, `T = r_contact x F_t`.

Important simplifications:

- rotational surface velocity is not included in `v_rel`;
- there is no stored tangential displacement or static-friction history;
- perfectly coincident centers (`distance^2 <= 1e-16`) receive no contact force;
- the RBC ellipsoid axes and orientation do not affect contact geometry.

The defaults are `k_c = 0.02`, `c_c = 0.04`, and `friction = 0.2`.

Contact candidates are found with a GPU linked-cell list. Each particle checks
its own voxel and the 26 neighboring voxels (a `3 x 3 x 3` neighborhood), and
`j > i` ensures that each pair is evaluated once.

### 2.3 Particle-wall force

Only the six face-adjacent lattice cells are inspected. If one is solid (or is
outside the grid), its shared face is treated as an axis-aligned wall plane. For
an effective sphere overlapping that plane,

```text
F_wall = max(0, k_w * delta_w - c_w * v_n) * n_wall.
```

The defaults are `k_w = 0.03` and `c_w = 0.04`. This is a simple normal penalty
force: it has no wall friction, torque, lubrication correction, curved-surface
distance calculation, or continuous collision detection. Consequently, fast
particles can in principle cross thin voxel walls; the speed and force caps are
numerical safeguards, not a geometric collision guarantee.

## 3. Integrator and update order

### 3.1 Translation

The kernels label the method "Velocity Verlet", but the implemented sequence is
not textbook velocity Verlet. For every particle substep it first evaluates the
new force at the current position, then performs

```text
v <- v + 0.5 * (F_old + F_new) / m * dt_p
x <- x + v * dt_p + 0.5 * F_new / m * dt_p^2
F_old <- F_new
```

The position update therefore uses the already-updated velocity and adds another
half-acceleration term. In standard velocity Verlet, position is advanced before
the force at the new position is evaluated. The implementation is best described
as a **Verlet-inspired force-averaging scheme with nonstandard ordering**, rather
than canonical velocity Verlet.

For newly injected particles `F_old = 0`, so their first velocity update uses
half of the first computed force. By default `dt_p = 1 / 4` because four particle
substeps are taken per LBM step. Particle `age` is incremented in every substep,
so it counts particle substeps, not LBM steps.

The speed is capped after the velocity update and checked/capped again after the
position update. Non-finite positions or velocities deactivate the particle.

### 3.2 Rotation

Angular velocity is updated by explicit Euler with a diagonal inertia tensor:

```text
omega <- omega + (T_x/I_x, T_y/I_y, T_z/I_z) * dt_p.
```

Orientation is stored as a unit quaternion. It is advanced by explicit Euler
using `q_dot = 0.5 * q (x) [0, omega]` and normalized after every update. Only
tangential particle-particle contact creates torque. There is no fluid-induced
torque, rotational drag, wall torque, or gyroscopic `omega x (I omega)` term.

### 3.3 One coupled simulation step

For an enabled particle system, one LBM time step executes:

1. Inject the requested species at inlet cells using the current fluid velocity.
2. Reset the LBM body-force arrays once.
3. For every particle substep:
   1. reset particle forces, torques, and contact counts;
   2. reset and rebuild the linked-cell list;
   3. interpolate the unchanged current LBM fields and compute drag/reaction;
   4. compute particle contacts and wall forces;
   5. cap total force and update linear velocity;
   6. cap speed and update angular velocity;
   7. update position, check/cap velocity, and update orientation;
   8. deactivate particles outside the grid or in an outlet cell;
   9. copy the current force into `F_old`.
4. Advance the LBM once: BGK collision with forcing, streaming, boundary
   conditions, and macroscopic reconstruction.

Particle kernels and the subsequent LBM kernels run in the default CUDA stream,
so their launch order supplies the required device-side sequencing.

## 4. Particle shape and properties

| Species | Shape field | Effective radius | Mass | Semi-axes `(a,b,c)` | Principal inertia |
|---|---|---:|---:|---|---|
| RBC | Ellipsoid | `0.45` | `1.0` | `(0.65, 0.45, 0.18)` | `I_x=0.2m(b^2+c^2)`, `I_y=0.2m(a^2+c^2)`, `I_z=0.2m(a^2+b^2)` |
| Platelet | Sphere | `0.25` | `0.25` | `(r,r,r)` | `I_x=I_y=I_z=0.4mr^2` |
| Leukocyte | Sphere | `0.60` | `2.0` | `(r,r,r)` | `I_x=I_y=I_z=0.4mr^2` |

Particles are rigid and non-deformable. The RBC's ellipsoid metadata, quaternion,
and ellipsoidal inertia are stored and exported, but **all translational physics**
(drag, particle contact, and wall contact) uses the effective radius `0.45`.
Consequently, RBC orientation currently affects output only, not its fluid or
collision response.

Particle state is stored as a structure of arrays (SoA) in device memory:

- position, velocity, current/previous force, mass, and effective radius;
- diagonal inertia, angular velocity, torque, and orientation quaternion;
- species, shape, and ellipsoid semi-axes;
- active flag, age, and per-substep contact count;
- injection-slot and injected/dropped/exited counters.

The particle array has a fixed, append-only capacity. Injection reserves a slot
with `atomicAdd(next_slot, 1)`; exited slots are **not reused**. Capacity therefore
limits the total number of successful slot reservations over the run, not merely
the number of simultaneously active particles.

An inlet cell is selected deterministically from a hash of step and injection
index. The center is shifted inward by `radius + 0.75` lattice units, linear
velocity is copied from that inlet lattice cell, and angular velocity is zero
with identity orientation. There is no injection-time overlap check against
walls or other particles.

## 5. Two-way interaction with the LBM fluid

### 5.1 Fluid to particle

At a particle center, density and velocity are trilinearly interpolated from the
eight surrounding lattice nodes. Out-of-domain and `SOLID` nodes are omitted;
the remaining weights are normalized to sum to one. Inlet and outlet nodes are
eligible because the test excludes only `SOLID` cells. If no valid weight or no
positive density remains, no drag is applied.

The interpolated density determines lattice viscosity and the interpolated
velocity determines slip in the Stokes drag. The fluid fields are not advanced
between particle substeps, so every substep samples the same LBM time level,
although the particle position and velocity change.

### 5.2 Particle to fluid

The equal-and-opposite drag reaction is spread back to the same eight valid
lattice nodes with the same normalized trilinear weights:

```text
F_fluid,node += -weight * F_drag.
```

With `S` particle substeps, only `1/S` of each substep's reaction is deposited.
The fluid force arrays are cleared once before the substep loop and accumulate
these contributions, so the LBM receives the substep-averaged drag reaction.
Only drag is coupled back: particle-particle and wall forces are not deposited
into the fluid.

The D3Q19 BGK collision consumes this lattice body force with a Guo forcing term.
After streaming and boundary treatment, macroscopic velocity includes the usual
half-force correction:

```text
u = (sum_q f_q c_q + 0.5 F_fluid) / rho.
```

This makes the momentum coupling two-way, but it remains an unresolved point-
particle approximation: particles neither displace fluid nor impose no-slip on
a resolved particle surface.

## 6. CUDA implementation features

### 6.1 Kernel parallelism

The code uses ordinary CUDA `__global__` kernels, generally launched with 256
threads per block. Depending on the operation there is one thread per particle,
fluid cell, or injection attempt. GPU kernels perform injection, resets, drag and
reaction spreading, contacts, wall forces, integration, linked-list construction,
exit handling, and all main LBM operations. Small indexing, interpolation, and
vector routines are `__device__` functions.

### 6.2 Memory organization

- Particle and fluid SoA fields live in CUDA global memory allocated with
  `cudaMalloc`; they are initialized with `cudaMemset` and released by
  `cudaFree`.
- Synchronous `cudaMemcpy` transfers data between host and device for setup,
  diagnostics, and VTK/CSV output.
- The mesh loader uses page-locked host memory (`cudaMallocHost`) for cell types,
  boundary IDs, and normals before copying them to the GPU.
- The small D3Q19 direction, weight, and opposite-direction tables use CUDA
  `__constant__` memory and are initialized with `cudaMemcpyToSymbol`.

### 6.3 Atomic operations

Concurrent updates are handled with CUDA atomics:

- `atomicAdd` reserves append-only injection slots and updates statistics;
- `atomicAdd` accumulates multiple particle reactions into the same fluid node;
- double-precision `atomicAdd` accumulates pair forces and torques;
- integer `atomicAdd` increments contact and exit counters;
- `atomicExch` pushes particles into voxel linked lists.

The linked-cell arrays are `cell_head[cell]`, `particle_next[particle]`, and
`particle_cell_id[particle]`. This broad phase avoids a full all-pairs scan,
although work can still become large in densely populated neighboring cells.

### 6.4 Build, synchronization, and features not used

The project uses CUDA/C++17 and separable compilation (`-rdc=true` or
`CUDA_SEPARABLE_COMPILATION ON`). CMake targets compute capabilities 60, 70, 75,
80, and 86 by default; capability 60 is also the minimum listed target supporting
native global-memory double-precision `atomicAdd`.

Kernel launches use the default stream. The main loop checks `cudaGetLastError`
and calls `cudaDeviceSynchronize` once after each complete coupled step before
host-side validation/output. The implementation does not use explicit streams,
shared-memory tiling, texture memory, unified memory, CUDA graphs, cooperative
groups, dynamic parallelism, or Thrust.

## 7. Main limitations to keep in mind

- RBCs have ellipsoid state but spherical drag and collision mechanics.
- The translational integrator is not canonical velocity Verlet despite its
  source comment, and its update ordering adds acceleration twice to position.
- Wall contact checks only six adjacent voxel faces and does not prevent
  tunneling robustly.
- The fluid coupling is point-based and does not represent excluded volume,
  resolved surface stress, deformation, or fluid torque.
- Injection storage is append-only; inactive slots cannot be recycled.
- Force/speed caps improve numerical robustness but are nonphysical and
  non-conservative.

## 8. Relevant source files

| Topic | Source |
|---|---|
| Particle state, species, and shape enums | `src/particles/particle_data.cuh` |
| Translational/rotational integration and cell-list kernels | `src/particles/particle_kernels.cuh` |
| Device allocation and kernel launch wrappers | `src/particles/particle_system.cpp` |
| Species defaults, drag, contact, wall force, and injection | `src/forces/force_model.cpp` |
| Runtime force parameters | `src/forces/force_model.hpp` |
| Coupled particle/LBM update order | `src/main.cu` |
| Guo forcing and macroscopic half-force correction | `src/lbm/lbm_kernels.cuh` |
| D3Q19 constant-memory tables | `src/lbm/lbm_constants.cuh`, `src/lbm/lbm_constants.cu` |
| CUDA build settings | `CMakeLists.txt`, `Makefile` |
