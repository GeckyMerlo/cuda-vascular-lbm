# Model and mathematics

This page translates the equations from the project presentation into a single
description of the implemented model. Symbols refer to lattice units unless
stated otherwise.

## 1. Fluid model: D3Q19 Lattice Boltzmann Method

Each voxel stores 19 distribution functions $f_q$, one for every D3Q19 lattice
direction $\mathbf{c}_q$. Density and velocity are reconstructed from their
moments:

$$
\rho = \sum_q f_q,
\qquad
\mathbf{u} = \frac{\sum_q f_q\mathbf{c}_q + \tfrac{1}{2}\mathbf{F}}{\rho}.
$$

The half-force correction in the velocity is part of the Guo forcing treatment.

### Equilibrium distribution

The second-order equilibrium used by the solver is

$$
f_q^{\mathrm{eq}} = w_q\rho
\left[
1 + 3\,\mathbf{c}_q\!\cdot\!\mathbf{u}
+ \frac{9}{2}(\mathbf{c}_q\!\cdot\!\mathbf{u})^2
- \frac{3}{2}\mathbf{u}\!\cdot\!\mathbf{u}
\right].
$$

Here $w_q$ is the D3Q19 weight for direction $q$.

### BGK collision and Guo forcing

During collision, each population relaxes toward equilibrium:

$$
f_q^* = f_q - \omega\left(f_q-f_q^{\mathrm{eq}}\right) + F_q,
\qquad \omega = \frac{1}{\tau}.
$$

The forcing contribution shown in the presentation is

$$
F_q = w_q\left(1-\frac{\omega}{2}\right)
\left[
3(\mathbf{c}_q-\mathbf{u})
+ 9(\mathbf{c}_q\!\cdot\!\mathbf{u})\mathbf{c}_q
\right]\!\cdot\!\mathbf{F}.
$$

The kinematic viscosity in lattice units is

$$
\nu = \frac{\tau-0.5}{3}.
$$

The code requires $\tau>0.5$.

### Pull streaming and wall bounce-back

The solver uses pull streaming, so a cell reads the post-collision population
from its neighbor in the opposite lattice direction:

$$
f_q(\mathbf{x},t+1) = f_q^*(\mathbf{x}-\mathbf{c}_q,t).
$$

When that source lies in a solid voxel, the population is reflected using its
opposite direction $\bar q$:

$$
f_q(\mathbf{x},t+1) = f_{\bar q}^*(\mathbf{x},t).
$$

This bounce-back rule approximates a no-slip vessel wall on the voxel grid.

## 2. Inlet and outlet reconstruction

For the z-aligned inlet, the imposed normal velocity and known populations give
the density

$$
\rho_{\mathrm{in}} = \frac{S_0+2S_-}{1-u_{z,\mathrm{in}}}.
$$

$S_0$ contains populations with zero normal component and $S_-$ contains the
known populations directed out of the domain. A missing population is
reconstructed by copying the opposite non-equilibrium part:

$$
f_q = f_q^{\mathrm{eq}} +
\left(f_{\bar q}-f_{\bar q}^{\mathrm{eq}}\right).
$$

For the density-imposed outlet used in the presentation,

$$
\rho_{\mathrm{out}}=1,
\qquad
u_{z,\mathrm{out}} = \frac{S_0+2S_+}{\rho_{\mathrm{out}}}-1.
$$

The missing outlet populations are reconstructed with the imposed density and
the computed outlet velocity:

$$
f_q = f_q^{\mathrm{eq}}(\rho_{\mathrm{out}},\mathbf{u}_{\mathrm{out}})
+ f_{\bar q}
- f_{\bar q}^{\mathrm{eq}}(\rho_{\mathrm{out}},\mathbf{u}_{\mathrm{out}}).
$$

The executable also contains copy and soft-convective outlet variants so their
behavior can be compared experimentally.

## 3. Particle representation

The model contains three rigid species:

| Species | Stored shape | Effective radius | Mass | Semi-axes $(a,b,c)$ |
| --- | --- | ---: | ---: | --- |
| Red blood cell | Ellipsoid | $0.45$ | $1.0$ | $(0.65, 0.45, 0.18)$ |
| Platelet | Sphere | $0.25$ | $0.25$ | $(r,r,r)$ |
| Leukocyte | Sphere | $0.60$ | $2.0$ | $(r,r,r)$ |

The RBC quaternion and ellipsoidal axes are used for state and visualization.
The current drag and contact calculations still treat every species as a sphere
with its effective radius.

The total translational force on particle $i$ is

$$
\mathbf{F}_i = \mathbf{F}_{\mathrm{drag},i}
+ \sum_j \mathbf{F}_{\mathrm{contact},ij}
+ \mathbf{F}_{\mathrm{wall},i}.
$$

The implementation excludes gravity, buoyancy, Brownian forcing, lift,
lubrication, adhesion, and membrane deformation.

## 4. Fluid-particle coupling

The eight lattice nodes around a particle center provide trilinearly
interpolated fluid density $\rho_f$ and velocity $\mathbf{u}_f$. Solid nodes are
discarded and the remaining weights are normalized.

The dynamic viscosity used by the particle model is

$$
\mu = \frac{\rho_f(\tau-0.5)}{3}.
$$

The particle receives a Stokes-type drag force

$$
\mathbf{F}_{\mathrm{drag}} =
6\pi\mu r\left(\mathbf{u}_f-\mathbf{v}_p\right).
$$

The same interpolation weights distribute the equal-and-opposite reaction back
to the fluid:

$$
\mathbf{F}_{\mathrm{fluid},k}mathrel{+}=
-w_k\mathbf{F}_{\mathrm{drag}}.
$$

With $N$ particle substeps, each substep deposits $1/N$ of its reaction. The
fluid force arrays are cleared once per LBM step, so the collision operator sees
the accumulated substep-average reaction. Contact and wall forces are not
deposited into the fluid.

## 5. Particle-particle contact

The GPU linked-cell list restricts each search to the particle's voxel and the
26 adjacent voxels. Particles $i$ and $j$ overlap when

$$
\delta = r_i+r_j-\lVert\mathbf{x}_i-\mathbf{x}_j\rVert > 0.
$$

Let $\mathbf{n}$ point from particle $j$ to particle $i$, let
$\mathbf{v}_{\mathrm{rel}}=\mathbf{v}_i-\mathbf{v}_j$, and define

$$
v_n=\mathbf{v}_{\mathrm{rel}}\cdot\mathbf{n}.
$$

The normal spring-dashpot magnitude is

$$
F_n=\max\left(0,k_c\delta-c_c v_n\right).
$$

The tangential relative velocity and trial force are

$$
\mathbf{v}_t=\mathbf{v}_{\mathrm{rel}}-v_n\mathbf{n},
\qquad
\mathbf{F}_{t}^{*}=-c_c\mathbf{v}_t.
$$

Its magnitude is capped by a Coulomb-style bound:

$$
\lVert\mathbf{F}_t\rVert \le \mu_f F_n.
$$

Equal and opposite forces are applied to the pair. The tangential force also
creates torque about the spherical contact point. The model has no persistent
tangential displacement, so it does not reproduce static-friction history.

## 6. Particle-wall contact

Only the six face-adjacent lattice cells are checked. If the effective sphere
overlaps a solid face,

$$
\delta_w=r-d_w,
$$

and the penalty force is

$$
\mathbf{F}_{\mathrm{wall}}=
\max\left(0,k_w\delta_w-c_wv_n\right)\mathbf{n}_{\mathrm{wall}}.
$$

This force keeps particles inside the voxelized vessel but does not calculate a
continuous distance to a curved wall.

## 7. Translation and rotation

For every particle substep $\Delta t_p$, the code applies

$$
\mathbf{v}\leftarrow\mathbf{v}
+\frac{1}{2}\frac{\mathbf{F}_{\mathrm{old}}+\mathbf{F}_{\mathrm{new}}}{m}\Delta t_p,
$$

$$
\mathbf{x}\leftarrow\mathbf{x}+\mathbf{v}\Delta t_p
+\frac{1}{2}\frac{\mathbf{F}_{\mathrm{new}}}{m}\Delta t_p^2,
$$

$$
\mathbf{F}_{\mathrm{old}}\leftarrow\mathbf{F}_{\mathrm{new}}.
$$

The kernels call this Velocity Verlet, but the ordering differs from canonical
Velocity Verlet: the velocity is updated before the position and the new force
is evaluated at the current position. It is more accurately described as a
Verlet-inspired force-averaging update.

Angular velocity uses explicit Euler with diagonal inertia:

$$
\boldsymbol{\omega}\leftarrow\boldsymbol{\omega}
+\mathbf{I}^{-1}\boldsymbol{\tau}\,\Delta t_p.
$$

The orientation quaternion follows

$$
\dot{\mathbf{q}}=\frac{1}{2}\mathbf{q}\otimes(0,\boldsymbol{\omega}),
$$

$$
\mathbf{q}\leftarrow
\operatorname{normalize}\left(\mathbf{q}+\dot{\mathbf{q}}\Delta t_p\right).
$$

Only tangential particle contact currently produces torque. There is no fluid
rotational drag or wall torque.

## 8. Coupled update order

One LBM time step performs the following operations:

1. Inject the requested species through inlet cells using local fluid velocity.
2. Clear the lattice body-force arrays.
3. For every particle substep, rebuild the cell list, interpolate fluid fields,
   compute drag and reaction, evaluate contacts and wall forces, then update
   translation and rotation.
4. Deactivate particles that leave the grid or enter an outlet cell.
5. Advance the LBM through collision, streaming, boundary reconstruction, and
   macroscopic-field reconstruction.

CUDA kernels run in the default stream, so launch order provides the required
device-side sequencing.

## Sources within this repository

- [Project presentation](HESP-Project-Presentation.pptx)
- [Particle implementation notes](../PARTICLE_SYSTEM.md)
- [`src/lbm/lbm_kernels.cuh`](../src/lbm/lbm_kernels.cuh)
- [`src/particles/particle_kernels.cuh`](../src/particles/particle_kernels.cuh)
- [`src/forces/force_model.cpp`](../src/forces/force_model.cpp)
