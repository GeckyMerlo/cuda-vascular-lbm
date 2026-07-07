import gmsh
import trimesh
import numpy as np
from tqdm import tqdm
from time import perf_counter
from pathlib import Path
from collections import deque

# Cell types
FLUID  = 0
SOLID  = 1
INLET  = 2
OUTLET = 3


def tic(msg):
    print(f"\n[START] {msg}")
    return perf_counter()


def toc(t):
    print(f"[DONE] {perf_counter() - t:.2f} s")


def get_nodes():
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    vertices = node_coords.reshape(-1, 3)
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    return vertices, tag_to_idx


def get_faces_from_entities(entity_tags, tag_to_idx):
    faces = []

    for entity_tag in entity_tags:
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(
            dim=2,
            tag=entity_tag
        )

        for etype, node_tags in zip(elem_types, elem_node_tags):
            if etype != 2:  # 2 = linear triangle
                continue

            triangles = np.array(node_tags, dtype=np.int64).reshape(-1, 3)

            for tri in triangles:
                faces.append([
                    tag_to_idx[int(tri[0])],
                    tag_to_idx[int(tri[1])],
                    tag_to_idx[int(tri[2])]
                ])

    return np.asarray(faces, dtype=np.int64)


def get_physical_surface_faces(tag_to_idx):
    groups = {}

    for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=2):
        name = gmsh.model.getPhysicalName(dim, phys_tag).lower()
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)

        faces = get_faces_from_entities(entities, tag_to_idx)
        groups[name] = faces

        print(f"{name}: {len(faces)} triangles")

    return groups

def flood_fill_boundary_limited(cell_type, seed_ids, allowed_mask, new_type, nx, ny, nz):
    visited = np.zeros_like(cell_type, dtype=bool)
    q = deque()

    for sid in seed_ids:
        if allowed_mask[sid]:
            visited[sid] = True
            q.append(sid)

    dirs = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1)
    ]

    while q:
        id = q.popleft()
        cell_type[id] = new_type

        z = id // (nx * ny)
        y = (id % (nx * ny)) // nx
        x = id % nx

        for dz, dy, dx_ in dirs:
            zz = z + dz
            yy = y + dy
            xx = x + dx_

            if not (0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx):
                continue

            nid = zz * (nx * ny) + yy * nx + xx

            if visited[nid]:
                continue

            if allowed_mask[nid]:
                visited[nid] = True
                q.append(nid)


def main():
    project_root = Path(__file__).resolve().parents[1]
    geo_file = project_root / "msh" / "vena_cilindrica.geo"
    output_file = project_root / "msh" / "voxel_domain.bin"

    dx = 0.25          # start with 0.5, then try 0.25, then 0.1
    batch_z = 2      # number of z-slices processed at once

    gmsh.initialize()

    try:
        t = tic("Reading Gmsh geometry")
        gmsh.open(str(geo_file))
        gmsh.model.occ.synchronize()
        toc(t)

        t = tic("Generating 3D mesh")
        gmsh.model.mesh.generate(3)
        toc(t)

        vtk_file = project_root / "msh" / "vena_cilindrica.vtk"
        gmsh.write(str(vtk_file))
        print(f"Mesh saved to {vtk_file}")

        t = tic("Extracting nodes and physical surfaces")
        vertices, tag_to_idx = get_nodes()
        groups = get_physical_surface_faces(tag_to_idx)
        toc(t)

        wall_faces = groups["wall"]
        inlet_faces = groups["inlet"]
        outlet_faces = groups["outlet"]

        all_faces = np.vstack([
            wall_faces,
            inlet_faces,
            outlet_faces
        ])

        t = tic("Building trimesh surface")
        surface_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=all_faces,
            process=True
        )

        inlet_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=inlet_faces,
            process=False
        )

        outlet_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=outlet_faces,
            process=False
        )

        print("Watertight:", surface_mesh.is_watertight)
        print("Bounds:", surface_mesh.bounds)
        toc(t)

        t = tic("Creating voxel grid metadata")
        bounds = surface_mesh.bounds
        padding_vec = np.array([2*dx, 2*dx, 0.0])
        origin = bounds[0] - padding_vec
        max_p  = bounds[1] + padding_vec

        extent = max_p - origin
        nx, ny, nz = np.ceil(extent / dx).astype(int)  # + 1 <- commentato per il momento 
        total_voxels = nx * ny * nz

        xs = origin[0] + (np.arange(nx) + 0.5) * dx
        ys = origin[1] + (np.arange(ny) + 0.5) * dx

        print(f"dx = {dx}")
        print(f"nx = {nx}, ny = {ny}, nz = {nz}")
        print(f"total voxels = {total_voxels:,}")
        toc(t)

        cell_type = np.full(total_voxels, SOLID, dtype=np.int32)
        normal = np.zeros((total_voxels, 3), dtype=np.float64)

        t = tic("Voxelizing geometry")

        tol = 1.0 * dx

        mean_inlet_normal = inlet_mesh.face_normals.mean(axis=0)
        mean_inlet_normal /= np.linalg.norm(mean_inlet_normal)

        mean_outlet_normal = outlet_mesh.face_normals.mean(axis=0)
        mean_outlet_normal /= np.linalg.norm(mean_outlet_normal)

        normal_threshold = 0.7

        for k0 in tqdm(range(0, nz, batch_z), desc="Voxel slices"):
            k1 = min(k0 + batch_z, nz)

            zs = origin[2] + (np.arange(k0, k1) + 0.5) * dx

            Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
            centers = np.column_stack([
                X.ravel(),
                Y.ravel(),
                Z.ravel()
            ])

            inside = surface_mesh.contains(centers)

            local_type = np.full(len(centers), SOLID, dtype=np.int32)
            local_type[inside] = FLUID

            _, inlet_dist, inlet_tri_id = inlet_mesh.nearest.on_surface(centers)
            _, outlet_dist, outlet_tri_id = outlet_mesh.nearest.on_surface(centers)

            inlet_normals = inlet_mesh.face_normals[inlet_tri_id]
            outlet_normals = outlet_mesh.face_normals[outlet_tri_id]

            inlet_dot = inlet_normals @ mean_inlet_normal
            outlet_dot = outlet_normals @ mean_outlet_normal

            is_inlet = (
                (local_type == FLUID) &
                (inlet_dist < tol) &
                (inlet_dot > normal_threshold)
            )

            is_outlet = (
                (local_type == FLUID) &
                (outlet_dist < tol) &
                (outlet_dot > normal_threshold)
            )

            both = is_inlet & is_outlet
            is_inlet[both] = False
            is_outlet[both] = False

            local_type[is_inlet] = INLET
            local_type[is_outlet] = OUTLET

            local_normal = np.zeros((len(centers), 3))
            local_normal[is_inlet] = inlet_normals[is_inlet]
            local_normal[is_outlet] = outlet_normals[is_outlet]

            for local_k, global_k in enumerate(range(k0, k1)):
                local_start = local_k * nx * ny
                local_end = local_start + nx * ny

                global_start = global_k * nx * ny
                global_end = global_start + nx * ny

                cell_type[global_start:global_end] = local_type[local_start:local_end]
                normal[global_start:global_end] = local_normal[local_start:local_end]

        toc(t)

        
        t = tic("Flood-fill of the boundaries")

        cell_type_3d = cell_type.reshape((nz, ny, nx))

        inlet_allowed = np.zeros_like(cell_type, dtype=bool)
        outlet_allowed = np.zeros_like(cell_type, dtype=bool)

        inlet_slices = np.where(np.any(cell_type_3d == INLET, axis=(1,2)))[0]
        outlet_slices = np.where(np.any(cell_type_3d == OUTLET, axis=(1,2)))[0]

        for z in inlet_slices:
            ids = np.arange(z * nx * ny, (z + 1) * nx * ny)
            inlet_allowed[ids] = (cell_type[ids] == FLUID) | (cell_type[ids] == INLET)

        for z in outlet_slices:
            ids = np.arange(z * nx * ny, (z + 1) * nx * ny)
            outlet_allowed[ids] = (cell_type[ids] == FLUID) | (cell_type[ids] == OUTLET)

        flood_fill_boundary_limited(
            cell_type,
            np.where(cell_type == INLET)[0],
            inlet_allowed,
            INLET,
            nx, ny, nz
        )

        flood_fill_boundary_limited(
            cell_type,
            np.where(cell_type == OUTLET)[0],
            outlet_allowed,
            OUTLET,
            nx, ny, nz
        )

        normal[cell_type == INLET] = mean_inlet_normal
        normal[cell_type == OUTLET] = mean_outlet_normal
        
        toc(t)
        

        print("\nCell statistics:")
        print("SOLID :", np.sum(cell_type == SOLID))
        print("FLUID :", np.sum(cell_type == FLUID))
        print("INLET :", np.sum(cell_type == INLET))
        print("OUTLET:", np.sum(cell_type == OUTLET))

        t = tic("Extracting inlet/outlet ids")

        inlet_ids = np.where(cell_type == INLET)[0].astype(np.int32)
        outlet_ids = np.where(cell_type == OUTLET)[0].astype(np.int32)

        print(f"num_inlets  = {len(inlet_ids)}")
        print(f"num_outlets = {len(outlet_ids)}")

        print("mean inlet normal :", normal[inlet_ids].mean(axis=0))
        print("mean outlet normal:", normal[outlet_ids].mean(axis=0))


        toc(t)

        t = tic(f"Writing {output_file}")

        with open(output_file, "wb") as f:
            # Grid size
            np.array([nx, ny, nz], dtype=np.int32).tofile(f)

            # Grid spacing
            np.array([dx, dx, dx], dtype=np.float64).tofile(f)

            # Origin
            origin.astype(np.float64).tofile(f)

            # Cell types
            cell_type.astype(np.int32).tofile(f)

            #Normals
            normal.astype(np.float64).tofile(f)

            # Inlet ids
            np.array([len(inlet_ids)], dtype=np.int32).tofile(f)
            inlet_ids.tofile(f)

            # Outlet ids
            np.array([len(outlet_ids)], dtype=np.int32).tofile(f)
            outlet_ids.tofile(f)

        toc(t)


        print(f"\nSaved voxel domain to: {output_file}")

        print("\n=== Cross sections along z ===")

        cell_type_3d = cell_type.reshape((nz, ny, nx))

        for z in range(nz):
            n_fluid  = np.sum(cell_type_3d[z] == FLUID)
            n_solid  = np.sum(cell_type_3d[z] == SOLID)
            n_inlet  = np.sum(cell_type_3d[z] == INLET)
            n_outlet = np.sum(cell_type_3d[z] == OUTLET)

            if n_fluid or n_inlet or n_outlet:
                print(
                    f"z={z:3d}  "
                    f"FLUID={n_fluid:4d}  "
                    f"INLET={n_inlet:4d}  "
                    f"OUTLET={n_outlet:4d}  "
                    f"SOLID={n_solid:4d}"
                )

        print("\n=== Connectivity check ===")

        cell_type_3d = cell_type.reshape((nz, ny, nx))

        for z in range(nz):
            section = cell_type_3d[z]

            has_fluid = np.any(section == FLUID)
            has_inlet = np.any(section == INLET)
            has_outlet = np.any(section == OUTLET)

            if has_fluid or has_inlet or has_outlet:
                print(
                    f"slice {z:3d}: "
                    f"fluid={has_fluid} "
                    f"inlet={has_inlet} "
                    f"outlet={has_outlet}"
                )

        print("\n=== Outlet adjacency ===")

        cell_type_3d = cell_type.reshape((nz, ny, nx))

        bad = 0
        good = 0

        for oid in outlet_ids:
            z = oid // (nx * ny)
            y = (oid % (nx * ny)) // nx
            x = oid % nx

            neigh = []
            for dz, dy, dx_ in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                zz, yy, xx = z+dz, y+dy, x+dx_
                if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                    neigh.append(cell_type_3d[zz, yy, xx])

            n_fluid = sum(t == FLUID for t in neigh)

            if n_fluid == 0:
                bad += 1
            else:
                good += 1

        print("outlet with fluid neighbor:", good)
        print("outlet isolated:", bad)

        print("\n=== Last slices detail ===")
        for z in range(nz-5, nz):
            section = cell_type_3d[z]
            print(
                z,
                "FLUID", np.sum(section == FLUID),
                "OUTLET", np.sum(section == OUTLET),
                "SOLID", np.sum(section == SOLID)
            )

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
