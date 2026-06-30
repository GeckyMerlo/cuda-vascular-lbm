import gmsh
import trimesh
import numpy as np
from tqdm import tqdm
from time import perf_counter
from pathlib import Path

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
        origin = bounds[0]
        max_p = bounds[1]

        extent = max_p - origin
        nx, ny, nz = np.ceil(extent / dx).astype(int) + 1
        total_voxels = nx * ny * nz

        xs = origin[0] + (np.arange(nx) + 0.5) * dx
        ys = origin[1] + (np.arange(ny) + 0.5) * dx

        print(f"dx = {dx}")
        print(f"nx = {nx}, ny = {ny}, nz = {nz}")
        print(f"total voxels = {total_voxels:,}")
        toc(t)

        cell_type = np.full(total_voxels, SOLID, dtype=np.int32)

        tol = 1.5 * dx

        t = tic("Voxelizing geometry")

        for k0 in tqdm(range(0, nz, batch_z), desc="Voxel slices"):
            k1 = min(k0 + batch_z, nz)

            zs = origin[2] + (np.arange(k0, k1) + 0.5) * dx

            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            centers = np.column_stack([
                X.ravel(),
                Y.ravel(),
                Z.ravel()
            ])

            inside = surface_mesh.contains(centers)

            local_type = np.full(len(centers), SOLID, dtype=np.int32)
            local_type[inside] = FLUID

            inlet_dist = inlet_mesh.nearest.signed_distance(centers)
            outlet_dist = outlet_mesh.nearest.signed_distance(centers)

            local_type[
                (local_type == FLUID) & (np.abs(inlet_dist) < tol)
            ] = INLET

            local_type[
                (local_type == FLUID) & (np.abs(outlet_dist) < tol)
            ] = OUTLET

            for local_k, global_k in enumerate(range(k0, k1)):
                local_start = local_k * nx * ny
                local_end = local_start + nx * ny

                global_start = global_k * nx * ny
                global_end = global_start + nx * ny

                cell_type[global_start:global_end] = local_type[local_start:local_end]

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

            # Inlet ids
            np.array([len(inlet_ids)], dtype=np.int32).tofile(f)
            inlet_ids.tofile(f)

            # Outlet ids
            np.array([len(outlet_ids)], dtype=np.int32).tofile(f)
            outlet_ids.tofile(f)

        toc(t)


        print(f"\nSaved voxel domain to: {output_file}")

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
