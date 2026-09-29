import sys
import os
import numpy as np

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <mesh.vtk>")
    sys.exit(1)

infile = sys.argv[1]
base, ext = os.path.splitext(infile)
outfile = base + "_shifted" + ext

with open(infile, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip().startswith("POINTS"):
        header_idx = i
        n_points = int(line.split()[1])
        break
else:
    raise RuntimeError("POINTS section not found")

start = header_idx + 1

points = np.array([
    list(map(float, lines[start + k].split()))
    for k in range(n_points)
])

# Sposta la mesh così che il suo minimo diventi (0,0,0)
shift = -points.min(axis=0)
points += shift

for k in range(n_points):
    lines[start + k] = f"{points[k,0]} {points[k,1]} {points[k,2]}\n"

with open(outfile, "w") as f:
    f.writelines(lines)

print("Output:", outfile)
print("Applied shift:", shift)
print("New min:", points.min(axis=0))
print("New max:", points.max(axis=0))
