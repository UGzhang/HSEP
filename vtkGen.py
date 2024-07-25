import random
import numpy as np

# Function to generate a random point within the box
def generate_random_point():
    return [random.uniform(2, 8) for _ in range(3)]

# Function to check if a point is at least min_distance away from all points in the list
def is_valid_point(point, points, min_distance=0.2):
    for p in points:
        if np.linalg.norm(np.array(point) - np.array(p)) < min_distance:
            return False
    return True

# Generate points
points = []
while len(points) < 100:
    point = generate_random_point()
    if is_valid_point(point, points):
        points.append(point)

# Writing to VTK file
vtk_content = """# vtk DataFile Version 4.0
hesp visualization file
ASCII
DATASET UNSTRUCTURED_GRID
POINTS {0} double
""".format(len(points))

for point in points:
    vtk_content += "{0:.5f} {1:.5f} {2:.5f}\n".format(point[0], point[1], point[2])

vtk_content += """\nCELLS 0 0
CELL_TYPES 0
POINT_DATA {0}
SCALARS m double
LOOKUP_TABLE default
""".format(len(points))

for _ in points:
    vtk_content += "1.000000\n"

vtk_content += """\nVECTORS v double
"""

for _ in points:
    vtk_content += "0.000000 0.000000 0.000000\n"

# Save the content to a VTK file
with open("particles.vtk", "w") as file:
    file.write(vtk_content)

print("VTK file 'particles.vtk' has been generated.")