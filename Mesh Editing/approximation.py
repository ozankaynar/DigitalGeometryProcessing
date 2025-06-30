import open3d as o3d
import numpy as np
import copy
import heapq

print("Processing the mesh...")
num_sample = 128
lmbda = 0 # Smoothing parameter

mesh_name = "corrupted.off"
mesh = o3d.io.read_triangle_mesh(mesh_name)

vertices = np.asarray(mesh.vertices)
number_of_vertices = len(vertices)

x_coords = vertices[:, 0]
y_coords = vertices[:, 1]
z_coords = vertices[:, 2]

triangles = np.asarray(mesh.triangles)
number_of_triangles = len(triangles)

n = number_of_vertices
# Edge lengths
mat = np.zeros([n, n])
for i in range(number_of_triangles):
    mat[triangles[i,0], triangles[i,1]] = np.linalg.norm(vertices[triangles[i,0]] - vertices[triangles[i,1]])
    mat[triangles[i,0], triangles[i,2]] = np.linalg.norm(vertices[triangles[i,0]] - vertices[triangles[i,2]])
    mat[triangles[i,1], triangles[i,2]] = np.linalg.norm(vertices[triangles[i,1]] - vertices[triangles[i,2]])
    mat[triangles[i,1], triangles[i,0]] = mat[triangles[i,0], triangles[i,1]]
    mat[triangles[i,2], triangles[i,0]] = mat[triangles[i,0], triangles[i,2]]
    mat[triangles[i,2], triangles[i,1]] = mat[triangles[i,1], triangles[i,2]]

max_value = np.max(mat)
min_value = np.min(mat[mat != 0])
print("Max element of mat:", max_value)
print("Min element of mat:", min_value)

adjacency_matrix = (mat > 0).astype(int)
degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
laplacian_matrix = degree_matrix - adjacency_matrix
laplacian = laplacian_matrix / np.diag(laplacian_matrix)[:, np.newaxis]

# Compute differential coordinates for each vertex
differential_coords = np.zeros((number_of_vertices, 3))
for i in range(number_of_vertices):
    neighbors = np.where(adjacency_matrix[i] > 0)[0]
    if len(neighbors) == 0:
        continue
    neighbor_coords = vertices[neighbors]
    center_of_mass = neighbor_coords.mean(axis=0)
    differential_coords[i] = vertices[i] - center_of_mass

diff_x = differential_coords[:, 0]
diff_y = differential_coords[:, 1]
diff_z = differential_coords[:, 2]

# Function to find the shortest path between two points using Dijkstra's algorithm
def dijkstra_shortest_path(mat, start, end):
    n = mat.shape[0]
    dist = np.full(n, np.inf)
    dist[start] = 0
    prev = np.full(n, -1)
    used = set()
    min_heap = [(0, start)]

    while min_heap:
        current_dist, u = heapq.heappop(min_heap)
        if u in used:
            continue
        used.add(u)

        if u == end:
            break

        for v in range(n):
            if mat[u, v] != 0 and v not in used:
                trial = dist[u] + mat[u, v]
                if trial < dist[v]:
                    dist[v] = trial
                    prev[v] = u
                    heapq.heappush(min_heap, (trial, v))

    path = []
    u = end
    while prev[u] != -1:
        path.append((prev[u], u))
        u = prev[u]
    path.reverse()
    return path, dist

# Farthest Point Sampling using Dijkstra's algorithm
def farthest_point_sampling(mat, num_sample, first_point=None):
    n = mat.shape[0]
    selected_points = []
    if first_point is None:
        first_point = np.random.randint(n)
    selected_points.append(first_point)
    _, distances = dijkstra_shortest_path(mat, first_point, -1)
    for _ in range(1, num_sample):
        next_point = np.argmax(distances)
        selected_points.append(next_point)
        _, new_distances = dijkstra_shortest_path(mat, next_point, -1)
        distances = np.minimum(distances, new_distances)
    return selected_points

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def mesh_volume(vertices, triangles):
    v = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        v += np.dot(np.cross(v0, v1), v2)
    return abs(v) / 6.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)

picked = pick_points(pcd)
if not picked:
    print("No point picked, using the point with lowest x coordinate.")
    first_point = np.argmin(x_coords)
else:
    first_point = picked[0]
print(f"First FPS point: {first_point}")

control_points = farthest_point_sampling(mat, num_sample, first_point=first_point)
print("Control points:", control_points)

colors = np.tile([0.7, 0.7, 0.7], (number_of_vertices, 1))
for idx in control_points:
    colors[idx] = [1, 0, 0]
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd], window_name="FPS Points (red) on Mesh")

print("Differential coordinates for control points:")
for idx in control_points:
    print(f"Control point {idx}: {differential_coords[idx]}")

weights = np.full(number_of_vertices, 1.0**2)

boundary_x = lmbda*(laplacian @ diff_x)
boundary_y = lmbda*(laplacian @ diff_y)
boundary_z = lmbda*(laplacian @ diff_z)

matrix = np.zeros((number_of_vertices, number_of_vertices))

matrix = laplacian @ laplacian
x_matrix = matrix.copy()
y_matrix = matrix.copy()
z_matrix = matrix.copy()

for cp in control_points:
    x_matrix[cp, cp] += weights[cp]
    y_matrix[cp, cp] += weights[cp]
    z_matrix[cp, cp] += weights[cp]
    boundary_x[cp] += x_coords[cp]*weights[cp]
    boundary_y[cp] += y_coords[cp]*weights[cp]
    boundary_z[cp] += z_coords[cp]*weights[cp]

# Solve the linear system
solution_x = np.linalg.solve(x_matrix, boundary_x)
solution_y = np.linalg.solve(y_matrix, boundary_y)
solution_z = np.linalg.solve(z_matrix, boundary_z)

# Create a new mesh with the solved coordinates
new_vertices = np.column_stack((solution_x, solution_y, solution_z))
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()
new_mesh = o3d.geometry.TriangleMesh()
new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
new_mesh.triangles = mesh.triangles
new_mesh.compute_vertex_normals()

volume = mesh_volume(np.asarray(new_mesh.vertices), np.asarray(new_mesh.triangles))
print(f"Volume of the resultant mesh: {volume}")

original_volume = mesh_volume(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
print(f"Volume of the original mesh: {original_volume}")

# Create a new mesh with the rotated coordinates
new_mesh = o3d.geometry.TriangleMesh()
new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
new_mesh.triangles = mesh.triangles
new_mesh.compute_vertex_normals()

# # Corrupt the original mesh by adding Gaussian noise
# noise_sigma = 0.002 * np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))  # 1% of bounding box diagonal
# gaussian_noise = np.random.normal(0, noise_sigma, vertices.shape)
# noisy_vertices = vertices + gaussian_noise

# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(noisy_vertices)
# mesh.triangles = o3d.utility.Vector3iVector(triangles)
# mesh.compute_vertex_normals()

# # Save the corrupted (noisy) mesh
# corrupted_mesh_name = "corrupted_" + mesh_name
# o3d.io.write_triangle_mesh(corrupted_mesh_name, mesh)
# print(f"Corrupted mesh saved as {corrupted_mesh_name}")

# Visualize the original and new meshes
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([new_mesh], mesh_show_back_face=True)

# Save the new mesh
output_mesh_name = "output_" + mesh_name
o3d.io.write_triangle_mesh(output_mesh_name, new_mesh)
print(f"New mesh saved as {output_mesh_name}")

