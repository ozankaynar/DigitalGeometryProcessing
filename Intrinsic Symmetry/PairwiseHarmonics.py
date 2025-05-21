import open3d as o3d
import numpy as np
import itertools
import matplotlib.pyplot as plt  # Import Matplotlib for colormap

print("Processing the mesh...")
num_sample = 4

mesh_name = "46.off"
mesh = o3d.io.read_triangle_mesh(mesh_name)

vertices = np.asarray(mesh.vertices)
number_of_vertices = len(vertices)

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

# Find the minimum element of mat except 0
max_value = np.max(mat)
min_value = np.min(mat[mat != 0])
print("Max element of mat:", max_value)
print("Min element of mat:", min_value)

adjacency_matrix = (mat > 0).astype(int)
degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

# Function to find the shortest path between two points using Dijkstra's algorithm
def dijkstra_shortest_path(mat, start, end):
    n = mat.shape[0]
    dist = np.full(n, np.inf)
    dist[start] = 0
    prev = np.full(n, -1)
    used = set()
    vertices = list(range(n))

    while vertices:
        u = min(vertices, key=lambda vertex: dist[vertex])
        vertices.remove(u)
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

    path = []
    vertices_in_path = [start]
    u = end
    while prev[u] != -1:
        path.append((prev[u], u))
        vertices_in_path.append(u)
        u = prev[u]
    path.reverse()
    return path, vertices_in_path, dist

# Farthest Point Sampling using Dijkstra's algorithm
def farthest_point_sampling(mat, num_sample):
    n = mat.shape[0]
    selected_points = []
    _, _, rand_distance = dijkstra_shortest_path(mat, np.random.randint(n), -1)
    next_point = np.argmax(rand_distance)
    selected_points.append(next_point)
    _, _, distances = dijkstra_shortest_path(mat, selected_points[0], -1)

    for _ in range(1, num_sample):
        _, _, new_distances = dijkstra_shortest_path(mat, next_point, -1)
        distances = np.minimum(distances, new_distances)
        next_point = np.argmax(distances)
        selected_points.append(next_point)

    return selected_points

def visualization(symmetry_points):
    # Create a mesh from the vertices and triangles
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()

    # Create a point cloud for the FPS points
    fps_pcd = o3d.geometry.PointCloud()
    fps_pcd.points = o3d.utility.Vector3dVector(vertices[symmetry_points])
    fps_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(symmetry_points))])

    # Visualize the FPS points on the mesh
    mesh.compute_vertex_normals()
    mesh_material = o3d.visualization.rendering.MaterialRecord()
    mesh_material.shader = "defaultLitTransparency"
    mesh_material.base_color = [0.8, 0.8, 0.8, 0.5]

    fps_pcd = o3d.geometry.PointCloud()
    fps_pcd.points = o3d.utility.Vector3dVector(vertices[symmetry_points])
    fps_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(symmetry_points))])

    fps_material = o3d.visualization.rendering.MaterialRecord()
    fps_material.point_size = 10

    o3d.visualization.draw([{
            "name": "mesh",
            "geometry": mesh,
            "material": mesh_material
        },{
            "name": "fps_pcd",
            "geometry": fps_pcd,
            "material": fps_material
        }])
    return mesh, fps_pcd

# Function to compute geodesic rings and their lengths for a single FPS point
def compute_geodesic_rings_for_point(mat, fps_point, distance_step, tol):
    n = mat.shape[0]
    ring_lengths_list = {}
    ring_vertices_list = {}

    # Compute geodesic distances from the FPS point
    _, _, distances = dijkstra_shortest_path(mat, fps_point, -1)

    # Group vertices into rings based on distance thresholds
    max_distance = np.max(distances[distances < np.inf])
    num_rings = int(np.floor(max_distance / distance_step))

    for ring_idx in range(num_rings):
        lower_bound = (1 + ring_idx) * distance_step - tol
        upper_bound = (1 + ring_idx) * distance_step + tol

        # Find vertices in the current ring
        ring_vertices = np.where((distances >= lower_bound) & (distances <= upper_bound))[0]

        # Compute the total edge length within the ring
        ring_length = 0
        for v in ring_vertices:
            for u in range(n):
                if mat[v, u] != 0 and distances[u] >= lower_bound and distances[u] < upper_bound:
                    ring_length += mat[v, u]

        # Store the ring length
        ring_lengths_list[ring_idx] = ring_length / 2
        ring_vertices_list[ring_idx] = ring_vertices

    return ring_lengths_list, ring_vertices_list

# Parameters
distance_step = min_value*4
tol = min_value

all_fps_points = farthest_point_sampling(mat, num_sample)
#visualization(all_fps_points)
fps_pairs = list(itertools.combinations(all_fps_points, 2))

print("FPS pairs:", fps_pairs)

psi = np.zeros(len(fps_pairs))
for n in range(len(fps_pairs)):
    # Compute geodesic rings and their lengths for each FPS point
    gi_n, gi_vertices_n = compute_geodesic_rings_for_point(mat, fps_pairs[n][0], distance_step, tol)
    gj_n, gj_vertices_n = compute_geodesic_rings_for_point(mat, fps_pairs[n][1], distance_step, tol)

    num_rings = min(len(gi_n), len(gj_n))
    for k in range(num_rings):
        if (gi_n[k] + gj_n[k]) == 0:
            continue
        psi[n] += abs((gi_n[k] - gj_n[k])/(gi_n[k] + gj_n[k]))/num_rings

min_psi = np.min(psi)
min_psi_index = np.argmin(psi)
print("Minimum psi value:", min_psi)
print("Index of minimum psi value:", min_psi_index)
fps_points = list(fps_pairs[min_psi_index])
#visualization(fps_points)
_, _, fps_distances_0 = dijkstra_shortest_path(mat, fps_points[0], -1)
_, _, fps_distances_1 = dijkstra_shortest_path(mat, fps_points[1], -1)

fps_distances_0_normalized = (fps_distances_0 - np.min(fps_distances_0)) / (np.max(fps_distances_0) - np.min(fps_distances_0))
fps_distances_1_normalized = (fps_distances_1 - np.min(fps_distances_1)) / (np.max(fps_distances_1) - np.min(fps_distances_1))

threshold = 0.5
boundary_points_0 = np.where((fps_distances_1 - fps_distances_0)>threshold)[0]
boundary_points_1 = np.where((fps_distances_0 - fps_distances_1)>threshold)[0]

if fps_points[0] not in boundary_points_0:
    boundary_points_0 = np.append(boundary_points_0, fps_points[0])

if fps_points[1] not in boundary_points_1:
    boundary_points_1 = np.append(boundary_points_1, fps_points[1])

print("FPS point 0:", fps_points[0])
print("FPS point 1:", fps_points[1])

print("Boundary points for FPS point 0:", boundary_points_0)
print("Boundary points for FPS point 1:", boundary_points_1)


boundary_points = np.concatenate((boundary_points_0, boundary_points_1))

# Visualize boundary points
boundary_pcd = o3d.geometry.PointCloud()
boundary_pcd.points = o3d.utility.Vector3dVector(vertices[boundary_points])
boundary_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(boundary_points))])

# Visualize the mesh and boundary points together
#o3d.visualization.draw_geometries([mesh, boundary_pcd], window_name="Boundary Points Visualization")

print("Farthest Point Sampling (FPS) points:", fps_points)
print("Number of rings:", num_rings)

b_column = np.zeros(adjacency_matrix.shape[0], dtype=int)
for i in range(len(boundary_points_1)):
    b_column += adjacency_matrix[:, boundary_points_1[i]].copy()

laplacian_matrix = degree_matrix - adjacency_matrix

# Remove rows and columns corresponding to fps_points from laplacian_matrix
laplacian_matrix = np.delete(laplacian_matrix, sorted(boundary_points, reverse=True), axis=0)
laplacian_matrix = np.delete(laplacian_matrix, sorted(boundary_points, reverse=True), axis=1)

# Remove entries corresponding to fps_points from b_column
b_column = np.delete(b_column, sorted(boundary_points, reverse=True), axis=0)

f_column = np.linalg.solve(laplacian_matrix, b_column)

# Add 0 for the first FPS point and 1 for the second FPS point
full_f_column = np.zeros(vertices.shape[0])
full_f_column[boundary_points_0] = 0  # First FPS point
full_f_column[boundary_points_1] = 1  # Second FPS point

# Fill the remaining values from f_column
remaining_indices = [i for i in range(vertices.shape[0]) if i not in boundary_points]
for idx, value in zip(remaining_indices, f_column):
    full_f_column[idx] = value

print("Laplacian Matrix:\n", laplacian_matrix)
print("Boundary Condition Column:\n", b_column)
print("Solution Column:\n", full_f_column)

# Normalize f_column for color mapping
f_column_normalized = (2*np.abs(full_f_column - 0.5))**(1/2)
f_column_normalized = (f_column_normalized - np.min(f_column_normalized)) / (np.max(f_column_normalized) - np.min(f_column_normalized))

colormap = plt.cm.rainbow 
colors = colormap(f_column_normalized)[:, :3] 

# Create a mesh for visualization
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# Visualize the mesh with the color map
o3d.visualization.draw_geometries([mesh], window_name="Solution Color Map")

# Write the solution column to a text file
output_file = "solution_column.txt"
np.savetxt(output_file, full_f_column, fmt="%.6f", header="Solution Column (f_column)")

print(f"Solution column written to {output_file}")

b_column_file = "b_column.txt"
np.savetxt(b_column_file, b_column, fmt="%.6f", header="Boundary Condition Column (b_column)")