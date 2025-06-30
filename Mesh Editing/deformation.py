import open3d as o3d
import numpy as np
import copy
import heapq

print("Processing the mesh...")

mesh_name = "homer.obj"
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
def farthest_point_sampling(mat, num_sample):
    n = mat.shape[0]
    selected_points = []
    _, rand_distance = dijkstra_shortest_path(mat, np.random.randint(n), -1)
    next_point = np.argmax(rand_distance)
    selected_points.append(next_point)
    _, distances = dijkstra_shortest_path(mat, selected_points[0], -1)

    for _ in range(1, num_sample):
        _, new_distances = dijkstra_shortest_path(mat, next_point, -1)
        distances = np.minimum(distances, new_distances)
        next_point = np.argmax(distances)
        selected_points.append(next_point)

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

def get_near_points(mat, selected_points, k):
    near_points = set(selected_points)
    for idx in selected_points:
        _, dist = dijkstra_shortest_path(mat, idx, -1)
        
        nearest = np.argsort(dist)
        count = 0
        for n in nearest:
            if n != idx and n not in near_points and dist[n] < np.inf:
                near_points.add(n)
                count += 1
                if count >= k:
                    break
    return list(near_points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)

picked = pick_points(pcd)

if not picked:
    selected_points = [0]
else:
    selected_points = picked

# Add near points to selected points using Dijkstra
k = 800 if mesh_name == "homer.obj" else 143
near_points = get_near_points(mat, selected_points, k)

print("Selected points (including near points):", near_points)

# Visualize selected points (green) and near points (red)
pcd_colors = np.tile([0.7, 0.7, 0.7], (number_of_vertices, 1))
for idx in near_points:
    pcd_colors[idx] = [1, 0, 0]  
for idx in selected_points:
    pcd_colors[idx] = [0, 1, 0] 

pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
o3d.visualization.draw_geometries([pcd], window_name="Selected (green) and Near (red) Points")

control_points = near_points  # Region of Interest (ROI)

s = 1. # Scale factor for control points
w = 1**2 # Weight for other points

diff_x[control_points] = s*diff_x[control_points]
diff_y[control_points] = s*diff_y[control_points]
diff_z[control_points] = s*diff_z[control_points]

weights = np.full(number_of_vertices, w)
weights[control_points] = 1.

boundary_x = laplacian @ diff_x
boundary_y = laplacian @ diff_y
boundary_z = laplacian @ diff_z

matrix = np.zeros((number_of_vertices, number_of_vertices))

matrix = laplacian @ laplacian
x_matrix = matrix.copy()
y_matrix = matrix.copy()
z_matrix = matrix.copy()

def_x = float(input("Input x direction of deformation: "))
def_y = float(input("Input y direction of deformation: "))
def_z = float(input("Input z direction of deformation: "))

# Compute Euclidean distances from the selected point to all near points
selected_idx = selected_points[0]
selected_coord = vertices[selected_idx]
dist_from_selected = np.linalg.norm(vertices - selected_coord, axis=1)

mesh_with_selected = o3d.geometry.TriangleMesh()
mesh_with_selected.vertices = o3d.utility.Vector3dVector(vertices)
mesh_with_selected.triangles = o3d.utility.Vector3iVector(triangles)
mesh_with_selected.compute_vertex_normals()
vertex_colors = np.tile([0.7, 0.7, 0.7], (number_of_vertices, 1))

radius = 0.01 if mesh_name == "homer.obj" else 0.2

balls = []
for idx in selected_points:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=30)
    sphere.paint_uniform_color([1, 1, 0])
    sphere.translate(vertices[idx])
    sphere.compute_vertex_normals()
    balls.append(sphere)

deformed_coord = selected_coord + np.array([def_x, def_y, def_z])
deformed_sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=30)
deformed_sphere.paint_uniform_color([1, 1, 0])
deformed_sphere.translate(deformed_coord)
deformed_sphere.compute_vertex_normals()
balls.append(deformed_sphere)

o3d.visualization.draw_geometries([mesh_with_selected] + balls, window_name="Selected Points (Green) on Mesh with Balls")

max_dist = max([dist_from_selected[idx] for idx in control_points])

for i in range(number_of_vertices):
    x_matrix[i, i] += weights[i]
    y_matrix[i, i] += weights[i]
    z_matrix[i, i] += weights[i]

    if i in control_points:
        d = dist_from_selected[i]
        x = d / max_dist
        #t = 1 - x  # Transition factor based on distance
        t = (1.0 - np.tanh(4*(x-0.5)))/2. # Smooth transition based on distance
        if t < 0: t = 0.

        boundary_x[i] += (x_coords[i] + def_x * t)*weights[i]
        boundary_y[i] += (y_coords[i] + def_y * t)*weights[i]
        boundary_z[i] += (z_coords[i] + def_z * t)*weights[i]
    else:
        boundary_x[i] += x_coords[i]*weights[i]
        boundary_y[i] += y_coords[i]*weights[i]
        boundary_z[i] += z_coords[i]*weights[i]

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

# Visualize the original and new meshes
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
o3d.visualization.draw_geometries([new_mesh], mesh_show_back_face=True)

# Save the new mesh
output_mesh_name = "output_" + mesh_name
o3d.io.write_triangle_mesh(output_mesh_name, new_mesh)
print(f"New mesh saved as {output_mesh_name}")

def mesh_volume(vertices, triangles):
    v = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        v += np.dot(np.cross(v0, v1), v2)
    return abs(v) / 6.0

volume = mesh_volume(np.asarray(new_mesh.vertices), np.asarray(new_mesh.triangles))
print(f"Volume of the resultant mesh: {volume}")

