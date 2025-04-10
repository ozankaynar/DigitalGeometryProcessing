import open3d as o3d
import numpy as np
from collections import Counter
from scipy.spatial import Delaunay

print("Processing the mesh...")

mesh_name = "353.off"
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
def farthest_point_sampling(mat, num_samples):
    n = mat.shape[0]
    selected_points = []
    _, _, rand_distance = dijkstra_shortest_path(mat, np.random.randint(n), -1)
    next_point = np.argmax(rand_distance)
    selected_points.append(next_point)
    _, _, distances = dijkstra_shortest_path(mat, selected_points[0], -1)

    for _ in range(1, num_samples):
        _, _, new_distances = dijkstra_shortest_path(mat, next_point, -1)
        distances = np.minimum(distances, new_distances)
        next_point = np.argmax(distances)
        selected_points.append(next_point)

    return selected_points

# Function to compute the shortest paths between FPS points and find intersection points
def compute_paths_and_intersections(fps_points, mat, num_samples):
    paths = []
    all_vertices_in_paths = []
    used_index = []
    i = 0
    while len(paths) < num_samples - 1:
        min_dist = np.inf
        min_index = -1
        for j in range(num_samples):
            if j != i and j not in used_index:
                _, _, distances = dijkstra_shortest_path(mat, fps_points[i], fps_points[j])
                if distances[fps_points[j]] < min_dist:
                    min_dist = distances[fps_points[j]]
                    min_index = j
        if min_index == -1:
            break
        path, vertices_in_path, _ = dijkstra_shortest_path(mat, fps_points[i], fps_points[min_index])
        paths.append(path)
        all_vertices_in_paths.extend(vertices_in_path)
        used_index.append(i)
        i = min_index
    path, vertices_in_path, _ = dijkstra_shortest_path(mat, fps_points[i], fps_points[0])
    paths.append(path)
    used_index.append(i)
    all_vertices_in_paths.extend(vertices_in_path)

    point_counts = Counter(all_vertices_in_paths)
    intersection_points = {point for point, count in point_counts.items() if count == 2}

    return paths, intersection_points, all_vertices_in_paths, used_index

# loop to find FPS points and compute paths until the number of intersection points are 4
num_samples = 4
intersection_points = set()
iteration = 0
print("Perform FPS and compute paths until the number of intersection points are 4")
while len(intersection_points) != 4:
    iteration += 1
    print(f"Trial {iteration}:")
    fps_points = farthest_point_sampling(mat, num_samples)
    print("FPS points:", fps_points)
    paths, intersection_points, all_vertices_in_paths, used_index = compute_paths_and_intersections(fps_points, mat, num_samples)
    print("Intersection points:", intersection_points)

# Improve the mesh by adding more points by drawing another two paths between the FPS points
_, vertices_in_path, _ = dijkstra_shortest_path(mat, fps_points[used_index[0]], fps_points[used_index[2]])
all_vertices_in_paths.extend(vertices_in_path)
_, vertices_in_path, _ = dijkstra_shortest_path(mat, fps_points[used_index[1]], fps_points[used_index[3]])
all_vertices_in_paths.extend(vertices_in_path)

# Combine FPS points and all points on the paths for the mesh
all_points_indices = np.unique(np.concatenate((fps_points, all_vertices_in_paths)))
all_points = vertices[all_points_indices]

# Create mesh for the patch
projected_points = all_points[:, :2]
tri = Delaunay(projected_points)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(all_points)
mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)

# Create point cloud for the patch
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)

# Save the patch to an OFF file
o3d.io.write_triangle_mesh("patch.off", mesh)

# Visualize the patch
""" vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
vis.add_geometry(mesh)
opt = vis.get_render_option()
opt.mesh_show_back_face = True
opt.mesh_show_wireframe = True
vis.run()
vis.destroy_window() """

# ...existing code...

# Visualize the patch
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
vis.add_geometry(mesh)
opt = vis.get_render_option()
opt.mesh_show_back_face = True
opt.mesh_show_wireframe = True

# Set the mesh to be semi-transparent
mesh_material = o3d.visualization.rendering.MaterialRecord()
mesh_material.shader = "defaultLitTransparency"
mesh_material.base_color = [1.0, 1.0, 1.0, 0.5]  # RGBA, where A is the alpha value for transparency
vis.get_render_option().mesh_material = mesh_material

vis.run()
vis.destroy_window()