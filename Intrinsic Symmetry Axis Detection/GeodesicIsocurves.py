import open3d as o3d
import numpy as np
import itertools

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

def visualization(symmetry_vertices):
    # Create a mesh from the vertices and triangles
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()

    # Create a point cloud for the FPS points
    fps_pcd = o3d.geometry.PointCloud()
    fps_pcd.points = o3d.utility.Vector3dVector(vertices[symmetry_vertices])
    fps_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(symmetry_vertices))])

    # Visualize the FPS points on the mesh
    mesh.compute_vertex_normals()
    mesh_material = o3d.visualization.rendering.MaterialRecord()
    mesh_material.shader = "defaultLitTransparency"
    mesh_material.base_color = [0.8, 0.8, 0.8, 0.5]

    fps_pcd = o3d.geometry.PointCloud()
    fps_pcd.points = o3d.utility.Vector3dVector(vertices[symmetry_vertices])
    fps_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(symmetry_vertices))])

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

# Function to compute the shortest paths between FPS points and find intersection points
def compute_paths_and_intersections(symmetry_points, mat):
    num_samples = len(symmetry_points)
    paths = []
    all_vertices_in_paths = set()
    used_index = []
    start_index = min(range(len(symmetry_points)), key=lambda idx: vertices[symmetry_points[idx]][1])
    i = start_index

    while len(paths) < num_samples - 1:
        min_dist = np.inf
        min_index = -1
        for j in range(num_samples):
            if j != i and j not in used_index:
                _, _, distances = dijkstra_shortest_path(mat, symmetry_points[i], symmetry_points[j])
                if distances[symmetry_points[j]] < min_dist:
                    min_dist = distances[symmetry_points[j]]
                    min_index = j
        if min_index == -1:
            break
        path, vertices_in_path, _ = dijkstra_shortest_path(mat, symmetry_points[i], symmetry_points[min_index])
        paths.append(path)
        all_vertices_in_paths.update(vertices_in_path)
        used_index.append(i)
        i = min_index

    return paths, all_vertices_in_paths, start_index, i

def visualize_geodesic_ring(mesh, vertices, vis_vertices, paths):
    # Create a point cloud for the overlap vertices
    overlap_pcd = o3d.geometry.PointCloud()
    overlap_pcd.points = o3d.utility.Vector3dVector(vertices[vis_vertices])
    overlap_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(vis_vertices))])

    # Create a line set for the paths
    lines = []
    for path in paths:
        for edge in path:
            lines.append(edge)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])

    # Visualize the mesh and the overlap vertices
    mesh.compute_vertex_normals()
    mesh_material = o3d.visualization.rendering.MaterialRecord()
    mesh_material.shader = "defaultLitTransparency"
    mesh_material.base_color = [0.8, 0.8, 0.8, 0.8]

    overlap_material = o3d.visualization.rendering.MaterialRecord()
    overlap_material.point_size = 10

    o3d.visualization.draw([{
            "name": "mesh",
            "geometry": mesh,
            "material": mesh_material
        }, {
            "name": "overlap_pcd",
            "geometry": overlap_pcd,
            "material": overlap_material
        }, {
            "name": "paths",
            "geometry": line_set
        }])

# Parameters
distance_step = min_value*4 
tol = min_value

all_fps_points = farthest_point_sampling(mat, num_sample)
#visualization(all_fps_points)
fps_pairs = list(itertools.combinations(all_fps_points, 2))

print("FPS pairs:", fps_pairs)

psi = np.zeros(len(fps_pairs))
gi = []
gj = []
gi_vertices = []
gj_vertices = []
for n in range(len(fps_pairs)):
    # Compute geodesic rings and their lengths for each FPS point
    gi_n, gi_vertices_n = compute_geodesic_rings_for_point(mat, fps_pairs[n][0], distance_step, tol)
    gj_n, gj_vertices_n = compute_geodesic_rings_for_point(mat, fps_pairs[n][1], distance_step, tol)

    gi.append(gi_n)
    gj.append(gj_n)
    gi_vertices.append(gi_vertices_n)
    gj_vertices.append(gj_vertices_n)

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
gi_min = gi[min_psi_index]
gj_min = gj[min_psi_index]
gi_vertices_min = gi_vertices[min_psi_index]
gj_vertices_min = gj_vertices[min_psi_index]
print("FPS points with minimum psi value:", fps_points)
num_rings = min(len(gi_min), len(gj_min))
#visualization(fps_points)

_, _, fps_distances_1 = dijkstra_shortest_path(mat, fps_points[0], -1)
_, _, fps_distances_2 = dijkstra_shortest_path(mat, fps_points[1], -1)

distance_threshold = min_value  # Adjust this value as needed

# Find vertices with close values in fps_distances_1 and fps_distances_2
close_vertices = np.where(np.abs(fps_distances_1 - fps_distances_2) < distance_threshold)[0]

#visualization(close_vertices)

# Print the vertices in the geodesic rings for each FPS point
print("Geodesic rings for FPS point 1 (vertices):")
for k in range(num_rings):
    print(f"Ring {k}: Vertices = {gi_vertices_min[k]}")

print("Geodesic rings for FPS point 1 (vertices):")
for k in range(num_rings):
    print(f"Ring {k}: Vertices = {gj_vertices_min[k]}")

overlap_vertices = {}
symmetry_vertices = set()
for k_start in range(num_rings):
    overlap_vertices = np.intersect1d(gi_vertices_min[k_start], gj_vertices_min[k_start])
    if len(overlap_vertices) >= 2:
        symmetry_vertices.update(overlap_vertices)
        break

k_end = num_rings - 1
while k_end >= 0:
    overlap_vertices = np.intersect1d(gi_vertices_min[k_end], gj_vertices_min[k_end])
    if len(overlap_vertices) >= 2:
        symmetry_vertices.update(overlap_vertices)
        break
    else:
        k_end -= 1

symmetry_vertices = sorted(set(symmetry_vertices).union(close_vertices))

filtered_mat = mat.copy()

print("Number of rings:", num_rings)
#symmetry_vertices = np.concatenate(symmetry_vertices)
print("Symmetry vertices:", symmetry_vertices)
#visualization(symmetry_vertices)
paths, all_vertices_in_paths, start_index, end_index = compute_paths_and_intersections(symmetry_vertices, mat)
last_point = symmetry_vertices[end_index]
first_point = symmetry_vertices[start_index]

for v in range(mat.shape[0]):  # Iterate over all vertices
    if v in all_vertices_in_paths and v != last_point and v != first_point:  
        # Keep last and first points reachable
        for u in range(mat.shape[0]):  # Iterate over all vertices
            if mat[v, u] != 0:  # Check if the element is not 0
                filtered_mat[v, u] = np.inf  # Make previously used vertices unreachable
                filtered_mat[u, v] = np.inf  # Ensure symmetry

# Run Dijkstra's algorithm from the last point to the first point
path, vertices_in_path, _ = dijkstra_shortest_path(filtered_mat, last_point, first_point)
all_vertices_in_paths.update(vertices_in_path)
paths.append(path)

# Visualize the sufficient_overlap vertices
visualize_geodesic_ring(mesh, vertices, symmetry_vertices, paths)


