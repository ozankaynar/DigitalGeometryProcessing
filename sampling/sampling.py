import open3d as o3d
import numpy as np

print("Processing the mesh...")
num_sample = 4

mesh_name = "man0.off"
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

fps_points = farthest_point_sampling(mat, num_sample)
print("Farthest Point Sampling (FPS) points:", fps_points)

# Visualize the FPS points on the mesh
mesh.compute_vertex_normals()
mesh_material = o3d.visualization.rendering.MaterialRecord()
mesh_material.shader = "defaultLitTransparency"
mesh_material.base_color = [0.8, 0.8, 0.8, 0.5]

fps_pcd = o3d.geometry.PointCloud()
fps_pcd.points = o3d.utility.Vector3dVector(vertices[fps_points])
fps_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(fps_points))])  # Color lines red

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