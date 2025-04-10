import open3d as o3d
import numpy as np

print("Processing the mesh...")

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

for i in range(n):
    for j in range(n):
        if mat[j, i] != 0:
            mat[i, j] = mat[j, i]

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
    u = end
    while prev[u] != -1:
        path.append((prev[u], u))
        u = prev[u]
    path.reverse()
    return path, dist

dist_matrix = np.zeros([n, n])
old_percent = 0
new_percent = 0
# Call Dijkstra's algorithm for all vertices to calculate the distance matrix
for i in range(n):
    new_percent = round(100*i/n)
    if new_percent != old_percent:
        print(f"{new_percent}% complete to calculate distance matrix")
    old_percent = new_percent
    _, distances = dijkstra_shortest_path(mat, i, -1)
    dist_matrix[i] = distances

formatted_dist_matrix = np.round(dist_matrix, 4)
# Save the distance matrix to a file
with open(f"M for {mesh_name}.txt", "w") as f:
    for row in formatted_dist_matrix:
        formatted_row = [str(float(x)) for x in row]
        f.write(" ".join(formatted_row) + "\n")

# Input points for the shortest path
start_point = int(input(f"Enter the start point (integer between 0 and {number_of_vertices - 1}): "))
end_point = int(input(f"Enter the end point (integer between 0 and {number_of_vertices - 1}): "))
print("Calculating the shortest path...")

# Find the shortest path
shortest_path, _ = dijkstra_shortest_path(mat, start_point, end_point)
print("Shortest path:", shortest_path)

# Create lines for the shortest path
lines = []
for u, v in shortest_path:
    lines.append([u, v])

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(vertices)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # Color lines red

line_material = o3d.visualization.rendering.MaterialRecord()
line_material.shader = "unlitLine"
line_material.line_width = 5

mesh.compute_vertex_normals()
mesh_material = o3d.visualization.rendering.MaterialRecord()
mesh_material.shader = "defaultLitTransparency"
mesh_material.base_color = [0.8, 0.8, 0.8, 0.5]

# Visualize the mesh and shortest paths
o3d.visualization.draw([{
        "name": "mesh",
        "geometry": mesh,
        "material": mesh_material
    },{
        "name": "line_set",
        "geometry": line_set,
        "material": line_material
    }])