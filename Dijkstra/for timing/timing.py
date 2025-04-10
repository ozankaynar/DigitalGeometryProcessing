import open3d as o3d
import numpy as np
import heapq
import time

print("Processing the mesh...")

mesh_name = "centaur.off"
mesh = o3d.io.read_triangle_mesh(mesh_name)

vertices = np.asarray(mesh.vertices)
number_of_vertices = len(vertices)

triangles = np.asarray(mesh.triangles)
number_of_triangles = len(triangles)

n = number_of_vertices
# Edge lengths
mat = np.zeros((n, n))
for i in range(number_of_triangles):
    mat[triangles[i,0], triangles[i,1]] = np.linalg.norm(vertices[triangles[i,0]] - vertices[triangles[i,1]])
    mat[triangles[i,0], triangles[i,2]] = np.linalg.norm(vertices[triangles[i,0]] - vertices[triangles[i,2]])
    mat[triangles[i,1], triangles[i,2]] = np.linalg.norm(vertices[triangles[i,1]] - vertices[triangles[i,2]])
    mat[triangles[i,1], triangles[i,0]] = mat[triangles[i,0], triangles[i,1]]
    mat[triangles[i,2], triangles[i,0]] = mat[triangles[i,0], triangles[i,2]]
    mat[triangles[i,2], triangles[i,1]] = mat[triangles[i,1], triangles[i,2]]

def dijkstra_shortest_path_array(mat, start, end):
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

def dijkstra_shortest_path_heap(mat, start, end):
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

# Input points for the shortest path
start_point = int(input(f"Enter the start point (integer between 0 and {n - 1}): "))
end_point = int(input(f"Enter the end point (integer between 0 and {n - 1}): "))
print("Calculating the shortest path...")

# Measure time for array implementation
start_time = time.time()
shortest_path_array, dist_array = dijkstra_shortest_path_array(mat, start_point, end_point)
array_time = time.time() - start_time
print("Array implementation time:", array_time)

# Measure time for min-heap implementation
start_time = time.time()
shortest_path_heap, dist_heap = dijkstra_shortest_path_heap(mat, start_point, end_point)
heap_time = time.time() - start_time
print("Min-heap implementation time:", heap_time)