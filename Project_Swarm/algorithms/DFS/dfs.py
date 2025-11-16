import time
def read_distance_matrix(filename):
    with open(filename, 'r') as f:
        matrix = [list(map(float, line.strip().split())) for line in f]
    return matrix

def dfs_tsp(distance_matrix, start=0):
    n = len(distance_matrix)
    visited = [False]*n
    path = []
    total_distance = 0

    def dfs(node):
        nonlocal total_distance
        visited[node] = True
        path.append(node)
        nearest = None
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and distance_matrix[node][i] < min_dist:
                min_dist = distance_matrix[node][i]
                nearest = i
        if nearest is not None:
            total_distance += distance_matrix[node][nearest]
            dfs(nearest)

    dfs(start)
    total_distance += distance_matrix[path[-1]][start]
    path.append(start)
    return path, total_distance
