import heapq
from collections import deque
from warehouse_spatial import SpatialWarehouse


class UFDS:
    """
    1. Union-Find Disjoint Set (Unidad 1) con compresión de caminos y unión por rango.
    """

    def __init__(self, elements: list[str]):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for item in self.parent:
            root = self.find(item)
            groups.setdefault(root, []).append(item)
        return groups


def bfs_co_occurrence(adj: dict[str, list[str]], start: str) -> list[str]:
    """
    2a. BFS (Unidad 1) sobre el grafo de co-ubicación para explorar niveles
    y detectar productos relacionados.
    """
    visited = {start}
    order = []
    queue = deque([start])

    while queue:
        curr = queue.popleft()
        order.append(curr)
        for nxt in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return order


def bfs_grid_path(warehouse: SpatialWarehouse, start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
    """
    2b. BFS sobre la grilla espacial (explora por capas sin heurística).
    """
    s = warehouse.nearest_walkable(*start)
    g = warehouse.nearest_walkable(*goal)
    if s == g:
        return [s], 0.0

    queue = deque([s])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {s: None}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        curr = queue.popleft()
        if curr == g:
            path = []
            node: tuple[int, int] | None = curr
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            dist = (len(path) - 1) * warehouse.cell_size_m
            return path, dist

        for dx, dy in directions:
            nxt = (curr[0] + dx, curr[1] + dy)
            if warehouse.is_walkable(*nxt) and nxt not in came_from:
                came_from[nxt] = curr
                queue.append(nxt)

    return [s], float("inf")


def dijkstra_path(warehouse: SpatialWarehouse, start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
    """
    3a. Dijkstra (costo acumulado real sin heurística dirigida).
    """
    s = warehouse.nearest_walkable(*start)
    g = warehouse.nearest_walkable(*goal)
    if s == g:
        return [s], 0.0

    pq = [(0.0, s)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {s: None}
    dist_map: dict[tuple[int, int], float] = {s: 0.0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while pq:
        curr_d, curr = heapq.heappop(pq)
        if curr == g:
            path = []
            node: tuple[int, int] | None = curr
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, curr_d

        if curr_d > dist_map.get(curr, float("inf")):
            continue

        for dx, dy in directions:
            nxt = (curr[0] + dx, curr[1] + dy)
            if not warehouse.is_walkable(*nxt):
                continue
            nd = curr_d + warehouse.cell_size_m
            if nd < dist_map.get(nxt, float("inf")):
                dist_map[nxt] = nd
                came_from[nxt] = curr
                heapq.heappush(pq, (nd, nxt))

    return [s], float("inf")


def astar_path(warehouse: SpatialWarehouse, start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
    """
    3b. Algoritmo Voraz / Informado A* (Unidad 2) con heurística Manhattan
    para esquivar estanterías y calcular la ruta física exacta en metros.
    """
    s = warehouse.nearest_walkable(*start)
    g = warehouse.nearest_walkable(*goal)

    if s == g:
        return [s], 0.0

    open_heap = [(warehouse.manhattan_distance(s, g), 0.0, s)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {s: None}
    g_score: dict[tuple[int, int], float] = {s: 0.0}

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current == g:
            path = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, current_g

        if current_g > g_score.get(current, float("inf")):
            continue

        for dx, dy in directions:
            nxt = (current[0] + dx, current[1] + dy)
            if not warehouse.is_walkable(*nxt):
                continue
            tentative_g = current_g + warehouse.cell_size_m
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + warehouse.manhattan_distance(nxt, g, warehouse.cell_size_m)
                heapq.heappush(open_heap, (f_score, tentative_g, nxt))

    return [s], float("inf")


def solve_tsp_picking(points: list[tuple[int, int]], warehouse: SpatialWarehouse) -> tuple[list[int], float]:
    """
    4. Programación Dinámica (Held-Karp para k <= 12) y Heurística Vecino Más Cercano + 2-Opt
    para órdenes mayores, minimizando la distancia total de recolección (picking routing).
    """
    n = len(points)
    if n <= 1:
        return list(range(n)), 0.0

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = warehouse.manhattan_distance(points[i], points[j], warehouse.cell_size_m)

    if n <= 12:
        inf = float("inf")
        dp = [[inf] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        dp[1][0] = 0.0

        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)) or dp[mask][u] == inf:
                    continue
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    nxt_mask = mask | (1 << v)
                    cand = dp[mask][u] + dist[u][v]
                    if cand < dp[nxt_mask][v]:
                        dp[nxt_mask][v] = cand
                        parent[nxt_mask][v] = u

        full_mask = (1 << n) - 1
        best_cost = inf
        last_node = 0
        for u in range(1, n):
            cost = dp[full_mask][u] + dist[u][0]
            if cost < best_cost:
                best_cost = cost
                last_node = u

        route = []
        curr_mask = full_mask
        curr = last_node
        while curr != -1:
            route.append(curr)
            prev = parent[curr_mask][curr]
            curr_mask ^= (1 << curr)
            curr = prev
        route.reverse()
        route.append(0)
        return route, best_cost

    visited = [False] * n
    visited[0] = True
    tour = [0]
    curr = 0
    for _ in range(n - 1):
        nxt = min((j for j in range(n) if not visited[j]), key=lambda j: dist[curr][j])
        visited[nxt] = True
        tour.append(nxt)
        curr = nxt
    tour.append(0)

    def tour_length(t: list[int]) -> float:
        return sum(dist[t[i]][t[i + 1]] for i in range(len(t) - 1))

    improved = True
    while improved:
        improved = False
        best_len = tour_length(tour)
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                candidate = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                c_len = tour_length(candidate)
                if c_len < best_len - 1e-9:
                    tour = candidate
                    best_len = c_len
                    improved = True

    return tour, tour_length(tour)