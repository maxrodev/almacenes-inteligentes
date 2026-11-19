import math
import time
from flask import Flask, jsonify, render_template, request
from algorithms_engine import (
    UFDS,
    astar_path,
    bfs_co_occurrence,
    bfs_grid_path,
    dijkstra_path,
    solve_tsp_picking,
)
from warehouse_spatial import CATEGORY_PALETTE, SpatialWarehouse

app = Flask(__name__)

WAREHOUSE = SpatialWarehouse(width=30, height=20, cell_size_m=1.0, seed=42)
PRODUCTS = WAREHOUSE.load_or_generate_products(gexf_path="grafo_abid.gexf", fallback_nodes=2600)

UFDS_ENGINE = UFDS(PRODUCTS)
for u, v in WAREHOUSE.co_occurrence_graph.edges():
    UFDS_ENGINE.union(u, v)
COMPONENTS = sorted(UFDS_ENGINE.components().values(), key=len, reverse=True)


def build_cooccurrence_preview(limit: int = 36) -> dict:
    subset = PRODUCTS[:limit]
    subset_set = set(subset)
    nodes = []
    n = len(subset)
    for idx, sku in enumerate(subset):
        angle = (2 * math.pi * idx) / max(1, n)
        ring = 0.24 if idx % 3 == 0 else 0.40
        nx_pos = 0.5 + ring * math.cos(angle)
        ny_pos = 0.5 + ring * math.sin(angle)
        deg = int(WAREHOUSE.co_occurrence_graph.degree[sku]) if sku in WAREHOUSE.co_occurrence_graph else 0
        cat = WAREHOUSE.product_categories[sku]
        nodes.append({
            "sku": sku,
            "name": WAREHOUSE.product_names[sku],
            "category": cat,
            "color": CATEGORY_PALETTE[cat],
            "weight_kg": WAREHOUSE.product_weights_kg[sku],
            "coord": WAREHOUSE.product_coords[sku],
            "degree": deg,
            "x": round(nx_pos, 4),
            "y": round(ny_pos, 4),
        })

    edges = []
    for u, v, attrs in WAREHOUSE.co_occurrence_graph.edges(data=True):
        if u in subset_set and v in subset_set:
            edges.append({
                "source": u,
                "target": v,
                "weight": attrs.get("weight", 1),
            })

    return {"nodes": nodes, "edges": edges}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/warehouse-info", methods=["GET"])
def warehouse_info():
    sample_products = [
        {
            "sku": sku,
            "name": WAREHOUSE.product_names[sku],
            "category": WAREHOUSE.product_categories[sku],
            "color": CATEGORY_PALETTE[WAREHOUSE.product_categories[sku]],
            "weight_kg": WAREHOUSE.product_weights_kg[sku],
            "coord": WAREHOUSE.product_coords[sku],
        }
        for sku in PRODUCTS[:36]
    ]
    return jsonify({
        "width": WAREHOUSE.width,
        "height": WAREHOUSE.height,
        "grid": WAREHOUSE.grid,
        "total_nodes": len(PRODUCTS),
        "total_components": len(COMPONENTS),
        "categories": CATEGORY_PALETTE,
        "products": sample_products,
        "graph_preview": build_cooccurrence_preview(limit=36),
    })


def plan_trips_by_weight(ordered_skus: list[str], max_weight_kg: float) -> list[list[str]]:
    trips: list[list[str]] = []
    current_trip: list[str] = []
    current_weight = 0.0

    for sku in ordered_skus:
        w = WAREHOUSE.product_weights_kg[sku]
        if current_trip and (current_weight + w > max_weight_kg + 1e-9):
            trips.append(current_trip)
            current_trip = [sku]
            current_weight = w
        else:
            current_trip.append(sku)
            current_weight += w

    if current_trip:
        trips.append(current_trip)

    return trips


def trace_trips(trips: list[list[str]], depot: tuple[int, int], path_fn) -> tuple[list[tuple[int, int]], float, list[dict]]:
    total_dist = 0.0
    full_path: list[tuple[int, int]] = []
    stops_info: list[dict] = []

    for trip_idx, trip_skus in enumerate(trips, start=1):
        curr_pos = depot
        trip_weight = sum(WAREHOUSE.product_weights_kg[s] for s in trip_skus)

        for sku in trip_skus:
            shelf_coord = WAREHOUSE.product_coords[sku]
            seg, d = path_fn(WAREHOUSE, curr_pos, shelf_coord)
            total_dist += d
            full_path.extend(seg if not full_path else seg[1:])
            aisle_coord = seg[-1] if seg else depot
            curr_pos = aisle_coord

            stops_info.append({
                "sku": sku,
                "name": WAREHOUSE.product_names[sku],
                "category": WAREHOUSE.product_categories[sku],
                "weight_kg": WAREHOUSE.product_weights_kg[sku],
                "trip": trip_idx,
                "trip_weight_kg": round(trip_weight, 2),
                "shelf_coord": shelf_coord,
                "aisle_coord": aisle_coord,
            })

        seg_back, d_back = path_fn(WAREHOUSE, curr_pos, depot)
        total_dist += d_back
        full_path.extend(seg_back[1:])

    return full_path, total_dist, stops_info


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json() or {}
    selected_skus = data.get("skus", PRODUCTS[:6])
    max_weight_kg = max(0.5, float(data.get("max_weight_kg", 12.0)))
    robot_speed_mps = float(data.get("speed", 1.5))
    pick_time_sec = float(data.get("pick_time", 3.0))

    depot = (0, 0)
    waypoints = [depot] + [WAREHOUSE.product_coords[sku] for sku in selected_skus]
    n_items = len(selected_skus)

    # 1. Solo BFS (orden secuencial de llegada)
    naive_trips = plan_trips_by_weight(selected_skus, max_weight_kg)
    t_bfs = time.perf_counter()
    bfs_path, bfs_dist, bfs_stops = trace_trips(naive_trips, depot, bfs_grid_path)
    bfs_ms = (time.perf_counter() - t_bfs) * 1000.0
    bfs_time = (bfs_dist / robot_speed_mps) + (n_items * pick_time_sec)

    # 2. Solo Dijkstra (orden secuencial de llegada)
    t_dij = time.perf_counter()
    dij_path, dij_dist, dij_stops = trace_trips(naive_trips, depot, dijkstra_path)
    dij_ms = (time.perf_counter() - t_dij) * 1000.0
    dij_time = (dij_dist / robot_speed_mps) + (n_items * pick_time_sec)

    # 3. Solo A* Manhattan (orden secuencial de llegada)
    t_ast = time.perf_counter()
    astar_path_seq, astar_dist, astar_stops = trace_trips(naive_trips, depot, astar_path)
    astar_ms = (time.perf_counter() - t_ast) * 1000.0
    astar_time = (astar_dist / robot_speed_mps) + (n_items * pick_time_sec)

    # 4. A* Manhattan + TSP (Secuencia óptima + capacidad por viaje)
    t_opt = time.perf_counter()
    tour_indices, _ = solve_tsp_picking(waypoints, WAREHOUSE)
    tsp_ordered_skus = [selected_skus[idx - 1] for idx in tour_indices if idx != 0]
    opt_trips = plan_trips_by_weight(tsp_ordered_skus, max_weight_kg)
    opt_path, opt_dist, opt_stops = trace_trips(opt_trips, depot, astar_path)
    opt_ms = (time.perf_counter() - t_opt) * 1000.0
    opt_time = (opt_dist / robot_speed_mps) + (n_items * pick_time_sec)

    dist_saving_pct = ((astar_dist - opt_dist) / astar_dist * 100.0) if astar_dist > 0 else 0.0

    adj = {n: list(WAREHOUSE.co_occurrence_graph.neighbors(n)) for n in WAREHOUSE.co_occurrence_graph.nodes()}
    bfs_related = bfs_co_occurrence(adj, selected_skus[0])[:5] if selected_skus else []

    return jsonify({
        "routes": {
            "bfs": {
                "label": "Solo BFS (Secuencial)",
                "distance_m": round(bfs_dist, 2),
                "time_s": round(bfs_time, 2),
                "exec_ms": round(bfs_ms, 3),
                "trips_count": len(naive_trips),
                "path": bfs_path,
                "ordered_stops": bfs_stops,
            },
            "dijkstra": {
                "label": "Solo Dijkstra (Secuencial)",
                "distance_m": round(dij_dist, 2),
                "time_s": round(dij_time, 2),
                "exec_ms": round(dij_ms, 3),
                "trips_count": len(naive_trips),
                "path": dij_path,
                "ordered_stops": dij_stops,
            },
            "astar": {
                "label": "Solo A* Manhattan (Secuencial)",
                "distance_m": round(astar_dist, 2),
                "time_s": round(astar_time, 2),
                "exec_ms": round(astar_ms, 3),
                "trips_count": len(naive_trips),
                "path": astar_path_seq,
                "ordered_stops": astar_stops,
            },
            "astar_tsp": {
                "label": "A* Manhattan + TSP (Optimizado)",
                "distance_m": round(opt_dist, 2),
                "time_s": round(opt_time, 2),
                "exec_ms": round(opt_ms, 3),
                "trips_count": len(opt_trips),
                "max_weight_kg": max_weight_kg,
                "path": opt_path,
                "ordered_stops": opt_stops,
            },
        },
        "metrics": {
            "saving_percent": round(dist_saving_pct, 1),
            "bfs_cluster_sample": bfs_related,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)