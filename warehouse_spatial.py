import os
import random
import networkx as nx


CATEGORY_PALETTE = {
    "Tecnología": "#38bdf8",
    "Outdoor y Hogar": "#f59e0b",
    "Salud y Cuidado": "#22c55e",
    "Oficina y Accesorios": "#c084fc",
}


class SpatialWarehouse:
    """
    Representa una planta de almacén con pasillos transitables (0)
    y bloques de estanterías/obstáculos (1) en coordenadas (x, y) en metros.
    """

    def __init__(self, width: int = 30, height: int = 20, cell_size_m: float = 1.0, seed: int = 42):
        self.width = width
        self.height = height
        self.cell_size_m = cell_size_m
        self.rng = random.Random(seed)
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.product_coords: dict[str, tuple[int, int]] = {}
        self.product_names: dict[str, str] = {}
        self.product_weights_kg: dict[str, float] = {}
        self.product_categories: dict[str, str] = {}
        self.co_occurrence_graph = nx.Graph()
        self._build_layout()

    def _build_layout(self) -> None:
        for y in range(2, self.height - 2):
            if y % 4 == 0:
                continue
            for x in range(3, self.width - 3):
                if x % 6 == 0:
                    continue
                self.grid[y][x] = 1

    def is_walkable(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == 0

    def nearest_walkable(self, x: int, y: int) -> tuple[int, int]:
        if self.is_walkable(x, y):
            return (x, y)

        candidates = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx_pos, ny_pos = x + dx, y + dy
                if self.is_walkable(nx_pos, ny_pos):
                    dist = abs(dx) + abs(dy)
                    candidates.append((dist, nx_pos, ny_pos))

        if not candidates:
            return (0, 0)
        candidates.sort()
        _, bx, by = candidates[0]
        return (bx, by)

    def load_or_generate_products(self, gexf_path: str = "grafo_abid.gexf", fallback_nodes: int = 2600) -> list[str]:
        friendly_catalog = [
            ("Hornillo de Camping", "Outdoor y Hogar"),
            ("Suplemento de Colágeno", "Salud y Cuidado"),
            ("Teclado Mecánico", "Tecnología"),
            ("Mouse Inalámbrico", "Tecnología"),
            ("Botella Térmica 1L", "Outdoor y Hogar"),
            ("Lámpara LED Escritorio", "Oficina y Accesorios"),
            ("Mochila Urbana 25L", "Outdoor y Hogar"),
            ("Audífonos Bluetooth", "Tecnología"),
            ("Cargador USB-C 65W", "Tecnología"),
            ("Cuaderno Tapa Dura", "Oficina y Accesorios"),
            ("Filtro de Agua Portátil", "Outdoor y Hogar"),
            ("Linterna Recargable", "Outdoor y Hogar"),
            ("Organizador de Cables", "Oficina y Accesorios"),
            ("Almohadilla XXL", "Oficina y Accesorios"),
            ("Termo de Acero", "Salud y Cuidado"),
            ("Candado de Seguridad", "Oficina y Accesorios"),
            ("Cinta Métrica 5m", "Oficina y Accesorios"),
            ("Adaptador HDMI", "Tecnología"),
            ("Batería Portátil 20k", "Tecnología"),
            ("Soporte de Laptop", "Oficina y Accesorios"),
        ]
        categories = list(CATEGORY_PALETTE.keys())

        if os.path.exists(gexf_path):
            self.co_occurrence_graph = nx.read_gexf(gexf_path)
            products = list(self.co_occurrence_graph.nodes())
        else:
            products = [f"ASIN_{i:04d}" for i in range(fallback_nodes)]
            self.co_occurrence_graph.add_nodes_from(products)
            # Generar relaciones de co-ubicación con afinidad dentro de categorías
            for i in range(0, len(products) - 4, 2):
                self.co_occurrence_graph.add_edge(products[i], products[i + 1], weight=self.rng.randint(1, 4))
                if i % 4 == 0 and i + 20 < len(products):
                    self.co_occurrence_graph.add_edge(products[i], products[i + 20], weight=self.rng.randint(1, 3))

        for idx, sku in enumerate(products):
            rx = self.rng.randint(1, self.width - 2)
            ry = self.rng.randint(1, self.height - 2)
            self.product_coords[sku] = (rx, ry)

            base_name, default_cat = friendly_catalog[idx % len(friendly_catalog)]
            attrs = self.co_occurrence_graph.nodes[sku]

            raw_name = attrs.get("name")
            if raw_name and str(raw_name).strip():
                self.product_names[sku] = str(raw_name).strip()[:36]
            else:
                self.product_names[sku] = f"{base_name} #{idx + 1}"

            raw_cat = attrs.get("category")
            if raw_cat in CATEGORY_PALETTE:
                self.product_categories[sku] = str(raw_cat)
            else:
                self.product_categories[sku] = default_cat if idx < len(friendly_catalog) else categories[idx % len(categories)]

            raw_weight = attrs.get("weight_kg")
            if raw_weight is not None:
                self.product_weights_kg[sku] = round(float(raw_weight), 2)
            else:
                self.product_weights_kg[sku] = round(self.rng.uniform(0.5, 4.5), 2)

        return products

    @staticmethod
    def manhattan_distance(p1: tuple[int, int], p2: tuple[int, int], scale_m: float = 1.0) -> float:
        return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) * scale_m