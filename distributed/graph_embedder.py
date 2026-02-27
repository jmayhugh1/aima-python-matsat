import json
import random
import math
import os
from private_path_query_utils import Graph

def get_lon_scale(lat):
    return math.cos(math.radians(lat))

def distance_m(p1, p2, lon_scale):
    """p1, p2 are (lat, lon) tuples. Returns distance in meters."""
    dy = (p1[0] - p2[0]) * 111132.0
    dx = (p1[1] - p2[1]) * 111132.0 * lon_scale
    return math.sqrt(dx**2 + dy**2)

def point_to_segment_distance_m(p, a, b, lon_scale):
    """p, a, b are (lat, lon) tuples. Returns distance from p to segment ab in meters."""
    # Project to local meter-based Cartesian space relative to a
    ay, ax = 0.0, 0.0
    by = (b[0] - a[0]) * 111132.0
    bx = (b[1] - a[1]) * 111132.0 * lon_scale
    py = (p[0] - a[0]) * 111132.0
    px = (p[1] - a[1]) * 111132.0 * lon_scale
    
    # Vector V = B - A
    vy = by - ay
    vx = bx - ax
    
    # Vector W = P - A
    wy = py - ay
    wx = px - ax
    
    l2 = vx*vx + vy*vy
    if l2 == 0:
        return math.sqrt(wx*wx + wy*wy)
    
    # Projection parameter t
    t = (wx * vx + wy * vy) / l2
    t = max(0, min(1, t))
    
    # Closest point C on segment
    cy = ay + t * vy
    cx = ax + t * vx
    
    return math.sqrt((px - cx)**2 + (py - cy)**2)

def generate_spatial_embedding(graph: Graph, bounds_path: str, max_attempts=5000, strict=True, seed=None, min_dist_m=6.0, node_regions=None, safe_path_edge_indices=None):
    if seed is not None:
        random.seed(seed)
    with open(bounds_path, 'r') as f:
        config = json.load(f)
        
    tl = config['top_left']
    br = config['bottom_right']
    
    lat_max, lon_min = tl[0], tl[1]
    lat_min, lon_max = br[0], br[1]
    
    fountains = config.get('fountains', [])
    # Obstacle radius in meters, add 1.0m safety margin
    rad_m = config.get('obstacle_radius', 3.0) + 1.0
    lon_scale = get_lon_scale((lat_max + lat_min) / 2)
    
    vertices = list(graph.vertices)
    edges = list(graph.edges)
    
    # Pre-calculate adjacency map: node_id -> list of (neighbor_id, edge_index)
    adj = {v.id: [] for v in vertices}
    for i, e in enumerate(edges):
        adj[e.vertex1.id].append((e.vertex2.id, i))
        adj[e.vertex2.id].append((e.vertex1.id, i))
    
    def is_point_valid(lat, lon):
        p = (lat, lon)
        for f in fountains:
            if distance_m(p, f, lon_scale) < rad_m:
                return False
        return True
        
    def is_edge_valid(p1, p2, edge_idx):
        # Only enforce fountain avoidance if:
        # 1. Global strict mode is ON
        # 2. OR this edge is explicitly marked as part of the safe path
        should_avoid = strict or (safe_path_edge_indices is not None and edge_idx in safe_path_edge_indices)
        
        if not should_avoid:
            return True 
            
        for f in fountains:
            if point_to_segment_distance_m(f, p1, p2, lon_scale) < rad_m:
                return False
        return True

    # Backtracking placement
    coords = {}
    
    def backtrack(v_idx):
        if v_idx == len(vertices):
            return True
        
        v = vertices[v_idx]
        
        # Region constraints
        l_min, l_max = lat_min, lat_max
        ln_min, ln_max = lon_min, lon_max
        if node_regions and str(v.id) in node_regions:
            region = node_regions[str(v.id)]
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min
            if region == "top_left":
                l_min = lat_min + lat_range * 0.7
                ln_max = lon_min + lon_range * 0.3
            elif region == "bottom_right":
                l_max = lat_min + lat_range * 0.3
                ln_min = lon_min + lon_range * 0.7
            elif region == "top_right":
                l_min = lat_min + lat_range * 0.7
                ln_min = lon_min + lon_range * 0.7
            elif region == "bottom_left":
                l_max = lat_min + lat_range * 0.3
                ln_max = lon_min + lon_range * 0.3

        for attempt in range(max_attempts):
            lat = random.uniform(l_min, l_max)
            lon = random.uniform(ln_min, ln_max)
            p = (lat, lon)
            
            if not is_point_valid(lat, lon):
                continue
                
            # ENFORCE MINIMUM DISTANCE TO OTHER NODES
            too_close = False
            for prev_p in coords.values():
                if distance_m(p, prev_p, lon_scale) < min_dist_m:
                    too_close = True
                    break
            if too_close:
                continue
                
            # Check edges to already placed nodes
            edge_ok = True
            for neighbor_id, edge_idx in adj[v.id]:
                if neighbor_id in coords:
                    if not is_edge_valid(p, coords[neighbor_id], edge_idx):
                        edge_ok = False
                        break
            
            if edge_ok:
                coords[v.id] = p
                if backtrack(v_idx + 1):
                    return True
                del coords[v.id]
                
        return False
        
    success = backtrack(0)
    if success:
        return {
            str(v_id): {"lat": lat, "lon": lon} 
            for v_id, (lat, lon) in coords.items()
        }
    else:
        # Fallback to deterministic corners for small graphs just in case
        if len(vertices) <= 4:
            corners = [
                (lat_max, lon_min), # top-left
                (lat_min, lon_max), # bottom-right
                (lat_max, lon_max), # top-right
                (lat_min, lon_min)  # bottom-left
            ]
            return {
                str(vertices[i].id): {"lat": corners[i][0], "lon": corners[i][1]}
                for i in range(len(vertices))
            }
        raise RuntimeError("Could not find a valid spatial embedding.")

def get_fountain_collisions(graph: Graph, coords: dict, bounds_path: str):
    """Identify which edges in the graph intersect with fountains."""
    with open(bounds_path, 'r') as f:
        config = json.load(f)
    
    fountains = config.get('fountains', [])
    rad_m = (config.get('obstacle_radius', 3.0) + 1.0)
    
    tl = config['top_left']
    br = config['bottom_right']
    lon_scale = get_lon_scale((tl[0] + br[0]) / 2)
    
    intersecting_edge_indices = []
    edges = list(graph.edges)
    
    for i, edge in enumerate(edges):
        v1_id, v2_id = str(edge.vertex1.id), str(edge.vertex2.id)
        if v1_id in coords and v2_id in coords:
            p1 = (coords[v1_id]["lat"], coords[v1_id]["lon"])
            p2 = (coords[v2_id]["lat"], coords[v2_id]["lon"])
            
            for f_idx, f in enumerate(fountains):
                dist = point_to_segment_distance_m(f, p1, p2, lon_scale)
                if dist < rad_m:
                    intersecting_edge_indices.append(i)
                    break
                    
    return intersecting_edge_indices

def visualize_embedding(graph: Graph, coords: dict, bounds_path: str, output_path: str, path: list = None):
    """Generate a satellite map visualization of the graph embedding."""
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.io.img_tiles as cimgt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Visualization dependencies (matplotlib, cartopy) missing. Skipping visualization.")
        return

    with open(bounds_path, 'r') as f:
        config = json.load(f)
    
    tl = config['top_left']
    br = config['bottom_right']
    fountains = config.get('fountains', [])
    rad_m = config.get('obstacle_radius', 3.0)
    
    # For visualization we still need degrees for patches
    rad_deg_lat = rad_m / 111132.0
    lon_scale = get_lon_scale((tl[0] + br[0]) / 2)
    rad_deg_lon = rad_m / (111132.0 * lon_scale)
    
    tiler = cimgt.GoogleTiles(style='satellite')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    
    # Set extent with padding
    pad_lat = (tl[0] - br[0]) * 0.2
    pad_lon = (br[1] - tl[1]) * 0.2
    ax.set_extent([tl[1] - pad_lon, br[1] + pad_lon, br[0] - pad_lat, tl[0] + pad_lat], crs=ccrs.PlateCarree())
    
    
    # Try to add satellite image
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'satellite_background.png')
    
    added_image = False
    try:
        ax.add_image(tiler, 20)
        added_image = True
        # Save cache if possible (matplotlib doesn't make this easy here, but we can save the whole figure later)
    except Exception as e:
        print(f"Warning: Could not download satellite tiles ({e}).")
        if os.path.exists(cache_file):
            print("Using cached background image.")
            img = plt.imread(cache_file)
            ax.imshow(img, extent=ax.get_extent(), transform=tiler.crs)
            added_image = True

    transform = ccrs.PlateCarree()
    
    # Plot Boundary Box
    # [lon, lon, lon, lon, lon], [lat, lat, lat, lat, lat]
    box_lons = [tl[1], br[1], br[1], tl[1], tl[1]]
    box_lats = [tl[0], tl[0], br[0], br[0], tl[0]]
    ax.plot(box_lons, box_lats, 'w--', transform=transform, linewidth=2, alpha=0.8, zorder=8, label='Boundary')
    
    # Plot Fountains
    for f in fountains:
        # Ellipse to account for lat/lon scaling
        circ = mpatches.Ellipse((f[1], f[0]), width=rad_deg_lon*2, height=rad_deg_lat*2, transform=transform, color='red', alpha=0.3, zorder=5)
        ax.add_patch(circ)
        ax.plot(f[1], f[0], 'r^', transform=transform, markersize=8, zorder=6)
    
    # Plot Graph Edges
    from private_path_query_utils import EdgeState
    for edge in graph.edges:
        v1_id, v2_id = str(edge.vertex1.id), str(edge.vertex2.id)
        if v1_id in coords and v2_id in coords:
            p1 = coords[v1_id]
            p2 = coords[v2_id]
            
            # Color based on edge state
            if edge.state == EdgeState.BLOCKED:
                color = 'red'
                alpha = 0.6
                linewidth = 1.5
            elif edge.state == EdgeState.TRAVERSABLE:
                color = 'lime'
                alpha = 0.8
                linewidth = 1.5
            else: # UNKNOWN or NO_EDGE
                color = 'lightgrey'
                alpha = 0.4
                linewidth = 1.0
            
            ax.plot([p1["lon"], p2["lon"]], [p1["lat"], p2["lat"]], color=color, alpha=alpha, transform=transform, linewidth=linewidth, zorder=2)

    # Plot Nodes
    for v_id, p in coords.items():
        # Larger white circle with black edge for better visibility
        ax.plot(p["lon"], p["lat"], 'o', markerfacecolor='white', markeredgecolor='black', 
                markersize=18, transform=transform, zorder=10)
        # Center the text inside the node. Use va='center' and ha='center'.
        ax.text(p["lon"], p["lat"], v_id, transform=transform, fontsize=12, weight='bold', 
                color='black', ha='center', va='center', zorder=11)

    # Plot Path if provided
    if path:
        path_lats = []
        path_lons = []
        for node_id in path:
            p = coords[str(node_id)]
            path_lats.append(p["lat"])
            path_lons.append(p["lon"])
        # Dashed overlay for the path
        ax.plot(path_lons, path_lats, 'cyan', linestyle='--', transform=transform, linewidth=3, zorder=7, label='Safe Path')
        ax.legend()

    plt.title(f"PRM Spatial Embedding - {os.path.basename(bounds_path)}")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # If we successfully got an image but didn't have a cache, save the figure as a potential future background?
    # Not trivial without UI, so let's skip for now unless user really needs it.
    
    plt.close()

if __name__ == "__main__":
    # Small test
    from private_path_query_utils import Vertex, Edge, EdgeState
    v0, v1 = Vertex(0), Vertex(1)
    g = Graph([v0, v1], [Edge(v0, v1, EdgeState.TRAVERSABLE)])
    res = generate_spatial_embedding(g, "distributed/spatial_bounds.json")
    print(json.dumps(res, indent=2))
