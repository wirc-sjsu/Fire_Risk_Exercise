import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# ============================================================
# Input data: 10 building classes represented by 2D points
# ============================================================
# These values are derived from the WUDAPT LCZ definitions and 
# represent typical building side lengths and separation lengths for each class.
aspect_ratio_min = np.array([2, 0.75, 0.75, 0.75, 0.3, 0.3, 1, 0.1, 0.1, 0.2]) # lower bound
aspect_ratio_max = np.array([5, 2, 1.5, 1.25, 0.75, 0.75, 2, 0.3, 0.25, 0.5]) # upper bound
aspect_ratio_mean = 0.5 * (aspect_ratio_min + aspect_ratio_max)

b_height_min = np.array([25, 10, 3, 25, 10, 3, 2, 3, 3, 5]) # lower bound
b_height_max = np.array([60, 25, 10, 40, 25, 10, 4, 10, 10, 15]) # upper bound
b_height_mean = 0.5 * (b_height_min + b_height_max)

b_surf_frac_min = np.array([0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.6, 0.3, 0.1, 0.2]) # lower bound
b_surf_frac_max = np.array([0.6, 0.7, 0.7, 0.4, 0.4, 0.4, 0.9, 0.5, 0.2, 0.3]) # upper bound
b_surf_frac_mean=0.5 * (b_surf_frac_min + b_surf_frac_max)

b_separation_len_min = b_height_min / aspect_ratio_min
b_side_len_min = b_separation_len_min * np.sqrt(b_surf_frac_min) / (1 - np.sqrt(b_surf_frac_min))

b_separation_len_max = b_height_max / aspect_ratio_max
b_side_len_max = b_separation_len_max * np.sqrt(b_surf_frac_max) / (1 - np.sqrt(b_surf_frac_max))

b_separation_len_mean = b_height_mean / aspect_ratio_mean
b_side_len_mean = b_separation_len_mean * np.sqrt(b_surf_frac_mean) / (1 - np.sqrt(b_surf_frac_mean))

# For simplicity, we will use the mean values as the class prototypes for Voronoi partitioning
b_side_len = b_side_len_mean
b_separation_len = b_separation_len_mean

# Adjust a few classes to better match typical LCZ prototypes 
# (e.g., LCZ9 is very sparse, so we use max separation and min side length)
b_side_len[9] = b_side_len_max[9]
b_separation_len[9] = b_separation_len_max[9]

b_side_len[3] = b_side_len_min[3]
b_separation_len[3] = b_separation_len_min[3]

b_side_len[5] = b_side_len_min[5]
class_ids = np.array([
    "LCZ1", "LCZ2", "LCZ3", "LCZ4", "LCZ5",
    "LCZ6", "LCZ7", "LCZ8", "LCZ9", "LCZ10"
])

# Assign numeric codes for easier indexing
class_codes = np.arange(1, len(class_ids) + 1, dtype=np.uint8)
# Create a mapping from class ID to code
points = np.column_stack([b_side_len, b_separation_len])

# ============================================================
# LCZ colors from uploaded WUDAPT colormap
# ============================================================
lcz_colors = {
    "LCZ1":  "#8c0000",
    "LCZ2":  "#d10000",
    "LCZ3":  "#ff0000",
    "LCZ4":  "#bf4d00",
    "LCZ5":  "#ff6600",
    "LCZ6":  "#ff9955",
    "LCZ7":  "#faee05",
    "LCZ8":  "#bcbcbc",
    "LCZ9":  "#ffccaa",
    "LCZ10": "#555555",
}

# ============================================================
# Build Voronoi and nearest-neighbor search structure
# ============================================================
vor = Voronoi(points)
tree = cKDTree(points)

# ============================================================
# Query function
# ============================================================
def get_building_class(side_len, separation_len, return_distance=False):
    query = np.array([side_len, separation_len], dtype=float)
    dist, idx = tree.query(query)

    if return_distance:
        return class_ids[idx], idx, dist
    return class_ids[idx], idx

# ============================================================
# Helper: convert infinite Voronoi regions to finite polygons
# Source logic adapted from standard SciPy Voronoi finite-polygons recipe
# ============================================================
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = 2 * np.ptp(vor.points, axis=0).max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

# ============================================================
# Print all class prototypes
# ============================================================
def print_class_table():
    print("Building class prototypes:")
    print("-" * 60)
    print(f"{'Class':<10} {'b_side_len [m]':>16} {'b_separation_len [m]':>24}")
    print("-" * 60)
    for cid, (x, y) in zip(class_ids, points):
        print(f"{cid:<10} {x:>16.2f} {y:>24.2f}")
    print("-" * 60)

# ============================================================
# Plot Voronoi diagram with LCZ-colored cells
# ============================================================
def plot_voronoi(query_point=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    regions, vertices = voronoi_finite_polygons_2d(vor, radius=200)

    patches = []
    facecolors = []

    # Build filled polygons, one per class seed
    for i, region in enumerate(regions):
        polygon = vertices[region]
        patches.append(Polygon(polygon, closed=True))
        facecolors.append(lcz_colors[class_ids[i]])

    pc = PatchCollection(
        patches,
        facecolor=facecolors,
        edgecolor="0.35",
        linewidth=1.2,
        alpha=0.85,
        zorder=1
    )
    ax.add_collection(pc)

    # Plot class prototype points
    ax.scatter(points[:, 0], points[:, 1], s=80, c="white", edgecolors="black", zorder=3)

    # Label each class
    for i, (x, y) in enumerate(points):
        txt_color = "white" if class_ids[i] in ["LCZ1", "LCZ2", "LCZ4", "LCZ10"] else "black"
        ax.text(
            x + 0.5, y + 0.5, class_ids[i],
            fontsize=10, weight="bold", color=txt_color, zorder=4
        )

    # Query point
    if query_point is not None:
        qx, qy = query_point
        class_id, idx, dist = get_building_class(qx, qy, return_distance=True)

        ax.scatter([qx], [qy], s=140, c="yellow", edgecolors="black", marker="*", zorder=5)
        ax.plot(
            [qx, points[idx, 0]],
            [qy, points[idx, 1]],
            ls="--", lw=1.5, c="black", zorder=4
        )
        ax.text(
            qx + 0.7, qy + 0.7,
            f"{class_id}\nd={dist:.2f}",
            fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", alpha=0.9),
            zorder=6
        )

    ax.set_xlabel("Length L [m]")
    ax.set_ylabel("Separation S [m]")
    ax.set_title("Voronoi Partition of Building Classes, LCZ Colors")
    ax.grid(True, alpha=0.20, zorder=0)

    pad_x = 5
    pad_y = 5
    ax.set_xlim(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x)
    ax.set_ylim(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("voronoi.png", dpi=200)

# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    print_class_table()

    q_side = 30.0
    q_sep = 18.0

    class_id, idx, dist = get_building_class(q_side, q_sep, return_distance=True)
    print(f"\nQuery point: ({q_side:.2f}, {q_sep:.2f})")
    print(f"Assigned building class: {class_id} (index={idx}, distance={dist:.3f})")

    plot_voronoi()