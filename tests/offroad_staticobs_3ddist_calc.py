import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, KDTree
import pyvista as pv
from sklearn.neighbors import NearestNeighbors


# ==================== 生成点云数据 ====================
def generate_polyhedron(num_points=300):
    vertices = np.random.rand(10, 3) * 5
    hull = ConvexHull(vertices)
    points = []
    for simplex in hull.simplices:
        for _ in range(num_points // len(hull.simplices)):
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            point = u * vertices[simplex[0]] + v * vertices[simplex[1]] + w * vertices[simplex[2]]
            points.append(point)
    return np.array(points)

def generate_1_3_cone(height=5, radius=3, num_points=500):
    theta = np.linspace(0, 2*np.pi/3, num_points)  # 仅生成120°圆弧
    z = np.linspace(0, height, num_points)
    r = radius * (1 - z/height)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # 平移至x轴正方向（远离自车）
    return np.column_stack((x, y, z)) + [10, 0, 0]


def generate_external_hemisphere(radius=4, num_points=400):
    phi = np.linspace(0, np.pi/2, int(np.sqrt(num_points)))
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(num_points)))
    phi, theta = np.meshgrid(phi, theta)
    # 法线朝外的半球，中心位置在(0, 10, 0)，自车位于原点(0,0,0)在其外部
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta) + 10  # 沿y轴正方向平移10米
    z = radius * np.cos(phi)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def generate_half_16hedron(radius=5, num_points=800):
    """生成半截断的16面体点云，切割面位于x=5，整体中心在(10,0,0)"""
    # 生成16面体顶点（正八面体扩展）
    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.extend([
                    [i * radius, j * radius, k * radius],
                    [i * radius * 0.5, j * radius * 1.5, k * radius * 0.5],
                    [i * radius * 1.5, j * radius * 0.5, k * radius * 0.5]
                ])
    vertices = np.unique(np.array(vertices), axis=0)

    # 切割x>=5的半个多面体
    vertices = vertices[vertices[:, 0] >= 5]  # 保留x≥5的部分
    vertices += [5, 0, 0]  # 平移使切割面位于x=5，整体中心移至(10,0,0)

    # 在凸包表面均匀采样
    hull = ConvexHull(vertices)
    points = []
    for simplex in hull.simplices:
        for _ in range(num_points // len(hull.simplices)):
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            point = u * vertices[simplex[0]] + v * vertices[simplex[1]] + w * vertices[simplex[2]]
            points.append(point)
    return np.array(points)


def generate_cone_3d(height=6, base_radius=4, num_points=2000):
    """生成三维圆锥体点云，包含底面和侧面"""
    # 底面圆周点
    theta = np.linspace(0, 2 * np.pi, int(num_points * 0.3))
    x_base = base_radius * np.cos(theta)
    y_base = base_radius * np.sin(theta)
    z_base = np.zeros_like(theta)

    # 侧面点（参数化）
    z_side = np.linspace(0, height, int(num_points * 0.7))
    r_side = base_radius * (1 - z_side / height)
    theta_side = np.random.rand(len(z_side)) * 2 * np.pi
    x_side = r_side * np.cos(theta_side)
    y_side = r_side * np.sin(theta_side)

    # 合并并平移至(15,0,0)
    points = np.vstack((
        np.column_stack((x_base, y_base, z_base)),
        np.column_stack((x_side, y_side, z_side))
    )) + [15, 0, 0]
    return points

def generate_half_16hedron_3d(radius=5, num_points=1500):
    """生成半16面体，切割平面x=5，保留x≥5部分"""
    # 生成正16面体顶点（由两个八面体组合）
    vertices = []
    for i in [-1, 1]:
        vertices.extend([
            [i * radius, 0, 0], [0, i * radius, 0], [0, 0, i * radius],
            [i * radius * 0.5, i * radius * 0.5, i * radius * 0.5]
        ])
    vertices = np.unique(np.array(vertices), axis=0)

    # 切割保留x≥5的部分，并平移至x=5右侧
    vertices = vertices[vertices[:, 0] >= 0]  # 原始坐标系切割
    vertices[:, 0] += 5  # 平移切割面到x=5

    # 生成凸包表面点云
    hull = ConvexHull(vertices)
    points = []
    for simplex in hull.simplices:
        for _ in range(num_points // len(hull.simplices)):
            # 三角形面内插值
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            point = u * vertices[simplex[0]] + v * vertices[simplex[1]] + w * vertices[simplex[2]]
            points.append(point)
    return np.array(points)

poly = generate_polyhedron()
cone = generate_cone_3d()
hemi = generate_external_hemisphere()
hedron = generate_half_16hedron_3d()
points = np.vstack((poly, cone, hedron))

# ==================== DBSCAN聚类 ====================
db = DBSCAN(eps=1.5, min_samples=10).fit(points)
db_labels = db.labels_
db_clusters = [points[db_labels == i] for i in np.unique(db_labels) if i != -1]


# ==================== 区域生长聚类 ====================
def region_growing(points, normal_threshold=30, curvature_threshold=0.1):
    # 修正kneighbors返回值的解包
    neigh = NearestNeighbors(n_neighbors=20).fit(points)
    normals = []
    curvatures = []

    for i in range(len(points)):
        _, indices = neigh.kneighbors([points[i]])  # 正确获取索引
        if len(indices[0]) < 3:
            normals.append(np.array([0, 0, 1]))
            curvatures.append(0)
            continue
        cov = np.cov(points[indices[0]].T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normal /= (np.linalg.norm(normal) + 1e-6)
        normals.append(normal)
        curvatures.append(eigvals[0] / (np.sum(eigvals) + 1e-6))

    clusters = []
    unprocessed = set(range(len(points)))

    while unprocessed:
        seed = next(iter(unprocessed))
        queue = [seed]
        cluster = []
        while queue:
            idx = queue.pop(0)
            if idx not in unprocessed:
                continue
            cluster.append(idx)
            unprocessed.remove(idx)
            _, neighbors = neigh.kneighbors([points[idx]])
            for n in neighbors[0]:
                if n not in unprocessed:
                    continue
                dot_product = np.abs(normals[idx] @ normals[n])
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))
                if (angle < normal_threshold and
                        curvatures[n] < curvature_threshold):
                    queue.append(n)
        if len(cluster) > 50:
            clusters.append(points[cluster])
    return clusters


rg_clusters = region_growing(points)

vehicle_pos = np.array([10, 5, 2])

# ==================== 计算最近距离 ====================
def compute_min_distance(clusters, car_pos=vehicle_pos):
    min_distances = []
    nearest_points = []
    for cluster in clusters:
        tree = KDTree(cluster)
        dist, idx = tree.query(car_pos)
        min_distances.append(dist)
        nearest_points.append(cluster[idx])
    return min_distances, nearest_points


# DBSCAN结果距离
db_distances, db_nearest = compute_min_distance(db_clusters)
# 区域生长结果距离
rg_distances, rg_nearest = compute_min_distance(rg_clusters)

print("区域生长最近距离:", rg_distances)


# ==================== 可视化 ====================
plotter = pv.Plotter()

# 绘制DBSCAN聚类
colors_db = ['#FF6B6B', '#4ECDC4', '#45B7D1',"gray","purple" ]
for i, cluster in enumerate(db_clusters):
    mesh = pv.PolyData(cluster)
    plotter.add_mesh(mesh, point_size=3, color=colors_db[i % len(colors_db)])

    # 绘制区域生长聚类（半透明对比）
    colors_rg = ['#FF99A8', '#88D8B0', '#9AC6FF']
    for i, cluster in enumerate(rg_clusters):
        mesh = pv.PolyData(cluster)
    plotter.add_mesh(mesh, point_size=3, color=colors_rg[i % len(colors_rg)], opacity=0.5)

    # 自车位置和最近点
    plotter.add_mesh(pv.Sphere(0.2, vehicle_pos), color='red')
    all_nearest_points = db_nearest + rg_nearest
    for p in all_nearest_points:
        plotter.add_mesh(pv.Sphere(0.2, p), color='lime', label="Nearest Point")
        # 绘制连接线（修正参数传递方式）
        line_points = np.array([vehicle_pos, p])  # 形状为(2,3)的坐标数组
        line = pv.Line(pointa=line_points[0], pointb=line_points[1])  # 显式指定起点终点
        plotter.add_mesh(line, color='yellow', line_width=2)

    # for p in db_nearest + rg_nearest:
    #     plotter.add_mesh(pv.Sphere(0.2, p), color='green')
    # plotter.add_lines(np.array([vehicle_pos, p]), color='yellow')

    # plotter.add_legend()
    plotter.add_axes(
        xlabel="X", ylabel="Y", zlabel="Z",
        line_width=4, labels_off=False, color='black'
    )
    plotter.show()