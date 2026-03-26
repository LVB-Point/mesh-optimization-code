import numpy as np
from scipy.spatial import Delaunay
import open3d as o3d

# 计算三角形的角度
def compute_triangle_angles(A, B, C):
    # 计算边的长度
    a = np.linalg.norm(B - C)  # 边BC
    b = np.linalg.norm(A - C)  # 边AC
    c = np.linalg.norm(A - B)  # 边AB
    epsilon = 1e-7  # 防止浮点数误差

    # 使用余弦定理计算角度，限制值在 [-1, 1] 之间
    cos_A = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1 + epsilon, 1 - epsilon)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1 + epsilon, 1 - epsilon)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1 + epsilon, 1 - epsilon)

    angle_A = np.arccos(cos_A)
    angle_B = np.arccos(cos_B)
    angle_C = np.arccos(cos_C)

    return angle_A, angle_B, angle_C

# 计算三角形的最小角度
def compute_min_angle(A, B, C):
    angles = compute_triangle_angles(A, B, C)
    return min(angles)  # 返回最小角度

# 计算三角形面积
def triangle_area_3d(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * np.linalg.norm(np.cross(v1, v2))


# 确保点在三角形平面内
def project_point_to_plane(point, A, normal):
    vector = point - A
    dist = np.dot(vector, normal)
    return point - dist * normal


#确保点在三角形内部
def point_in_triangle(pt, A, B, C):
    v0, v1, v2 = C - A, B - A, pt - A
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def constrain_point_to_triangle(pt, A, B, C):
    # 投影点到三角形所在平面
    normal = compute_normal(A, B, C)
    pt = project_point_to_plane(pt, A, normal)

    # 如果点不在三角形内，将其投影到最近的边上
    if not point_in_triangle(pt, A, B, C):
        edges = [(A, B), (B, C), (C, A)]
        projections = [project_point_to_segment(pt, edge[0], edge[1]) for edge in edges]
        distances = [np.linalg.norm(pt - proj) for proj in projections]
        pt = projections[np.argmin(distances)]  # 选择最近的边投影点

    return pt

def project_point_to_segment(pt, v1, v2):
    # 将点投影到线段上
    v = v2 - v1
    t = np.dot(pt - v1, v) / np.dot(v, v)
    t = np.clip(t, 0, 1)  # 限制到线段范围
    return v1 + t * v

# 计算三角形法向量
def compute_normal(A, B, C):
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    norm = np.linalg.norm(normal)
    if norm == 0:
        print(f"Warning: Zero normal vector for triangle with vertices {A}, {B}, {C}")
        return np.array([0, 0, 0])  # 返回一个零向量，或者处理为一个默认法向量
    return normal / norm
    #return normal / np.linalg.norm(normal)


# 可视化点云和三角形
def visualize_open3d(points, triangles, title):
    triangle_lines = []
    for t in triangles:
        triangle_lines.append([t[0], t[1]])
        triangle_lines.append([t[1], t[2]])
        triangle_lines.append([t[2], t[0]])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(triangle_lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(triangle_lines))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(points))  # 红色点

    print(f"Displaying: {title}")
    o3d.visualization.draw_geometries([line_set, point_cloud])

# 计算插入点与顶点之间的距离
def compute_distances_with_vertices(points, A, B, C):
    distances = []
    for i in range(len(points)):
        # 插入点到三个顶点的距离
        distances.append(np.linalg.norm(points[i] - A))  # 插入点到顶点A的距离
        distances.append(np.linalg.norm(points[i] - B))  # 插入点到顶点B的距离
        distances.append(np.linalg.norm(points[i] - C))  # 插入点到顶点C的距离
    return distances

# 计算插入点之间的距离（用于均匀性优化）
def compute_inserted_point_distances(points):
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
    return distances

# 计算所有相关距离：插入点与顶点之间的距离 + 插入点之间的距离
def compute_all_distances(inserted_points, A, B, C):
    vertex_distances = compute_distances_with_vertices(inserted_points, A, B, C)
    point_distances = compute_inserted_point_distances(inserted_points)
    return vertex_distances + point_distances

# 计算目标距离d_ideal
def compute_ideal_distance(inserted_points, A, B, C):
    all_distances = compute_all_distances(inserted_points, A, B, C)
    return np.mean(all_distances)  # 使用所有距离的平均值作为目标距离

# 计算形状能量
def shape_energy(triangles, points, theta_ideal=60.0):
    # 将理想的最小角度转换为弧度
    theta_ideal_rad = np.radians(theta_ideal)
    
    E_shape = 0.0
    for t in triangles:
        A = points[t[0]]
        B = points[t[1]]
        C = points[t[2]]
        
        # 计算当前三角形的最小角度
        theta_min = compute_min_angle(A, B, C)
        
        # 计算形状能量
        E_shape += (theta_min - theta_ideal_rad) ** 2
    
    return E_shape

# 合并能量函数：面积优化 + 距离优化 + 形状优化
def total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1=1.0, lambda_2=1.0, lambda_3=0.5):
    # 面积优化能量
    E_area = sum(abs(triangle_area_3d(points[t[0]], points[t[1]], points[t[2]]) - avg_area) for t in triangles)

    # 统一距离优化能量
    all_distances = compute_all_distances(inserted_points, A, B, C)
    #E_distance = sum((d - d_ideal) ** 2 for d in all_distances)
    E_distance = sum((d - d_ideal) ** 2 for d in all_distances) / len(all_distances)  # 归一化

    # 计算形状能量
    E_shape = shape_energy(triangles, points)

    # 总能量（加权合并）
    E_total = lambda_1 * E_area + lambda_2 * E_distance + lambda_3  * E_shape
    return E_total

# 从 PLY 文件读取三角网格
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    # 放大坐标
    vertices *= 100  # 将坐标放大100倍

    # 使用放大的顶点重新构建 mesh
    scaled_mesh = o3d.geometry.TriangleMesh()
    scaled_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    scaled_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 重新计算法向量
    scaled_mesh.compute_vertex_normals()

    return vertices, triangles,scaled_mesh

def compute_triangle_area(vertices, triangle):
    """计算三角形面积"""
    v0, v1, v2 = vertices[triangle]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def compute_triangle_curvature(mesh, triangle):
    """简单估计三角形的曲率，基于顶点法向量差异"""
    v0, v1, v2 = triangle
    n0, n1, n2 = mesh.vertex_normals[v0], mesh.vertex_normals[v1], mesh.vertex_normals[v2]
    curvature = (np.linalg.norm(n0 - n1) + np.linalg.norm(n1 - n2) + np.linalg.norm(n2 - n0)) / 3
    return curvature

def calculate_points_to_insert(curvature, area, total_curvature, total_area, total_points):
    """根据曲率和面积计算需要插入的点数"""
    return max(1, round((curvature * area / (total_curvature * total_area)) * total_points * 1000))

def generate_random_points_in_triangle(v0, v1, v2, n_points):
    """在三角形内生成均匀分布的随机点"""
    points = []
    for _ in range(n_points):
        s, t = np.random.uniform(0, 1), np.random.uniform(0, 1)
        if s + t > 1:  # 保证点在三角形内
            s, t = 1 - s, 1 - t
        a = 1 - np.sqrt(t)
        b = (1 - s) * np.sqrt(t)
        c = s * np.sqrt(t)
        point = a * v0 + b * v1 + c * v2
        points.append(point)
    return points

def process_mesh_with_curvature_and_area(mesh, total_points=3000, iterations=100, learning_rate=0.01):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # 初始化总曲率和面积
    total_curvature = 0
    total_area = 0
    triangle_info = []

    # 计算每个三角形的面积和曲率
    for triangle in triangles:
        area = compute_triangle_area(vertices, triangle)
        curvature = compute_triangle_curvature(mesh, triangle)
        triangle_info.append((triangle, area, curvature))
        total_area += area
        total_curvature += curvature

    # 插入点存储
    all_optimized_points = []
    all_points = 0

    for idx, (triangle, area, curvature) in enumerate(triangle_info):
        # 根据曲率和面积计算需要插入的点数
        points_to_insert = calculate_points_to_insert(curvature, area, total_curvature, total_area, total_points)
        all_points += points_to_insert
        print(points_to_insert)
        v0, v1, v2 = vertices[triangle]
        inserted_points = generate_random_points_in_triangle(v0, v1, v2, points_to_insert)

        # 调用优化函数
        optimized_points = optimize_inserted_points_with_distance(
            [v0, v1, v2], inserted_points, iterations=iterations, learning_rate=learning_rate
        )

        # 保存优化结果
        all_optimized_points.extend(optimized_points)

        # 打印进度
        print(f"Processed triangle {idx + 1}/{len(triangle_info)}: Points Inserted = {points_to_insert}")

    print(all_points)

    return np.array(all_optimized_points)

# 优化插入点
def optimize_inserted_points_with_distance(original_triangle, inserted_points, iterations=100, learning_rate=0.01, lambda_1=1.0, lambda_2=1.0, lambda_3=0.5):
    A, B, C = original_triangle
    points = np.vstack([A, B, C, inserted_points])  # 保持原始三角形顶点不变
    normal = compute_normal(A, B, C)

    if np.isnan(points).any():
        print("发现 NaN 值，正在清理数据...")
        points = points[~np.isnan(points).any(axis=1)]

    # 初始三角剖分
    initial_triangulation = Delaunay(points[:, :2])
    #visualize_open3d(points, initial_triangulation.simplices, "Initial (Unoptimized) Triangulation")

    # 计算理想距离：统一插入点到顶点和插入点之间的目标距离
    d_ideal = compute_ideal_distance(inserted_points, A, B, C)

    # 初始化能量值
    previous_energy = float('inf')  # 用一个大的初始值

    for it in range(iterations):
        triangulation = Delaunay(points[:, :2])
        triangles = triangulation.simplices

        # 计算每一轮的总能量（包括形状能量）
        avg_area = np.mean([triangle_area_3d(points[t[0]], points[t[1]], points[t[2]]) for t in triangles])
        energy = total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1, lambda_2, lambda_3)

         # 提前停止条件
        if abs(previous_energy - energy) < 1e-6:  # 如果能量变化小于阈值
            print(f"Converged at iteration {it + 1} with energy: {energy:.6f}")
            break
        previous_energy = energy  # 更新上一轮能量值

        gradients = np.zeros_like(inserted_points)

        # 计算能量和梯度
        for i, p in enumerate(inserted_points):
            for dim in range(3): 
                original = p[dim]

                # 正向梯度计算
                p[dim] = original + 1e-5
                points[-len(inserted_points):] = inserted_points
                energy_plus = total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1, lambda_2, lambda_3)

                # 反向梯度计算
                p[dim] = original - 1e-5
                points[-len(inserted_points):] = inserted_points
                energy_minus = total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1, lambda_2, lambda_3)

                # 梯度值
                gradients[i, dim] = (energy_plus - energy_minus) / (2 * 1e-5)
                p[dim] = original

        # 更新插入点
        inserted_points -= learning_rate * gradients

        # 投影到平面并检查是否在三角形内
        for i in range(len(inserted_points)):
            inserted_points[i] = constrain_point_to_triangle(inserted_points[i], A, B, C)
            if not point_in_triangle(inserted_points[i], A, B, C):
                inserted_points[i] = np.clip(inserted_points[i], A, C)

        points[-len(inserted_points):] = inserted_points

        print(f"Iteration {it + 1}/{iterations}, Energy: {energy:.6f}")

    # 最终三角剖分
    final_triangulation = Delaunay(points[:, :2])
    #visualize_open3d(points, final_triangulation.simplices, "Optimized Triangulation")
    return inserted_points

# 示例运行
if __name__ == "__main__":


    # 指定 PLY 文件路径
    file_path = "F:\\thesis\\data\\newmethod_test\\hand\\hand.ply"
    
    # 读取网格数据
    vertices, triangles, mesh = load_mesh(file_path)

    # 遍历三角形并优化插入点
    optimized_points = process_mesh_with_curvature_and_area(mesh, total_points=3000,)

    all_inserted_points = []
    all_inserted_points.extend(optimized_points)
    all_point = np.vstack([vertices,all_inserted_points])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_point)

    o3d.io.write_point_cloud("F:\\thesis\\data\\newmethod_test\\hand\\hand_ad.ply", point_cloud)
    print("点云已保存")