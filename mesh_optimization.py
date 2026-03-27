
# Calculate the angle of a triangle
def compute_triangle_angles(A, B, C):
   
    a = np.linalg.norm(B - C)  # Edge BC
    b = np.linalg.norm(A - C)  # Edge AC
    c = np.linalg.norm(A - B)  # Edge AB
    epsilon = 1e-7  # Prevent floating point errors

    # Calculate the angle using the cosine theorem, with a limit value between [-1,1]
    cos_A = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1 + epsilon, 1 - epsilon)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1 + epsilon, 1 - epsilon)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1 + epsilon, 1 - epsilon)

    angle_A = np.arccos(cos_A)
    angle_B = np.arccos(cos_B)
    angle_C = np.arccos(cos_C)

    return angle_A, angle_B, angle_C

# Calculate the minimum angle of a triangle
def compute_min_angle(A, B, C):
    angles = compute_triangle_angles(A, B, C)
    return min(angles)  

# Calculate the area of a triangle
def triangle_area_3d(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * np.linalg.norm(np.cross(v1, v2))


# Ensure that the point is within the triangular plane
def project_point_to_plane(point, A, normal):
    vector = point - A
    dist = np.dot(vector, normal)
    return point - dist * normal


#Ensure that the point is inside the triangle
def point_in_triangle(pt, A, B, C):
    v0, v1, v2 = C - A, B - A, pt - A
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def constrain_point_to_triangle(pt, A, B, C):
    # Project the point onto the plane where the triangle is located
    normal = compute_normal(A, B, C)
    pt = project_point_to_plane(pt, A, normal)

    # If the point is not within the triangle, project it onto the nearest side
    if not point_in_triangle(pt, A, B, C):
        edges = [(A, B), (B, C), (C, A)]
        projections = [project_point_to_segment(pt, edge[0], edge[1]) for edge in edges]
        distances = [np.linalg.norm(pt - proj) for proj in projections]
        pt = projections[np.argmin(distances)]  # Select the nearest edge projection point

    return pt

def project_point_to_segment(pt, v1, v2):
    # Project points onto line segments
    v = v2 - v1
    t = np.dot(pt - v1, v) / np.dot(v, v)
    t = np.clip(t, 0, 1)  # Limit to line segment range
    return v1 + t * v

# Calculate the normal vector of a triangle
def compute_normal(A, B, C):
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    norm = np.linalg.norm(normal)
    if norm == 0:
        print(f"Warning: Zero normal vector for triangle with vertices {A}, {B}, {C}")
        return np.array([0, 0, 0])  # Return a zero vector or process it as a default normal vector
    return normal / norm
    #return normal / np.linalg.norm(normal)


# Calculate the distance between the insertion point and the vertex
def compute_distances_with_vertices(points, A, B, C):
    distances = []
    for i in range(len(points)):
        # Distance from insertion point to three vertices
        distances.append(np.linalg.norm(points[i] - A))  
        distances.append(np.linalg.norm(points[i] - B))  
        distances.append(np.linalg.norm(points[i] - C))  
    return distances

# Calculate the distance between insertion points (for uniformity optimization)
def compute_inserted_point_distances(points):
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
    return distances

# Calculate all relevant distances: distance between insertion point and vertex+distance between insertion points
def compute_all_distances(inserted_points, A, B, C):
    vertex_distances = compute_distances_with_vertices(inserted_points, A, B, C)
    point_distances = compute_inserted_point_distances(inserted_points)
    return vertex_distances + point_distances


def compute_ideal_distance(inserted_points, A, B, C):
    all_distances = compute_all_distances(inserted_points, A, B, C)
    return np.mean(all_distances)  

# Calculate Shape Energy
def shape_energy(triangles, points, theta_ideal=60.0):
    
    theta_ideal_rad = np.radians(theta_ideal)
    
    E_shape = 0.0
    for t in triangles:
        A = points[t[0]]
        B = points[t[1]]
        C = points[t[2]]
        
       
        theta_min = compute_min_angle(A, B, C)
        
        
        E_shape += (theta_min - theta_ideal_rad) ** 2
    
    return E_shape

# Merge energy function: area optimization+distance optimization+shape optimization
def total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1=1.0, lambda_2=1.0, lambda_3=0.5):
    
    E_area = sum(abs(triangle_area_3d(points[t[0]], points[t[1]], points[t[2]]) - avg_area) for t in triangles)

   
    all_distances = compute_all_distances(inserted_points, A, B, C)
    #E_distance = sum((d - d_ideal) ** 2 for d in all_distances)
    E_distance = sum((d - d_ideal) ** 2 for d in all_distances) / len(all_distances)  

    
    E_shape = shape_energy(triangles, points)

    
    E_total = lambda_1 * E_area + lambda_2 * E_distance + lambda_3  * E_shape
    return E_total

# Read triangular mesh from PLY file
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
  
    vertices *= 100  

    
    scaled_mesh = o3d.geometry.TriangleMesh()
    scaled_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    scaled_mesh.triangles = o3d.utility.Vector3iVector(triangles)

   
    scaled_mesh.compute_vertex_normals()

    return vertices, triangles,scaled_mesh

def compute_triangle_area(vertices, triangle):
    """Calculate the area of a triangle"""
    v0, v1, v2 = vertices[triangle]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def compute_triangle_curvature(mesh, triangle):
    """Simple estimation of the curvature of a triangle based on the difference in vertex normal vectors"""
    v0, v1, v2 = triangle
    n0, n1, n2 = mesh.vertex_normals[v0], mesh.vertex_normals[v1], mesh.vertex_normals[v2]
    curvature = (np.linalg.norm(n0 - n1) + np.linalg.norm(n1 - n2) + np.linalg.norm(n2 - n0)) / 3
    return curvature

def calculate_points_to_insert(curvature, area, total_curvature, total_area, total_points):
    """Calculate the number of points to be inserted based on curvature and area"""
    return max(1, round((curvature * area / (total_curvature * total_area)) * total_points * 1000))

def generate_random_points_in_triangle(v0, v1, v2, n_points):
    """Generate uniformly distributed random points within a triangle"""
    points = []
    for _ in range(n_points):
        s, t = np.random.uniform(0, 1), np.random.uniform(0, 1)
        if s + t > 1:  
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

   
    total_curvature = 0
    total_area = 0
    triangle_info = []

   
    for triangle in triangles:
        area = compute_triangle_area(vertices, triangle)
        curvature = compute_triangle_curvature(mesh, triangle)
        triangle_info.append((triangle, area, curvature))
        total_area += area
        total_curvature += curvature

   
    all_optimized_points = []
    all_points = 0

    for idx, (triangle, area, curvature) in enumerate(triangle_info):
       
        points_to_insert = calculate_points_to_insert(curvature, area, total_curvature, total_area, total_points)
        all_points += points_to_insert
        print(points_to_insert)
        v0, v1, v2 = vertices[triangle]
        inserted_points = generate_random_points_in_triangle(v0, v1, v2, points_to_insert)

       
        optimized_points = optimize_inserted_points_with_distance(
            [v0, v1, v2], inserted_points, iterations=iterations, learning_rate=learning_rate
        )

      
        all_optimized_points.extend(optimized_points)

      
        print(f"Processed triangle {idx + 1}/{len(triangle_info)}: Points Inserted = {points_to_insert}")

    print(all_points)

    return np.array(all_optimized_points)

# Optimize insertion point
def optimize_inserted_points_with_distance(original_triangle, inserted_points, iterations=100, learning_rate=0.01, lambda_1=1.0, lambda_2=1.0, lambda_3=0.5):
    A, B, C = original_triangle
    points = np.vstack([A, B, C, inserted_points])  
    normal = compute_normal(A, B, C)

    if np.isnan(points).any():
        print("Found NaN value, cleaning data ...")
        points = points[~np.isnan(points).any(axis=1)]

    
    initial_triangulation = Delaunay(points[:, :2])

    
    d_ideal = compute_ideal_distance(inserted_points, A, B, C)

    
    previous_energy = float('inf')  

    for it in range(iterations):
        triangulation = Delaunay(points[:, :2])
        triangles = triangulation.simplices

       
        avg_area = np.mean([triangle_area_3d(points[t[0]], points[t[1]], points[t[2]]) for t in triangles])
        energy = total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1, lambda_2, lambda_3)

        
        if abs(previous_energy - energy) < 1e-6:  
            print(f"Converged at iteration {it + 1} with energy: {energy:.6f}")
            break
        previous_energy = energy  

        gradients = np.zeros_like(inserted_points)

       
        for i, p in enumerate(inserted_points):
            for dim in range(3): 
                original = p[dim]

               
                p[dim] = original + 1e-5
                points[-len(inserted_points):] = inserted_points
                energy_plus = total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1, lambda_2, lambda_3)

               
                p[dim] = original - 1e-5
                points[-len(inserted_points):] = inserted_points
                energy_minus = total_energy_with_shape(points, inserted_points, triangles, avg_area, d_ideal, A, B, C, lambda_1, lambda_2, lambda_3)

               
                gradients[i, dim] = (energy_plus - energy_minus) / (2 * 1e-5)
                p[dim] = original

      
        inserted_points -= learning_rate * gradients

      
        for i in range(len(inserted_points)):
            inserted_points[i] = constrain_point_to_triangle(inserted_points[i], A, B, C)
            if not point_in_triangle(inserted_points[i], A, B, C):
                inserted_points[i] = np.clip(inserted_points[i], A, C)

        points[-len(inserted_points):] = inserted_points

        print(f"Iteration {it + 1}/{iterations}, Energy: {energy:.6f}")

    
    final_triangulation = Delaunay(points[:, :2])
   
    return inserted_points

