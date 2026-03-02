import numpy as np
import os

os.chdir(r"d:\University Files\Semester-7\IN4640 - Machine Vision\Assignment-2\Materials")

D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
X_cols = D[:, :3]
Y_cols = D[:, 3:]

# (a) TLS on first line (x1, y1)
x1 = X_cols[:, 0]
y1 = Y_cols[:, 0]

points1 = np.column_stack((x1, y1))
centroid1 = np.mean(points1, axis=0)
centered_points1 = points1 - centroid1
U, S, Vt = np.linalg.svd(centered_points1)
normal1 = Vt[-1, :] # The last row of Vt is the normal vector (a, b)
a, b = normal1
# ax + by + d = 0 -> d = -(ax_mean + by_mean)
d = -np.dot(normal1, centroid1)

print("Part A (TLS on Line 1):")
print(f"Equation: {a:.4f}x + {b:.4f}y + d = 0, where d = {d:.4f}")
if b != 0:
    m = -a / b
    c = -d / b
    print(f"y = {m:.4f}x + {c:.4f}")

# (b) RANSAC
print("\nPart B (RANSAC):")
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()
points_all = np.column_stack((X_all, Y_all))

def ransac_line_fit(points, n_iterations=1000, threshold=0.1):
    best_line = None
    best_inliers = []
    
    n_points = len(points)
    
    for _ in range(n_iterations):
        # Sample 2 distinct points
        idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[idx]
        
        # Line equation ax + by + d = 0
        diff = p2 - p1
        if np.linalg.norm(diff) < 1e-6:
            continue
            
        a, b = -diff[1], diff[0]
        norm = np.hypot(a, b)
        a, b = a / norm, b / norm
        d = -(a * p1[0] + b * p1[1])
        
        # Calculate distances of all points to the line
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + d)
        
        inlier_idx = np.where(distances < threshold)[0]
        
        if len(inlier_idx) > len(best_inliers):
            best_inliers = inlier_idx
            best_line = (a, b, d)
            
    return best_line, best_inliers

import copy
points_remaining = copy.deepcopy(points_all)
lines = []

for i in range(3):
    line, inliers = ransac_line_fit(points_remaining, n_iterations=2000, threshold=0.2)
    # Refine with TLS? No, just finding a line. Ask for TLS on inliers
    inlier_points = points_remaining[inliers]
    
    # TLS on inliers to refine
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid
    U, S, Vt = np.linalg.svd(centered)
    a, b = Vt[-1, :]
    d = -np.dot((a,b), centroid)
    
    lines.append((a, b, d, len(inliers)))
    
    # Remove inliers
    mask = np.ones(len(points_remaining), dtype=bool)
    mask[inliers] = False
    points_remaining = points_remaining[mask]
    
    print(f"Line {i+1}: {a:.4f}x + {b:.4f}y + ({d:.4f}) = 0 (Inliers: {len(inliers)})")
    if b != 0:
        print(f"  y = {-a/b:.4f}x + {-d/b:.4f}")

# Plotting
import matplotlib.pyplot as plt

# Part A Plot
plt.figure()
plt.scatter(x1, y1, color='blue', label='Data points (first line)')
if normal1[1] != 0:
    m = -normal1[0] / normal1[1]
    c = -(-np.dot(normal1, centroid1)) / normal1[1]
    x_vals = np.array([np.min(x1), np.max(x1)])
    y_vals = m * x_vals + c
    plt.plot(x_vals, y_vals, color='red', label='TLS Fit')
plt.title('TLS Fit on First Line Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('q1_part_a.png')
plt.close()

# Part B Plot
plt.figure()
plt.scatter(X_all, Y_all, color='gray', alpha=0.5, label='All Data points')
colors = ['red', 'green', 'blue']
for i, line in enumerate(lines):
    a, b, d, inliers_count = line
    if b != 0:
        m = -a / b
        c = -d / b
        x_vals = np.array([np.min(X_all), np.max(X_all)])
        y_vals = m * x_vals + c
        plt.plot(x_vals, y_vals, color=colors[i], label=f'RANSAC Line {i+1}')
plt.title('RANSAC Fit on All Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('q1_part_b.png')
plt.close()

print("Graphs saved as q1_part_a.png and q1_part_b.png")
