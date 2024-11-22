import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import numpy as np

def plot_ellipsoid_area(points1, points2):
    """
    Plot two sets of points on the surface of an ellipsoid in a 3D scatter plot.

    Parameters:
        points1 (np.ndarray): Array of shape (num_points, 3) containing points on the entire ellipsoid.
        points2 (np.ndarray): Array of shape (num_points, 3) containing points on the ellipsoid up to a specific height.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Compute and plot the convex hull
    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)
    for simplex in hull1.simplices:
        triangle = points1[simplex]
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='gray', alpha=0.1, edgecolor='gray')
    for simplex in hull2.simplices:
        triangle = points2[simplex]
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='blue', alpha=0.5, edgecolor='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Set the aspect ratio to be equal
    max_range = np.array([points1[:, 0].max() - points1[:, 0].min(),
                          points1[:, 1].max() - points1[:, 1].min(),
                          points1[:, 2].max() - points1[:, 2].min()]).max() / 2.0

    mid_x = (points1[:, 0].max() + points1[:, 0].min()) * 0.5
    mid_y = (points1[:, 1].max() + points1[:, 1].min()) * 0.5
    mid_z = (points1[:, 2].max() + points1[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()