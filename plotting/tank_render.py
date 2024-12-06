import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Surface element function for the ellipsoid in spherical coordinates
def surface_element(theta, phi, a, b):
    # Compute the parameterized coordinates
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = a * np.cos(theta)

    # Partial derivatives with respect to theta
    x_theta = a * np.cos(theta) * np.cos(phi)
    y_theta = b * np.cos(theta) * np.sin(phi)
    z_theta = -a * np.sin(theta)

    # Partial derivatives with respect to phi
    x_phi = -a * np.sin(theta) * np.sin(phi)
    y_phi = b * np.sin(theta) * np.cos(phi)
    z_phi = 0

    # Compute the cross product magnitude for the surface area element
    cross_product = np.sqrt((y_theta * z_phi - z_theta * y_phi)**2 +
                            (z_theta * x_phi - x_theta * z_phi)**2 +
                            (x_theta * y_phi - y_theta * x_phi)**2)
    return cross_product

def plot_open_cylinder(radius, length):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    z = np.linspace(0, length, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    ax.plot_surface(x_grid, z_grid, y_grid, color='blue', alpha=0.5, rstride=1, cstride=1)

    return fig  

def plot_ellipsoid(ax, radius, b, y_offset, top=True):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100) if top else np.linspace(np.pi / 2, np.pi, 100)
    u_grid, v_grid = np.meshgrid(u, v)
    x = radius * np.sin(v_grid) * np.cos(u_grid)
    z = radius * np.sin(v_grid) * np.sin(u_grid)
    y = b * np.cos(v_grid) + y_offset

    ax.plot_surface(x, y, z, color='orange', alpha=0.5, rstride=1, cstride=1)

def plot_tank(radius, b, length):
    fig = plot_open_cylinder(radius, length)
    ax = fig.gca(projection='3d')
    plot_ellipsoid(ax, radius, b, length, top=True)
    plot_ellipsoid(ax, radius, b, 0, top=False)
    return fig 

def main():
    pass


if __name__ == "__main__":
    main()