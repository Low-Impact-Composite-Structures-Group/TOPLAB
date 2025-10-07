"""
Phi Tank Visualization Script

This script visualizes four cylindrical tanks with hemispherical domes as a function
of their L/R ratio (phi). The tanks are shown for phi = 0, 1, 2, 3 where:
- phi = L/R (length of cylindrical section / radius)
- phi = 0 corresponds to a sphere
- R = 1 for all tanks (normalized)

Uses the Delft color palette for consistent styling.

Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the plotting directory to path to import style modules
plotting_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plotting')
sys.path.append(plotting_dir)


class MyCycler:
    def __init__(self):
        self.colors = ["#00A6D6", "#E03C31", "#009B77", "#A50034"]  # Delft colors

def set_font():
    pass

FIGURE_WIDTH = 10.0
FIGURE_HEIGHT = 8.8
CM2INCH = 0.393701


class TankGeometry:
    """Class to define tank geometry based on phi ratio"""

    def __init__(self, phi: float, radius: float = 1.0):
        """
        Initialize tank geometry

        Args:
            phi: L/R ratio where L is cylindrical length, R is radius
            radius: Tank radius (default = 1.0 for normalized visualization)
        """
        self.phi = phi
        self.radius = radius
        self.cylinder_length = phi * radius
        self.total_length = self.cylinder_length + 2 * radius  # Include both dome radii

    def get_cylinder_surface(self, n_points: int = 50):
        """Generate cylinder surface coordinates"""
        if self.phi == 0:  # Sphere case
            return None, None, None

        theta = np.linspace(0, 2 * np.pi, n_points)
        z = np.linspace(0, self.cylinder_length, n_points)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x_cyl = self.radius * np.cos(theta_grid)
        y_cyl = self.radius * np.sin(theta_grid)
        z_cyl = z_grid + self.radius  # Offset by radius (bottom dome)

        return x_cyl, y_cyl, z_cyl

    def get_dome_surface(self, top_dome: bool = True, n_points: int = 50):
        """Generate hemispherical dome surface coordinates"""
        u = np.linspace(0, 2 * np.pi, n_points)

        if self.phi == 0:  # Full sphere
            v = np.linspace(0, np.pi, n_points)
        else:  # Hemisphere
            v = np.linspace(0, np.pi/2, n_points) if top_dome else np.linspace(np.pi/2, np.pi, n_points)

        u_grid, v_grid = np.meshgrid(u, v)

        x_dome = self.radius * np.sin(v_grid) * np.cos(u_grid)
        y_dome = self.radius * np.sin(v_grid) * np.sin(u_grid)
        z_dome = self.radius * np.cos(v_grid)

        if self.phi > 0:  # Offset domes for cylindrical tanks
            if top_dome:
                z_dome = z_dome + self.radius + self.cylinder_length
            else:
                z_dome = z_dome + self.radius
        else:  # For sphere (phi=0), center at origin
            z_dome = z_dome + self.radius

        return x_dome, y_dome, z_dome


def create_tank_subplot(ax, tank_geo: TankGeometry, color: str, title: str):
    """Create a 3D subplot for a single tank"""

    # Plot cylinder (if phi > 0)
    if tank_geo.phi > 0:
        x_cyl, y_cyl, z_cyl = tank_geo.get_cylinder_surface()
        ax.plot_surface(x_cyl, y_cyl, z_cyl, color=color, alpha=0.7,
                       rstride=2, cstride=2, linewidth=0.5, edgecolor='black', linewidths=0.1)

        # Plot top dome
        x_top, y_top, z_top = tank_geo.get_dome_surface(top_dome=True)
        ax.plot_surface(x_top, y_top, z_top, color=color, alpha=0.7,
                       rstride=2, cstride=2, linewidth=0.5, edgecolor='black', linewidths=0.1)

        # Plot bottom dome
        x_bot, y_bot, z_bot = tank_geo.get_dome_surface(top_dome=False)
        ax.plot_surface(x_bot, y_bot, z_bot, color=color, alpha=0.7,
                       rstride=2, cstride=2, linewidth=0.5, edgecolor='black', linewidths=0.1)
    else:
        # Plot full sphere for phi = 0
        x_sphere, y_sphere, z_sphere = tank_geo.get_dome_surface()
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.7,
                       rstride=2, cstride=2, linewidth=0.5, edgecolor='black', linewidths=0.1)

    # Set equal aspect ratio and limits
    max_dim = max(tank_geo.radius * 1.2, tank_geo.total_length * 0.6)
    ax.set_xlim([-max_dim, max_dim])
    ax.set_ylim([-max_dim, max_dim])
    ax.set_zlim([0, tank_geo.total_length])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\nφ = L/R = {tank_geo.phi}', fontsize=10, pad=10)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)


def create_phi_comparison_plot():
    """Create a 2x2 subplot showing tanks for phi = 0, 1, 2, 3"""

    # Set up the plotting style
    set_font()
    colors = MyCycler().colors[:4]  # Use first 4 Delft colors
    phi_values = [0, 1, 2, 3]
    tank_names = ['Sphere', 'Short Cylinder', 'Medium Cylinder', 'Long Cylinder']

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(12, 10))

    for i, (phi, name, color) in enumerate(zip(phi_values, tank_names, colors)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        tank_geo = TankGeometry(phi=phi, radius=1.0)
        create_tank_subplot(ax, tank_geo, color, name)

    plt.tight_layout()
    plt.suptitle('Hydrogen Tank Geometry as Function of φ = L/R Ratio',
                 fontsize=14, y=0.95)

    return fig


def create_side_view_comparison():
    """Create a 2D side view comparison of all tank geometries"""

    set_font()
    colors = MyCycler().colors[:4]
    phi_values = [0, 1, 2, 3]

    # Make figure wider than tall (landscape orientation)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    y_offset = 0
    spacing = 2.5

    for phi, color in zip(phi_values, colors):
        tank_geo = TankGeometry(phi=phi, radius=1.0)

        # Draw tank outline
        if phi == 0:  # Sphere
            circle = plt.Circle((0, y_offset), tank_geo.radius,
                              fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
        else:  # Cylinder with domes
            # Draw only horizontal lines of cylinder (no vertical sides)
            rect_width = tank_geo.cylinder_length
            # Top horizontal line
            ax.plot([-rect_width/2, rect_width/2],
                   [y_offset + tank_geo.radius, y_offset + tank_geo.radius],
                   color=color, linewidth=2, linestyle='-', marker='')
            # Bottom horizontal line
            ax.plot([-rect_width/2, rect_width/2],
                   [y_offset - tank_geo.radius, y_offset - tank_geo.radius],
                   color=color, linewidth=2, linestyle='-', marker='')

            # Draw only curved parts of semicircles (no straight diameter)
            # Left semicircle - draw as arc only
            theta_left = np.linspace(np.pi/2, 3*np.pi/2, 100)
            x_left = -rect_width/2 + tank_geo.radius * np.cos(theta_left)
            y_left = y_offset + tank_geo.radius * np.sin(theta_left)
            ax.plot(x_left, y_left, color=color, linewidth=2, linestyle='-', marker='')

            # Right semicircle - draw as arc only
            theta_right = np.linspace(3*np.pi/2, 5*np.pi/2, 100)
            x_right = rect_width/2 + tank_geo.radius * np.cos(theta_right)
            y_right = y_offset + tank_geo.radius * np.sin(theta_right)
            ax.plot(x_right, y_right, color=color, linewidth=2, linestyle='-', marker='')

        # Add to legend with solid line only (no markers)
        ax.plot([], [], color=color, linewidth=2, linestyle='-', marker='', label=f'φ = {phi}')

        y_offset += spacing

    # Set equal aspect ratio and limits for vertical layout
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.5, 9.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    # Place legend on plot but out of the way with 3D shadow effect
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                      shadow=True, framealpha=0.9, edgecolor='black')
    # Additional styling for 3D effect
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.2)
    ax.set_xlabel('Normalized Length')
    # ax.set_ylabel('Tank Position')
    # Set y-ticks for grid but hide the labels
    y_ticks = np.arange(-1, 10, 1)  # Y-ticks every 1 unit from -1 to 9
    ax.set_yticks(y_ticks)
    x_ticks = np.arange(-4, 4, 1)  # X-ticks every 1 unit from -4 to 4
    ax.set_xticks(x_ticks)
    ax.set_yticklabels([])  # Hide the labels but keep the tick positions for grid
    # Remove title as requested

    return fig


def show_figures():
    """Generate and display both visualization figures without saving"""

    print("Generating tank phi ratio visualizations...")

    # Create and display 3D comparison plot
    fig_3d = create_phi_comparison_plot()

    # Create and display 2D side view comparison
    fig_2d = create_side_view_comparison()

    print("✓ Figures generated and ready to display")

    return fig_3d, fig_2d


def visualize_single_tank(phi: float, radius: float = 1.0, color: str = None,
                          title: str = None, show_plot: bool = True):
    """
    Create a 3D visualization of a single tank with given phi ratio

    Args:
        phi: L/R ratio (0 = sphere, >0 = cylinder with domes)
        radius: Tank radius (default = 1.0)
        color: Tank color (default uses Delft blue)
        title: Plot title (default based on phi value)
        show_plot: Whether to display the plot (default True)

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if color is None:
        color = "#00A6D6"  # Delft Blue

    if title is None:
        if phi == 0:
            title = f"Sphere (φ = {phi})"
        else:
            title = f"Cylindrical Tank (φ = {phi})"

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create and plot tank geometry
    tank_geo = TankGeometry(phi=phi, radius=radius)
    create_tank_subplot(ax, tank_geo, color, title)

    if show_plot:
        plt.show()

    return fig


def get_tank_volume(phi: float, radius: float = 1.0):
    """
    Calculate the volume of a tank with given phi ratio

    Args:
        phi: L/R ratio
        radius: Tank radius

    Returns:
        float: Tank volume
    """
    if phi == 0:
        # Sphere volume: (4/3) * π * r³
        return (4/3) * np.pi * radius**3
    else:
        # Cylinder volume + 2 hemisphere volumes
        cylinder_vol = np.pi * radius**2 * (phi * radius)  # πr²L where L = φr
        sphere_vol = (4/3) * np.pi * radius**3  # Two hemispheres = one sphere
        return cylinder_vol + sphere_vol


def get_tank_surface_area(phi: float, radius: float = 1.0):
    """
    Calculate the surface area of a tank with given phi ratio

    Args:
        phi: L/R ratio
        radius: Tank radius

    Returns:
        float: Tank surface area
    """
    if phi == 0:
        # Sphere surface area: 4πr²
        return 4 * np.pi * radius**2
    else:
        # Cylinder surface + 2 hemisphere surfaces
        cylinder_area = 2 * np.pi * radius * (phi * radius)  # 2πrL where L = φr
        sphere_area = 4 * np.pi * radius**2  # Two hemispheres = one sphere surface
        return cylinder_area + sphere_area


def main():
    """Main function to demonstrate the tank phi visualization"""

    print("Hydrogen Tank φ (L/R) Ratio Visualization")
    print("=" * 45)
    print("φ = 0: Sphere")
    print("φ = 1: Short cylindrical tank (L = R)")
    print("φ = 2: Medium cylindrical tank (L = 2R)")
    print("φ = 3: Long cylindrical tank (L = 3R)")
    print()

    # Display volume and surface area information
    phi_values = [0, 1, 2, 3]
    print("Tank Properties (R = 1):")
    print("-" * 40)
    for phi in phi_values:
        vol = get_tank_volume(phi)
        area = get_tank_surface_area(phi)
        vol_ratio = vol / get_tank_volume(0)  # Relative to sphere
        area_ratio = area / get_tank_surface_area(0)  # Relative to sphere
        print(f"φ = {phi}: Volume = {vol:.3f} ({vol_ratio:.2f}×), Surface = {area:.3f} ({area_ratio:.2f}×)")
    print()

    # Generate and display figures
    fig_3d, fig_2d = show_figures()

    # Display the plots
    plt.show()

    return fig_3d, fig_2d


if __name__ == "__main__":
    main()
