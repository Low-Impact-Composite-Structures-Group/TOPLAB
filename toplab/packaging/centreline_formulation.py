import numpy as np
import plotly.graph_objects as go


class PiecewiseAxisymmetricVolume:
    """Cylinder followed by two oblique frusta.

    Generalized coordinates:
        s     : global axial coordinate
        eta   : normalized radial coordinate, 0 <= eta <= 1
        theta : circumferential angle [rad]

    Cartesian mapping:
        x = eta * R(s) * cos(theta)
        y = y_c(s) + eta * R(s) * sin(theta)
        z = s

        psi is the angle of the frustum with respect to the axis of the cylinder (about the y-axis).
    """

    def __init__(
        self,
        D1,
        D2,
        D3,
        L1,
        L2,
        L3,
        psi_1,
        psi_2,
    ):
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3

        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        self.psi_1 = psi_1
        self.psi_2 = psi_2

        self.R1 = D1 / 2
        self.R2 = D2 / 2
        self.R3 = D3 / 2

        self.L_total = L1 + L2 + L3

        self.s1 = L1
        self.s2 = L1 + L2
        self.s3 = self.L_total

        self.y_interface_2 = L2 * np.tan(psi_1)

    # ========================================================
    # Geometry
    # ========================================================

    def radius(self, s):
        """Return allowable cross-sectional radius R(s)."""

        if not 0 <= s <= self.L_total:
            raise ValueError(
                f"s = {s} lies outside [0, {self.L_total}]."
            )

        # Cylinder
        if s <= self.L1:
            return self.R1

        # Frustum 1
        elif s <= self.L1 + self.L2:
            s_local = s - self.L1

            return (
                self.R1
                + (self.R2 - self.R1)
                * s_local
                / self.L2
            )

        # Frustum 2
        else:
            s_local = s - (self.L1 + self.L2)

            return (
                self.R2
                + (self.R3 - self.R2)
                * s_local
                / self.L3
            )

    def centerline(self, s):
        """Return Cartesian coordinates of the centreline at s."""

        if not 0 <= s <= self.L_total:
            raise ValueError(
                f"s = {s} lies outside [0, {self.L_total}]."
            )

        # Cylinder
        if s <= self.L1:
            y_c = 0.0

        # Frustum 1
        elif s <= self.L1 + self.L2:
            s_local = s - self.L1

            y_c = (
                s_local
                * np.tan(self.psi_1)
            )

        # Frustum 2
        else:
            s_local = s - (self.L1 + self.L2)

            y_c = (
                self.y_interface_2
                + s_local
                * np.tan(self.psi_2)
            )

        return np.array([
            0.0,
            y_c,
            s,
        ])

    # ========================================================
    # Coordinate transformation
    # ========================================================

    def to_cartesian(self, s, eta, theta):
        """Convert generalized coordinates to Cartesian coordinates."""

        if not self.contains(s, eta, theta):
            raise ValueError(
                "Point is outside the generalized-coordinate domain."
            )

        R = self.radius(s)
        rho = eta * R

        centre = self.centerline(s)

        x = rho * np.cos(theta)
        y = centre[1] + rho * np.sin(theta)
        z = s

        return np.array([x, y, z])

    # ========================================================
    # Containment
    # ========================================================

    def contains(self, s, eta, theta=None):
        """Check whether generalized coordinates lie in the volume."""

        s_inside = (
            0.0 <= s <= self.L_total
        )

        eta_inside = (
            0.0 <= eta <= 1.0
        )

        if theta is None:
            theta_inside = True
        else:
            theta_inside = np.isfinite(theta)

        return (
            s_inside
            and eta_inside
            and theta_inside
        )

    def region(self, s):
        """Return region number containing axial position s."""

        if not 0 <= s <= self.L_total:
            return None

        if s <= self.L1:
            return 1

        elif s <= self.L1 + self.L2:
            return 2

        else:
            return 3

    # ========================================================
    # Plotting
    # ========================================================

    def plot(
        self,
        test_point=None,
        n_theta=120,
        n_s=80,
    ):
        """Plot the volume and optionally a test point.

        Parameters
        ----------
        test_point : tuple or None
            Optional generalized point:
                (s, eta, theta)

        n_theta : int
            Circumferential mesh resolution.

        n_s : int
            Axial mesh resolution per region.
        """

        theta = np.linspace(
            0,
            2 * np.pi,
            n_theta,
        )

        fig = go.Figure()

        # ====================================================
        # Region 1: cylinder
        # ====================================================

        s_cyl = np.linspace(
            0,
            self.L1,
            n_s,
        )

        Theta_cyl, S_cyl = np.meshgrid(
            theta,
            s_cyl,
        )

        X_cyl = (
            self.R1
            * np.cos(Theta_cyl)
        )

        Y_cyl = (
            self.R1
            * np.sin(Theta_cyl)
        )

        Z_cyl = S_cyl

        fig.add_trace(
            go.Surface(
                x=X_cyl,
                y=Y_cyl,
                z=Z_cyl,
                opacity=0.45,
                colorscale=[
                    [0, "steelblue"],
                    [1, "steelblue"],
                ],
                showscale=False,
                name="Cylinder",
            )
        )

        # ====================================================
        # Region 2: frustum 1
        # ====================================================

        s_local_1 = np.linspace(
            0,
            self.L2,
            n_s,
        )

        Theta_1, S1 = np.meshgrid(
            theta,
            s_local_1,
        )

        R_1 = (
            self.R1
            + (self.R2 - self.R1)
            * S1
            / self.L2
        )

        X_1 = (
            R_1
            * np.cos(Theta_1)
        )

        Y_1 = (
            R_1
            * np.sin(Theta_1)
            + S1
            * np.tan(self.psi_1)
        )

        Z_1 = (
            self.L1
            + S1
        )

        fig.add_trace(
            go.Surface(
                x=X_1,
                y=Y_1,
                z=Z_1,
                opacity=0.45,
                colorscale=[
                    [0, "darkorange"],
                    [1, "darkorange"],
                ],
                showscale=False,
                name="Frustum 1",
            )
        )

        # ====================================================
        # Region 3: frustum 2
        # ====================================================

        s_local_2 = np.linspace(
            0,
            self.L3,
            n_s,
        )

        Theta_2, S2 = np.meshgrid(
            theta,
            s_local_2,
        )

        R_2 = (
            self.R2
            + (self.R3 - self.R2)
            * S2
            / self.L3
        )

        X_2 = (
            R_2
            * np.cos(Theta_2)
        )

        Y_2 = (
            self.y_interface_2
            + R_2
            * np.sin(Theta_2)
            + S2
            * np.tan(self.psi_2)
        )

        Z_2 = (
            self.L1
            + self.L2
            + S2
        )

        fig.add_trace(
            go.Surface(
                x=X_2,
                y=Y_2,
                z=Z_2,
                opacity=0.45,
                colorscale=[
                    [0, "yellow"],
                    [1, "yellow"],
                ],
                showscale=False,
                name="Frustum 2",
            )
        )

        # ====================================================
        # End faces
        # ====================================================

        def add_face(
            s,
            radius,
            y_center,
            color,
            name,
        ):
            radial = np.linspace(
                0,
                radius,
                50,
            )

            Theta, Rad = np.meshgrid(
                theta,
                radial,
            )

            X = (
                Rad
                * np.cos(Theta)
            )

            Y = (
                y_center
                + Rad
                * np.sin(Theta)
            )

            Z = np.full_like(
                X,
                s,
            )

            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    opacity=0.45,
                    colorscale=[
                        [0, color],
                        [1, color],
                    ],
                    showscale=False,
                    name=name,
                )
            )

        # Left face
        add_face(
            s=0.0,
            radius=self.R1,
            y_center=0.0,
            color="steelblue",
            name="Inlet face",
        )

        # Final face
        y_end = (
            self.y_interface_2
            + self.L3
            * np.tan(self.psi_2)
        )

        add_face(
            s=self.L_total,
            radius=self.R3,
            y_center=y_end,
            color="yellow",
            name="Outlet face",
        )

        # ====================================================
        # Centreline
        # ====================================================

        s_line = np.linspace(
            0,
            self.L_total,
            300,
        )

        centres = np.array([
            self.centerline(s)
            for s in s_line
        ])

        fig.add_trace(
            go.Scatter3d(
                x=centres[:, 0],
                y=centres[:, 1],
                z=centres[:, 2],
                mode="lines",
                line=dict(
                    color="black",
                    width=5,
                    dash="dash",
                ),
                name="Centreline",
            )
        )

        # ====================================================
        # Test point
        # ====================================================

        if test_point is not None:

            s, eta, theta_point = test_point

            contained = self.contains(
                s,
                eta,
                theta_point,
            )

            if contained:
                point = self.to_cartesian(
                    s,
                    eta,
                    theta_point,
                )

                point_color = "limegreen"
                point_name = "Test point — inside"

            else:
                # We can still plot a point if s is valid,
                # even when eta > 1.
                if 0 <= s <= self.L_total:

                    R = self.radius(s)
                    rho = eta * R
                    centre = self.centerline(s)

                    point = np.array([
                        rho * np.cos(theta_point),
                        centre[1]
                        + rho
                        * np.sin(theta_point),
                        s,
                    ])

                    point_color = "red"
                    point_name = "Test point — outside"

                else:
                    point = None

            if point is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[point[0]],
                        y=[point[1]],
                        z=[point[2]],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=point_color,
                        ),
                        name=point_name,
                    )
                )

                # Line from centreline to point
                centre = self.centerline(s)

                fig.add_trace(
                    go.Scatter3d(
                        x=[
                            centre[0],
                            point[0],
                        ],
                        y=[
                            centre[1],
                            point[1],
                        ],
                        z=[
                            centre[2],
                            point[2],
                        ],
                        mode="lines",
                        line=dict(
                            color=point_color,
                            width=4,
                            dash="dot",
                        ),
                        showlegend=False,
                    )
                )

        # ====================================================
        # Layout
        # ====================================================

        fig.update_layout(
            title="Piecewise axisymmetric volume",

            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",

                aspectmode="data",

                camera=dict(
                    eye=dict(
                        x=-1,
                        y=1,
                        z=1,
                    ),
                    up=dict(
                        x=0,
                        y=1,
                        z=0,
                    ),
                ),
            ),

            width=1100,
            height=700,
        )

        fig.show()


# ============================================================
# Geometry definition
# ============================================================

D1 = 2.0
D2 = 1.0
D3 = 0.5

L1 = 3.0
L2 = 2.0
L3 = 1.5

psi_1 = np.deg2rad(12.0)
psi_2 = np.deg2rad(12.0)


volume = PiecewiseAxisymmetricVolume(
    D1=D1,
    D2=D2,
    D3=D3,
    L1=L1,
    L2=L2,
    L3=L3,
    psi_1=psi_1,
    psi_2=psi_2,
)


# ============================================================
# Test point
# ============================================================

s = 6.5
eta = 0.5
theta = np.deg2rad(45.0)

print(
    "Contained:",
    volume.contains(
        s,
        eta,
        theta,
    )
)

print(
    "Cartesian coordinate:",
    volume.to_cartesian(
        s,
        eta,
        theta,
    )
)


# Plot geometry + tested point
volume.plot(
    test_point=(
        s,
        eta,
        theta,
    )
)