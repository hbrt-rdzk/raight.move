from enum import Enum
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class Visualizer:
    """
    A class to visualize 2D and 3D joint data using plotly.

    Attributes:
        image (np.ndarray): Image on which 2D joints are to be plotted.
        joints_2d (np.ndarray): Array of 2D joint positions.
        joints_3d (np.ndarray): Array of 3D joint positions.
        config_data (dict): Configuration data including joint names and connections.
    """

    def __init__(
        self,
        image: np.ndarray,
        joints_2d: np.ndarray,
        joints_3d: np.ndarray,
        config_data: dict,
    ) -> None:
        self._image = image
        self._joints_2d = joints_2d
        self._joints_3d = joints_3d

        self.joint_names_2d = self._create_joint_names(config_data, "JointsCoCo2D")
        self.joint_names_3d = self._create_joint_names(config_data, "JointsHuman3D")
        self.joint_connections_2d = config_data.get("coco_connections", [])
        self.joint_connections_3d = config_data.get("human36_connections", [])
        self.angles = config_data.get("angles", [])

    def plot_2D(self, size: tuple = (800, 800), remove_head: bool = False) -> None:
        """
        Plot 2D joint data on an image.

        Args:
            size (tuple): Size of the plot (width, height).
            remove_head (bool): Whether to remove the head joints from the plot.
        """
        joints, connections = self._prepare_joints_for_plotting(
            self._joints_2d, self.joint_connections_2d, remove_head, [0, 1, 2, 3, 4]
        )
        self._create_2D_plot(joints, connections, size)

    def plot_3D(
        self, size: tuple = (800, 800), remove_head: bool = False, offset: float = 0.05
    ) -> None:
        """
        Plot 3D joint data in a scatter plot.

        Args:
            size (tuple): Size of the plot (width, height).
            remove_head (bool): Whether to remove the head joints from the plot.
            offset (float): Offset to add to the axis limits for better visualization.
        """
        joints, connections = self._prepare_joints_for_plotting(
            self._joints_3d, self.joint_connections_3d, remove_head, [9, 10]
        )
        self._create_3D_plot(joints, connections, size, offset)

    def _create_2D_plot(
        self, joints: np.ndarray, connections: list, size: tuple
    ) -> None:
        """
        Create and display a 2D plot of joints on an image.

        Args:
            joints (np.ndarray): Array of joint positions.
            connections (list): List of joint connections.
            size (tuple): Size of the plot (width, height).
        """
        fig = px.imshow(self._image, width=size[0], height=size[1])
        fig.add_scatter(
            x=joints[:, 0],
            y=joints[:, 1],
            mode="markers",
            marker=dict(color="blue", size=6),
            text=self.joint_names_2d,
            textposition="top center",
        )
        for conn in connections:
            fig.add_trace(
                go.Scatter(
                    x=[joints[conn[0], 0], joints[conn[1], 0]],
                    y=[joints[conn[0], 1], joints[conn[1], 1]],
                    mode="lines",
                    line=dict(color="blue", width=2),
                )
            )
        calculated_angles = self._calculate_angles()
        for idx, (joint_numbers, angle) in enumerate(
            zip(self.angles, calculated_angles)
        ):
            if idx in [2, 3, 8, 9]:
                ay = 30
            else:
                ay = -30

            fig.add_annotation(
                x=joints[joint_numbers[1], 0],
                y=joints[joint_numbers[1], 1],
                text=f"{angle:.2f}°",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ay=ay,
                arrowcolor="#ff0000",
                font=dict(family="Courier New, monospace", size=12, color="#ffffff"),
            )

        x_range = [0, self._image.shape[1]]
        y_range = [self._image.shape[0], 0]
        fig.update_xaxes(showticklabels=False, range=x_range)
        fig.update_yaxes(showticklabels=False, range=y_range)
        fig.update_layout(showlegend=False)
        fig.show()

    def _create_3D_plot(
        self, joints: np.ndarray, connections: list, size: tuple, offset: float
    ) -> None:
        """
        Create and display a 3D plot of joints.

        Args:
            joints (np.ndarray): Array of joint positions.
            connections (list): List of joint connections.
            size (tuple): Size of the plot (width, height).
            offset (float): Offset for the axis limits.
        """
        fig = px.scatter_3d(
            x=joints[:, 0],
            y=joints[:, 1],
            z=joints[:, 2],
            width=size[0],
            height=size[1],
            text=self.joint_names_3d,
        )
        for conn in connections:
            fig.add_trace(
                go.Scatter3d(
                    x=[joints[conn[0], 0], joints[conn[1], 0]],
                    y=[joints[conn[0], 1], joints[conn[1], 1]],
                    z=[joints[conn[0], 2], joints[conn[1], 2]],
                    mode="lines",
                    line=dict(color="blue", width=2),
                )
            )

        useful_joints = joints[np.any(joints > -100, axis=1)]
        x_range = [
            useful_joints[:, 0].min() - offset,
            useful_joints[:, 0].max() + offset,
        ]
        y_range = [
            useful_joints[:, 1].min() - offset,
            useful_joints[:, 1].max() + offset,
        ]
        z_range = [
            useful_joints[:, 2].min() - offset,
            useful_joints[:, 2].max() + offset,
        ]
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
            )
        )
        fig.update_layout(showlegend=False)
        fig.show()

    def _create_joint_names(self, config_data: dict, key: str) -> list:
        """
        Create joint names from the configuration data.

        Args:
            config_data (dict): Configuration data containing joint names.
            key (str): Key to access the joint names in the config_data.

        Returns:
            list: List of joint names.
        """
        try:
            return [joint.name for joint in Enum(key, config_data[key])]
        except KeyError:
            print(
                f"Warning: '{key}' not found in config_data. Using empty joint names."
            )
            return []

    def _adjust_joints(
        self, joints: np.ndarray, connections: list, joints_to_remove: tuple
    ) -> tuple:
        """
        Adjust joints and connections by removing specified joints.

        Args:
            joints (np.ndarray): Array of joint positions.
            connections (list): List of joint connections.
            joints_to_remove (tuple): Indices of joints to remove.

        Returns:
            tuple: Adjusted joints and connections.
        """
        joints = np.array(
            [
                joint if idx not in joints_to_remove else [-100] * joints.shape[1]
                for idx, joint in enumerate(joints)
            ]
        )
        connections = [
            conn
            for conn in connections
            if all(idx not in joints_to_remove for idx in conn)
        ]

        return joints, connections

    def _prepare_joints_for_plotting(
        self,
        joints: np.ndarray,
        connections: list,
        remove: bool,
        joints_to_remove: list,
    ) -> tuple:
        """
        Prepare joints and connections for plotting based on the removal flag.

        Args:
            joints (np.ndarray): Array of joint positions.
            connections (list): List of joint connections.
            remove (bool): Flag to indicate whether to remove specific joints.
            joints_to_remove (list): Indices of joints to remove.

        Returns:
            tuple: Prepared joints and connections for plotting.
        """
        return (
            (joints, connections)
            if not remove
            else self._adjust_joints(joints, connections, joints_to_remove)
        )

    def _calculate_angles(self) -> list:
        """
        Calculate angles between joints in 2D space.

        Args:
            joints (list of tuples): List of points (x, y) representing joint positions in 2D space.
            angle_indices (list of tuples): List of tuples (i, j, k) where the angle at joint j
                                        between joints i and k is to be calculated.

        Returns:
            list: List of calculated angles in degrees.
        """

        def angle_between(v1, v2):
            """Calculate the angle in degrees between vectors 'v1' and 'v2'."""
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

        angles = []
        for i, j, k in self.angles:
            vec_ji = np.array(self._joints_2d[j]) - np.array(
                self._joints_2d[i]
            )  # Vector from j to i
            vec_jk = np.array(self._joints_2d[j]) - np.array(
                self._joints_2d[k]
            )  # Vector from j to k
            angle = angle_between(vec_ji, vec_jk)
            angles.append(angle)

        return angles