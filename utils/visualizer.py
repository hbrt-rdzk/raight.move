from enum import Enum

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class Joint(Enum):
    LEFT_SHOULDER = 0
    RIGHT_SHOULDER = 1
    LEFT_ELBOW = 2
    RIGHT_ELBOW = 3
    LEFT_HAND = 4
    RIGHT_HAND = 5
    LEFT_HIP = 6
    RIGHT_HIP = 7
    LEFT_KNEE = 8
    RIGHT_KNEE = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11


class Visualizer:
    joint_names = [joint.name for joint in Joint]
    joint_connections = [
        (Joint.LEFT_SHOULDER, Joint.LEFT_HIP),
        (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
        (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
        (Joint.RIGHT_SHOULDER, Joint.RIGHT_HIP),
        (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
        (Joint.LEFT_ELBOW, Joint.LEFT_HAND),
        (Joint.RIGHT_ELBOW, Joint.RIGHT_HAND),
        (Joint.RIGHT_HIP, Joint.LEFT_HIP),
        (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
        (Joint.RIGHT_KNEE, Joint.RIGHT_FOOT),
        (Joint.LEFT_HIP, Joint.LEFT_KNEE),
        (Joint.LEFT_KNEE, Joint.LEFT_FOOT),
    ]

    def __init__(self, image: np.ndarray, joints_2d: list, joints_3d: list) -> None:
        self._image = image
        self._joints_2d = joints_2d[5:]
        self._joints_3d = joints_3d[5:]

    def plot_2D(self, size: tuple = (800, 800)) -> None:
        width, height = size
        fig = px.imshow(self._image, width=width, height=height)
        fig.add_scatter(
            x=self._joints_2d[:, 0],
            y=self._joints_2d[:, 1],
            mode="markers",
            marker=dict(color="blue", size=6),
            text=self.joint_names,
            textposition="top center",
        )

        for conn in self.joint_connections:
            fig.add_trace(
                go.Scatter(
                    x=[
                        self._joints_2d[conn[0].value, 0],
                        self._joints_2d[conn[1].value, 0],
                    ],
                    y=[
                        self._joints_2d[conn[0].value, 1],
                        self._joints_2d[conn[1].value, 1],
                    ],
                    mode="lines",
                    line=dict(color="blue", width=2),
                )
            )
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        fig.show()

    def plot_3D(self, size: tuple = (800, 800)) -> None:
        width, height = size
        fig = px.scatter_3d(
            x=self._joints_3d[:, 0],
            y=self._joints_3d[:, 1],
            z=self._joints_3d[:, 2],
            width=width,
            height=height,
            text=self.joint_names,
        )
        for conn in self.joint_connections:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        self._joints_3d[conn[0].value, 0],
                        self._joints_3d[conn[1].value, 0],
                    ],
                    y=[
                        self._joints_3d[conn[0].value, 1],
                        self._joints_3d[conn[1].value, 1],
                    ],
                    z=[
                        self._joints_3d[conn[0].value, 2],
                        self._joints_3d[conn[1].value, 2],
                    ],
                    mode="lines",
                    line=dict(color="blue", width=2),
                )
            )

        fig.show()

    def set_joints(self, joints: list) -> None:
        self._joints = joints[5:]

    def set_image(self, image: np.ndarray) -> None:
        self._image = image
