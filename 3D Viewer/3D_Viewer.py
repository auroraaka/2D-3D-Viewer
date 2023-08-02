import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

class Viewer3D:
    def __init__(self):
        self.points = []
        self.lines = []
        self.polygons = []
        self.prisms = []
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('3D VIEWER')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.animation = None

    def add(self, coordinates):
        if len(coordinates) == 1:
            self.points.append(np.array(coordinates[0]))
        elif len(coordinates) == 2:
            self.lines.append(coordinates)
        elif len(coordinates) > 2 and len(coordinates) < 8:
            self.polygons.append(coordinates)
        elif len(coordinates) == 8:
            self.prisms.append(coordinates)
        else:
            raise SyntaxError("3D Viewer handles maximum of 8 coordinates")

    def init(self):
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0, 4)
        self.ax.set_zlim(0, 4)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        for point in self.points:
            self.ax.scatter(*point, c='red')

        for line in self.lines:
            xl, yl, zl = line.T
            self.ax.plot(*line.T, c='blue')

        for polygon in self.polygons:
            self.ax.add_collection3d(Poly3DCollection([polygon], facecolors='green', alpha=0.5))

        faces = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],  
            [0, 1, 5, 4],  
            [1, 2, 6, 5], 
            [2, 3, 7, 6],  
            [3, 0, 4, 7]  
        ])

        face_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']

        for prism in self.prisms:
            for face, color in zip(faces, face_colors):
                face_coordinates = prism[face]
                self.ax.add_collection3d(Poly3DCollection([face_coordinates], facecolors=color, alpha=0.5))

        self.ax.autoscale_view()

    def animate(self, frame):
        def rotate_shape(shape):
            new_shape = []
            for sub_shape in shape:
                center = np.mean(sub_shape, axis=0)
                qt_rotation = slerp(np.array([0, 0, 0, 1]), self.q_rotation, 1/60)
                rotation_matrix = quaternion_to_rotation_matrix(qt_rotation)
                translated_shape = sub_shape - center
                rotated_sub_shape = np.dot(translated_shape, rotation_matrix.T)
                rotated_shape = rotated_sub_shape + center
                new_shape.append(rotated_shape)
            return new_shape

        self.points = rotate_shape(self.points)
        self.lines = rotate_shape(self.lines)
        self.polygons = rotate_shape(self.polygons)
        self.prisms = rotate_shape(self.prisms)

        self.ax.clear()
        self.init()

        self.current_angle += self.angle_increment
        if self.current_angle >= self.target_angle:
            self.stop_animation()

    def stop_animation(self):
        if self.animation is not None:
            self.animation.event_source.stop()

    def show(self, axis, angle, num_frames=100):
        self.stop_animation()
        self.current_angle = 0.0
        self.target_angle = angle
        self.angle_increment = angle / num_frames
        self.q_rotation = axis_angle_to_norm_quaternion(axis, angle)
        self.num_frames = num_frames
        self.animation = FuncAnimation(self.fig, self.animate, frames=num_frames,
                                       init_func=self.init, interval=10, blit=False)
        plt.show()

    def rotate(self, axis, angle, num_frames=60):
        self.show(axis, angle, num_frames)


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q

    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, xz, xw = x * y, x * z, x * w
    yz, yw, zw = y * z, y * w, z * w

    m00 = 1 - 2 * (y2 + z2)
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)
    m10 = 2 * (xy + zw)
    m11 = 1 - 2 * (x2 + z2)
    m12 = 2 * (yz - xw)
    m20 = 2 * (xz - yw)
    m21 = 2 * (yz + xw)
    m22 = 1 - 2 * (x2 + y2)

    rotation_matrix = np.array([[m00, m01, m02],
                                [m10, m11, m12],
                                [m20, m21, m22]])

    return rotation_matrix


def axis_angle_to_norm_quaternion(axis, angle):
    q = np.concatenate((np.sin(angle / 2) * axis, [np.cos(angle / 2)]))
    return q / np.linalg.norm(q)


def quaternion_multiplication(q0, q1):
    v0, w0 = q0[:3], q0[3]
    v1, w1 = q1[:3], q1[3]
    w = w0 * w1 - np.dot(v0, v1)
    v = np.cross(v0, v1) + w0 * v1 + w1 * v0
    return np.array([v[0], v[1], v[2], w])


def quaternion_conjugate(q):
    return np.concatenate((-q[:3], [q[3]]))


def quaternion_division(q0, q1):
    return quaternion_multiplication(q0, quaternion_conjugate(q1))


def slerp(q0, q1, t):
    qr = quaternion_division(q0, q1)
    vr, wr = qr[:3], qr[3]
    if wr < 0:
        qr = -qr
    magnitude_vr = np.linalg.norm(vr)
    thetar = 2 * np.arctan(magnitude_vr / wr)
    nr = vr / magnitude_vr
    thetaa = t * thetar
    qa = np.concatenate((np.sin(thetaa / 2) * nr, [np.cos(thetaa / 2)]))
    return quaternion_multiplication(qa, q0)


def handle_key_press(event):
    key = event.key.lower()
    angle = np.pi/2
    if key == 'up':
        viewer.rotate(np.array([0, 1, 0]), angle)
    elif key == 'down':
        viewer.rotate(np.array([0, -1, 0]), angle)
    elif key == 'left':
        viewer.rotate(np.array([0, 0, -1]), angle)
    elif key == 'right':
        viewer.rotate(np.array([0, 0, 1]), angle)
    elif key == '.':
        viewer.rotate(np.array([1, 0, 0]), angle)
    elif key == ',':
        viewer.rotate(np.array([-1, 0, 0]), angle)

viewer = Viewer3D()

viewer.add(np.array([[1, 0, 1]]))
viewer.add(np.array([[1, 0, 1], [1, 2, 0]]))
viewer.add(np.array([[1, 0, 1], [1, 2, 0], [1, 2, 1], [1, 0, 2]]))
viewer.add(np.array([
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]
    ]) + np.array([2, 2, 2]))

viewer.fig.canvas.mpl_connect('key_press_event', handle_key_press)
viewer.init()
plt.ion()
plt.show()
input("Press Enter to exit...")