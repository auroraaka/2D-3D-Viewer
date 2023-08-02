import matplotlib.pyplot as plt
import json
import numpy as np


class Editor:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.shape = None
        self.start_point = None
        self.shapes = []
        self.buttons = []

        plt.get_current_fig_manager().set_window_title("2D TRANSFORM EDITOR")
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.create_buttons()

    def create_buttons(self):
        buttons = [
            ("Clear", self.clear),
            ("Load", self.load),
            ("Save", self.save),
            ("Transform", self.transform),
        ]

        for i, (label, callback) in enumerate(buttons):
            button = plt.Button(plt.axes([0.75 - i * 0.15, 0.11, 0.15, 0.075]), label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def on_press(self, event):
        if event.button == 1 and all(
            event.inaxes != self.buttons[i].ax for i in range(len(self.buttons))
        ):
            self.start_point = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.start_point is not None:
            if self.shape:
                self.shape.set_width(event.xdata - self.start_point[0])
                self.shape.set_height(event.ydata - self.start_point[1])
            else:
                self.shape = plt.Rectangle(self.start_point, 0, 0, fill=False)
                self.ax.add_patch(self.shape)

        self.fig.canvas.draw()

    def on_release(self, event):
        if event.button == 1 and self.start_point is not None:
            x, y = self.start_point
            vertices = np.array(
                [
                    self.start_point,
                    (event.xdata, y),
                    (event.xdata, event.ydata),
                    (x, event.ydata),
                ]
            )

            self.shapes.append(Rectangle(vertices))
            self.start_point = None
            self.shape = None

        self.fig.canvas.draw()

    def save(self, event):
        if self.shapes:
            filename = "shapes.json"
            with open(filename, "w") as file:
                json.dump([shape.vertices.tolist() for shape in self.shapes], file)
            print(f"Data saved to {filename}")

    def load(self, event):
        try:
            filename = "shapes.json"
            with open(filename, "r") as file:
                data = json.load(file)
            self.shapes = [Rectangle(np.array(shape)) for shape in data]
            for shape in data:
                self.ax.add_patch(
                    plt.Polygon(np.array(shape), fill=False, edgecolor="black")
                )
            self.ax.autoscale()
            print(f"Data loaded from {filename}")
        except FileNotFoundError:
            pass

    def clear(self, event):
        for rectangle in self.ax.patches:
            rectangle.remove()

        self.shapes = []
        self.fig.canvas.draw()

    def transform(self, event):
        transformation_type = input(
            "Choose a transformation (Translation/Rigid/Similarity/Affine/Projective): "
        )

        if transformation_type == "Translation":
            dx = float(input("Enter the translation in the x-direction: "))
            dy = float(input("Enter the translation in the y-direction: "))
            transformation = Translation(np.array([dx, dy]))
        elif transformation_type == "Rigid":
            dx = float(input("Enter the translation in the x-direction: "))
            dy = float(input("Enter the translation in the y-direction: "))
            theta = np.radians(float(input("Enter the rotation angle in degrees: ")))
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            transformation = Rigid(np.array([dx, dy]), rotation_matrix)
        elif transformation_type == "Similarity":
            dx = float(input("Enter the translation in the x-direction: "))
            dy = float(input("Enter the translation in the y-direction: "))
            theta = np.radians(float(input("Enter the rotation angle in degrees: ")))
            scale = float(input("Enter the scale factor: "))
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            transformation = Similarity(np.array([dx, dy]), rotation_matrix, scale)
        elif transformation_type == "Affine":
            matrix = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    matrix[i, j] = float(
                        input(
                            f"Enter the value for row {i+1}, column {j+1} of the affine matrix: "
                        )
                    )
            transformation = Affine(matrix)
        elif transformation_type == "Projective":
            matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    matrix[i, j] = float(
                        input(
                            f"Enter the value for row {i+1}, column {j+1} of the projective matrix: "
                        )
                    )
            transformation = Projective(matrix)
        else:
            print("Invalid transformation type. Please choose a valid transformation.")
            return

        transformed_shapes = []
        for shape in self.shapes:
            transformed_shape = shape.apply_transformation(transformation)
            transformed_shapes.append(Rectangle(transformed_shape))
            self.ax.add_patch(
                plt.Polygon(transformed_shape, fill=False, edgecolor="red")
            )

        self.shapes = transformed_shapes
        self.ax.autoscale()
        self.fig.canvas.draw()


class Rectangle:
    def __init__(self, vertices):
        self.vertices = vertices

    def apply_transformation(self, transformation):
        return transformation.transform(self)


class Transformation:
    def __init__(self):
        pass

    def transform(self, rectangle):
        raise NotImplementedError


class Translation(Transformation):
    def __init__(self, translation):
        super().__init__()
        self.translation = translation

    def transform(self, rectangle):
        translated_vertices = rectangle.vertices + self.translation
        return translated_vertices


class Rigid(Translation):
    def __init__(self, translation, rotation):
        super().__init__(translation)
        self.rotation = rotation

    def transform(self, rectangle):
        translated_vertices = super().transform(rectangle)
        rigid_vertices = translated_vertices @ self.rotation
        return rigid_vertices


class Similarity(Rigid):
    def __init__(self, translation, rotation, scale):
        super().__init__(translation, rotation)
        self.scale = scale

    def transform(self, rectangle):
        scaled_vertices = rectangle.vertices * self.scale
        similarity_vertices = super().transform(Rectangle(scaled_vertices))
        return similarity_vertices


class Affine(Transformation):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    def transform(self, rectangle):
        affine_vertices = rectangle.vertices @ self.matrix.T
        return affine_vertices


class Projective(Transformation):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    def transform(self, rectangle):
        homogeneous_vertices = np.hstack((rectangle.vertices, np.ones((4, 1))))
        projective_vertices = self.matrix @ homogeneous_vertices.T
        projective_vertices /= projective_vertices[2]
        return projective_vertices.T[:, :2]


if __name__ == "__main__":
    editor = Editor()
    plt.show()
