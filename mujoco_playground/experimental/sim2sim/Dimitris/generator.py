import numpy as np


class ShapeGenerator:
    def __init__(self, desired_shape: str, shape_size: float):
        self.shape_size = shape_size
        self.shape_type = desired_shape.lower()

    def generate(self):
        """Returns a flat list of (x, y) waypoints for the selected shape."""
        if self.shape_type == 'circle':
            return self._generate_circle()
        elif self.shape_type == 'rectangle':
            return self._generate_rectangle()
        elif self.shape_type == 'triangle':
            return self._generate_triangle()
        else:
            raise ValueError(f"Invalid shape type: {self.shape_type}")

    def _generate_circle(self, num_points=100):
        return [
            (self.shape_size * np.cos(t), self.shape_size * np.sin(t))
            for t in np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        ]

    def _generate_rectangle(self):
        length = self.shape_size
        width = self.shape_size

        # Points per side
        num_points_per_side = 25
        points = []

        # Bottom (left to right)
        for i in range(num_points_per_side):
            x = i / num_points_per_side * length
            points.append((x, 0.0))

        # Right (bottom to top)
        for i in range(num_points_per_side):
            y = i / num_points_per_side * width
            points.append((length, y))

        # Top (right to left)
        for i in range(num_points_per_side):
            x = length - (i / num_points_per_side * length)
            points.append((x, width))

        # Left (top to bottom)
        for i in range(num_points_per_side):
            y = width - (i / num_points_per_side * width)
            points.append((0.0, y))

        return points

    def _generate_triangle(self):
        # Equilateral triangle
        p1 = (0.0, 0.0)
        p2 = (self.shape_size, 0.0)
        p3 = (self.shape_size / 2, self.shape_size * np.sqrt(3) / 2)

        return [p1, p2, p3, p1]  # loop back to start