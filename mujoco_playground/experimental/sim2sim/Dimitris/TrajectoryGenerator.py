# ==============================================================================
# ==============================================================================

# Hey, I modified Copyright 2025 DeepMind Technologies Limited
# for an upcoming project regarding navigation policies for Go2 Unitree's quadrupedal robot.

# ==============================================================================
# ==============================================================================
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np


class TrajectoryGenerator:
    def __init__(self):
        pass

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
    
    def _generate_simple_target(self):
        """Return x and y coordinates of a target point."""
        
        #generate random x and y coordinates
        target_x = np.random.uniform(-5, 5)
        target_y = np.random.uniform(-5, 5)
        
        target_coords = [(target_x, target_y)]
        return target_coords
        
        