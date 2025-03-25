import numpy as np
from mujoco_playground.experimental.sim2sim.Dimitris.generator import ShapeGenerator

from mujoco_playground.experimental.sim2sim.Dimitris.LocomotionPolicy import LocomotionPolicy
class TrajectoryPolicy:
    def __init__(
        self,
        action_scale: float = 0.5,
        n_substeps: int = 5,
        shape_generator: ShapeGenerator = None,
        locomotion_policy: LocomotionPolicy = None,
    ):
        self.action_scale = action_scale
        self._counter = 0
        self._n_substeps = n_substeps
        self.shape_generator = shape_generator
        self.current_target_idx = 0
        self.position = np.array([0.0, 0.0])  # start at origin
        self.orientation = 0.0  # radians
        self.prev_time = 0.0
        self.locomotion_policy = locomotion_policy

        # Velocity parameters
        self.max_vel = 10  # m/s
        self.max_yaw_rate = np.pi / 4  # rad/s

        # List of (x, y) waypoints from shape
        if shape_generator:
            self.trajectory = shape_generator.generate()
        else:
            self.trajectory = [(1.0, 0.0)]  # simple default

    def _compute_cmd_from_waypoints(self, current_pos, target_pos):
        error = np.array(target_pos) - np.array(current_pos)
        distance_error = np.linalg.norm(error) * 20
        direction = error / (distance_error + 1e-6)

        x_vel = direction[0] * min(self.max_vel, distance_error)
        y_vel = direction[1] * min(self.max_vel, distance_error)
        yaw_rate = 0.0 
        
        print(f"Error: {error}, Distance error: {distance_error}, Direction: {direction}, X vel: {x_vel}, Y vel: {y_vel}")

        return x_vel, y_vel, yaw_rate

    def controller(self, model, data):
        self._counter += 1
        if self._counter % self._n_substeps != 0:
            return

        if self.current_target_idx >= len(self.trajectory):
            print("Trajectory complete.")
            return

        current_pos = data.qpos[0:2]
        target_pos = self.trajectory[self.current_target_idx]
        print(f"Current pos: {current_pos}, Target pos: {target_pos}")

        if np.linalg.norm(current_pos - target_pos) < 0.05:
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.trajectory):
                print("Trajectory complete.")
                return
            target_pos = self.trajectory[self.current_target_idx]

        x_vel, y_vel, yaw_rate = self._compute_cmd_from_waypoints(current_pos, target_pos)

        if self.locomotion_policy:
            self.locomotion_policy.set_cmd_vel(x_vel, y_vel, yaw_rate)
            self.locomotion_policy.get_control(model, data)