import numpy as np
import time  # Add at the top
from mujoco_playground.experimental.sim2sim.Dimitris.LocomotionPolicy import LocomotionPolicy

class TrajectoryPolicy:
    def __init__(
        self,
        n_substeps: int = 5,
        locomotion_policy: LocomotionPolicy = None,
        max_vel: float = 5,
    ):
        self._counter = 0
        self._n_substeps = n_substeps
        self.locomotion_policy = locomotion_policy
        self.max_vel = max_vel
        self.last_update_time = 0.0

    def controller(self, model, data):
        self._counter += 1
        if self._counter % self._n_substeps != 0:
            return

        current_time = time.time()
        if current_time - self.last_update_time < 1.0:
            return  # Wait at least 1 second before sending a new command
        self.last_update_time = current_time

        # Generate random velocities in range [-max_vel, max_vel]
        x_vel = np.random.uniform(0, self.max_vel)
        # y_vel = np.random.uniform(-self.max_vel, self.max_vel)
        yaw_rate = 0.0  # keep orientation fixed

        print(f"[RandomController] Sending: x_vel={x_vel:.2f}, y_vel={0:.2f}")

        if self.locomotion_policy:
            self.locomotion_policy.set_cmd_vel(x_vel, 0, yaw_rate)
            self.locomotion_policy.get_control(model, data)