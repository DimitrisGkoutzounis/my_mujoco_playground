import numpy as np
import time
from mujoco_playground.experimental.sim2sim.Dimitris.LocomotionPolicy import LocomotionPolicy

class TrajectoryPolicy:
    def __init__(
        self,
        n_substeps: int = 5,
        locomotion_policy: LocomotionPolicy = None,
    ):
        self._counter = 0
        self._n_substeps = n_substeps
        self.locomotion_policy = locomotion_policy

        self.last_update_time = time.time()  # initialize properly
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0

    def controller(self, model, data):
        current_time = time.time()


        if (current_time - self.last_update_time) >= 2.0: # 2 second has passed
            self.update_control()
            print(f"[cmd_vel updated] x: {self.cmd_vel_x}, y: {self.cmd_vel_y}")
            self.last_update_time = current_time
        # Always apply control
        if self.locomotion_policy:
            self.locomotion_policy.get_control(model, data)

    def update_control(self):
        # Generate random velocity
        self.generate_velocity()
        self.locomotion_policy.set_cmd_vel(self.cmd_vel_x, self.cmd_vel_y)

    def generate_velocity(self):
        
        self.cmd_vel_x = np.around(np.random.uniform(-1, 1), 3)
        self.cmd_vel_y = np.around(np.random.uniform(-1, 1), 3)