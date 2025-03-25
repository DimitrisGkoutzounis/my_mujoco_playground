
from Navigator import Navigator
import mujoco
import numpy as np

# NavigationPolicy Class: Defines the navigation policy, TBD
#  Until now: it has Navigator() which decides randomly the vel cmds
class NavigationPolicy:
    def __init__(
        self,
        default_angles: np.ndarray,
        n_substeps: int = 0,
        action_scale: float = 0.5,
        vel_scale_x: float = 1.5,
        vel_scale_y: float = 0.8,
        vel_scale_rot: float = 2 * np.pi,     
       ):
        self._default_angles = default_angles
        self._n_substeps = n_substeps
        self._action_scale = action_scale
        self._counter = 0
        # Added this to decide the vel Cmds: in future, it will come from Navigation Policy
        self.Navigator = Navigator()

    def get_action(self, model, data) -> np.ndarray:
        """Returns zero action (does nothing)."""
        return np.zeros_like(self._default_angles, dtype=np.float32)
        
    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            data.ctrl[:] = self.get_action(model, data)
            print(f"Simulation time: {data.time:.4f}")
            print(f"Action: {data.ctrl}")
            print(f"Sensor data: {data.sensor('local_linvel').data}")
            print(f"Sensor data: {data.sensor('gyro').data}")
