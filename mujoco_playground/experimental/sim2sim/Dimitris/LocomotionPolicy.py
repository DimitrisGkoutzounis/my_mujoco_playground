
#this is the locomotion policy for the go2.
from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
import pygame




_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent / "onnx"

class LocomotionPolicy:
  """Locomotion policy for the Go2 robot."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.5,
      vel_scale_y: float = 0.8,
      vel_scale_rot: float = 2 * np.pi,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps
    self.cmd_x = 0.0
    self.cmd_y = 0.0
    self.cmd_yaw = 0.0
    
    self.curr_x = 0.0
    self.curr_y = 0.0
    self.curr_yaw = 0.0
    

     
  def set_cmd_vel(self, x, y):
    """Set command velocities from high-level trajectory controller."""
    
    self.cmd_x = x
    self.cmd_y = y
    
  def current_pos(self, data):
    """Get the current position of the robot."""
    self.curr_x = data.qpos[0]
    self.curr_y = data.qpos[1]
    self.curr_yaw = data.qpos[2]
    
    return np.array([self.curr_x, self.curr_y, self.curr_yaw])

  def get_obs(self, model, data) -> np.ndarray:
    #print all detected sensors
        
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    
    cmd_vels = np.array([self.cmd_x, self.cmd_y, self.cmd_yaw])
    print(f"cmd_vels: {cmd_vels}")
    
    
    obs = np.hstack([
        linvel,
        gyro,
        gravity,
        joint_angles,
        joint_velocities,
        self._last_action,
        cmd_vels
    ])
    
    #print the position of the robot
    print(f"robot position: {data.qpos[:3]}")
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      # print current time
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      self._last_action = onnx_pred.copy()
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles

    