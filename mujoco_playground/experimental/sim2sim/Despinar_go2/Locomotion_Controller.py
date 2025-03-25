import mujoco
import numpy as np
import onnxruntime as rt
from Navigator import Navigator

class Locomotion_Controller:
  """Low level - Locomotion ONNX controller for Go2 robot."""

  def __init__(
      self,
      locomotion_policy_path: str,
      default_angles: np.ndarray,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.5,
      vel_scale_y: float = 0.8,
      vel_scale_rot: float = 2 * np.pi,
      locomotion_cmd: np.array = np.zeros(3) # Navigator = Navigator()
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        locomotion_policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps

    self._joystick = np.zeros(3)#locomotion_cmd#Navigator(
        #vel_scale_x=vel_scale_x,
        #vel_scale_y=vel_scale_y,
        #vel_scale_rot=vel_scale_rot,
    #)

  def update_locomotion_cmd(self, cmd_):
    self._joystick = cmd_

  def get_obs_locomotion(self, model, data) -> np.ndarray:
    #print all detected sensors

    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    obs_locomotion = np.hstack([
        linvel,
        gyro,
        gravity,
        joint_angles,
        joint_velocities,
        self._last_action,
        self._joystick,
    ])
    print("From get_obs_locomotion",self._joystick)

    return obs_locomotion.astype(np.float32)

  def exec_locomotion_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs_locomotion = self.get_obs_locomotion(model, data)
      # print current time
      onnx_input = {"obs": obs_locomotion.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      self._last_action = onnx_pred.copy()
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles

 
