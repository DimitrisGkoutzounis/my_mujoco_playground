import mujoco
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.dynamic_events.arm_mujoco.src.Robot  import RobotGo2


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
      locomotion_cmd: np.array = np.zeros(3)
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

    self._joystick = np.zeros(3)

  def update_locomotion_cmd(self, cmd_):
    self._joystick = cmd_

  def get_obs_locomotion(self, model, data, robot) -> np.ndarray:
    #print all detected sensors
    # print("In get_obs_locomotion")
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    #[7:] HERE
    joint_angles = data.qpos[robot.i_start_qpos:robot.i_end_qpos] - self._default_angles
    #[:6] HERE [6 arm, 6 free joint, and then start from 12 = 13(go2.i_start_qpos) -1]
    joint_velocities = data.qvel[robot.i_start_qpos-1:robot.i_end_qpos] 

    obs_locomotion = np.hstack([
        linvel,
        gyro,
        gravity,
        joint_angles,
        joint_velocities,
        self._last_action,
        self._joystick,
    ])
    # print("From get_obs_locomotion",self._joystick)

    return obs_locomotion.astype(np.float32)

  def exec_locomotion_control(self, model: mujoco.MjModel, data: mujoco.MjData, robot: RobotGo2) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs_locomotion = self.get_obs_locomotion(model, data, robot=robot)
      # print current time
      onnx_input = {"obs": obs_locomotion.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      self._last_action = onnx_pred.copy()
      # HERE [:]
      data.ctrl[robot.i_start_ctrl:robot.i_end_ctrl] = onnx_pred * self._action_scale + self._default_angles