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
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
import pygame

pygame.init()
pygame.display.set_mode((1, 1))  # Ensures the video system is initialized

from mujoco_playground._src.locomotion.go2 import go2_constants
from mujoco_playground._src.locomotion.go2.base import get_assets

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


class KeyboardController:
    def __init__(self, vel_scale_x=1.5, vel_scale_y=0.8, vel_scale_rot=2 * np.pi):
        self.vel_scale_x = vel_scale_x
        self.vel_scale_y = vel_scale_y
        self.vel_scale_rot = vel_scale_rot
        self.command = np.zeros(3)

    def get_command(self):
        keys = pygame.key.get_pressed()
        self.command = np.zeros(3)  # Reset command

        if keys[pygame.K_UP]:  # Forward
            self.command[0] = self.vel_scale_x
        if keys[pygame.K_s]:  # Backward
            self.command[0] = -self.vel_scale_x
        if keys[pygame.K_a]:  # Left
            self.command[1] = self.vel_scale_y
        if keys[pygame.K_d]:  # Right
            self.command[1] = -self.vel_scale_y
        if keys[pygame.K_q]:  # Rotate left
            self.command[2] = self.vel_scale_rot
        if keys[pygame.K_e]:  # Rotate right
            self.command[2] = -self.vel_scale_rot

        return self.command


class MyPolicy:
    def __init__(
        self,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
        vel_scale_x: float = 1.5,
        vel_scale_y: float = 0.8,
        vel_scale_rot: float = 2 * np.pi,     
       ):
        self._default_angles = default_angles
        self._n_substeps = n_substeps
        self._action_scale = action_scale
        self._counter = 0
        
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



class OnnxController:
  """ONNX controller for the Go-1 robot."""

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

    self._joystick = KeyboardController(
        vel_scale_x=vel_scale_x,
        vel_scale_y=vel_scale_y,
        vel_scale_rot=vel_scale_rot,
    )

  def get_obs(self, model, data) -> np.ndarray:
    #print all detected sensors

        
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    obs = np.hstack([
        linvel,
        gyro,
        gravity,
        joint_angles,
        joint_velocities,
        self._last_action,
        self._joystick.get_command(),
    ])
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




def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      go2_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02
  sim_dt = 0.004
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "go2_lest_see.onnx").as_posix(),
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      n_substeps=n_substeps,
      action_scale=0.5,
      vel_scale_x=1.5,
      vel_scale_y=0.8,
      vel_scale_rot=2 * np.pi,
    )


  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  pygame.event.pump()
  viewer.launch(loader=load_callback)
