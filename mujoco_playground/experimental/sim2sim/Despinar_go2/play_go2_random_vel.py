# Hey, I modified Copyright 2025 DeepMind Technologies Limited
# for an upcoming project regarding navigation policies for Go2 Unitree's quadrupedal robot.

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

from mujoco_playground._src.locomotion.go2 import go2_constants
from mujoco_playground._src.locomotion.go2.base import get_assets

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent / "onnx" # Modified this (_HERE.parent), one folder back before access .onnx

# Navigator Class: Decides nav. cmds - Generates random navigation commands
class Navigator:
    def __init__(self, vel_scale_x=1.5, vel_scale_y=0.8, vel_scale_rot=2*np.pi):
        self.vel_scale_x=vel_scale_x
        self.vel_scale_y=vel_scale_y
        self.vel_scale_rot=vel_scale_rot
        self.command = np.zeros(3)

    # Random x,y,rotation
    def generate_command(self):
        # Random vel
        self.vel_scale_x = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_y = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_rot = np.around(np.random.uniform(-1, 1)*2*np.pi,3) # Random between 0-2pi, 3 dec
        # Pass it to command
        self.command = np.zeros(3)
        self.command[0] = self.vel_scale_x
        self.command[1] = self.vel_scale_y
        self.command[2] = self.vel_scale_rot

        return self.command
    
    # Random x,y: no rotation
    def generate_command_norot(self):
        # Random vel
        self.vel_scale_x = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_y = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_rot = 0.0
        # Pass it to command
        self.command = np.zeros(3)
        self.command[0] = self.vel_scale_x
        self.command[1] = self.vel_scale_y
        self.command[2] = self.vel_scale_rot

        return self.command
    
# NavigationPolicy Class: Defines the navigation policy, TBD
#  Until now: it has Navigator() which decides randomly the vel cmds
class NavigationPolicy:
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



class Locomotion_OnnxController:
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

    self._joystick = Navigator(
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
        self._joystick.generate_command_norot(),
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

  def custom_obs(self, model, data) -> np.ndarray:
     
        print(data.sensor("local_linvel").data)
      
  def custom_control(self, model: mujoco.MjModel, data:mujoco.MjData) -> None:
     self._counter += 1

     if self._counter % self._n_substeps == 0:

        print(f"Simulation time: {data.time:.4f}")
       


def load_callback(model=None, data=None):
  # Set no mujoco control - default callback
  mujoco.set_mjcb_control(None)
  # Load 
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

  locomotion_policy = Locomotion_OnnxController(
      locomotion_policy_path=(_ONNX_DIR / "go2_policy_galloping.onnx").as_posix(),
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      n_substeps=n_substeps,
      action_scale=0.5,
      vel_scale_x=1.5,
      vel_scale_y=0.8,
      vel_scale_rot=2 * np.pi,
  )

#   navigation_policy = NavigationPolicy(
#         default_angles=np.array(model.keyframe("home").qpos[7:]),
#         n_substeps=n_substeps,
#         action_scale=0.5,
#     )


  mujoco.set_mjcb_control(locomotion_policy.get_control)

  return model, data


if __name__ == "__main__":
#   pygame.event.pump()
  viewer.launch(loader=load_callback)
