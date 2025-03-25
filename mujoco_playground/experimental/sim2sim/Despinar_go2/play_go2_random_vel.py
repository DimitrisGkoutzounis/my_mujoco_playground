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

from Locomotion_Controller import Locomotion_Controller
from Navigator import Navigator

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent / "onnx" # Modified this (_HERE.parent), one folder back before access .onnx


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

  locomotion_policy = Locomotion_Controller(
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
