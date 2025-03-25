# ==============================================================================
# ==============================================================================

# Hey, I modified Copyright 2025 DeepMind Technologies Limited
# for an upcoming project regarding navigation policies for Go2 Unitree's quadrupedal robot.

# ==============================================================================
# ==============================================================================
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
from NavigationPolicy import NavigationPolicy
from TopLevelController import TopLevelController

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent / "onnx" # Modified this (_HERE.parent), one folder back before access .onnx

def main_function(model=None, data=None):

    # Set no mujoco control - default callback
    mujoco.set_mjcb_control(None)
    # Load 
    model = mujoco.MjModel.from_xml_path(
        go2_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=get_assets(),
    )
    # Mujoco data
    data = mujoco.MjData(model)
    # Mujoco reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # Define params
    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt


    # Top Level Controller - Controls EVERYTHING
    top_controller = TopLevelController()
    # Locomotion Controller from Trained Onnx Policy
    top_controller.locomotion_ctrl_ = Locomotion_Controller(
        locomotion_policy_path=(_ONNX_DIR / "go2_policy_galloping.onnx").as_posix(),
        default_angles=np.array(model.keyframe("home").qpos[7:]),
        n_substeps=n_substeps,
        action_scale=0.5,
        vel_scale_x=1.5,
        vel_scale_y=0.8,
        vel_scale_rot=2 * np.pi,
        locomotion_cmd=np.zeros(3) # Define which Navigator-Boss the locomotion cotroller will have.
    )
    # TBD: Future definition of Navigation Policy
    top_controller.navigation_policy_ = NavigationPolicy(
            default_angles=np.array(model.keyframe("home").qpos[7:]), #TBD: FIX THOSE DEFAULT ANGELS
            n_substeps=n_substeps,  #TBD: FIX THOSE DEFAULT N_SUBSTEPS
            action_scale=0.5,
        )
    # Init time variables
    top_controller.t_last_cmd = data.time
    top_controller.dt_new_cmd = 1 #0.5sec - update vel command per delta t

    # Set the main controll callback from Top Controller
    mujoco.set_mjcb_control(top_controller.control_callback)

    return model, data



if __name__ == "__main__":
  
  viewer.launch(loader=main_function)
