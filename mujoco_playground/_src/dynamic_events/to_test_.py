# ==============================================================================
# ==============================================================================

# Hey, I modified Copyright 2025 DeepMind Technologies Limited
# for an upcoming project regarding navigation policies for Go2 Unitree's quadrupedal robot.
# Modified from: Copyright 2025 DeepMind Technologies Limited

# ==============================================================================
# ==============================================================================

"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""


from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

# This is mine! Place holder to add arm+go2
import new_go2_constants
from base import get_assets

from Locomotion_Controller import Locomotion_Controller
from Navigator import Navigator
from NavigationPolicy import NavigationPolicy
from TopLevelController import TopLevelController

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent.parent / "experimental/sim2sim"/ "onnx" # Modified this (_HERE.parent.parent), two folders back before access .onnx

def main_function(model=None, data=None):

    # Place Holder for ENV_NAME and extract model, data
    ##################################################

    ##################################################
    print(type(_ONNX_DIR))
    # Set no mujoco control - default callback
    mujoco.set_mjcb_control(None)
    # Load 
    model = mujoco.MjModel.from_xml_path(
        new_go2_constants.UR5E_GO2_SCENE.as_posix(),
        assets=get_assets(),
    )
    # Mujoco data
    data = mujoco.MjData(model)
    # Mujoco reset
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # # Define params
    # ctrl_dt = 0.01 # prev 0.02 -> 0.01 for better ctrl
    # sim_dt =  model.opt.timestep # prev 0.004 -> 0.002
    # n_substeps = int(round(ctrl_dt / sim_dt))

    # Top Level Controller
        # n_substeps: -> Navigation Policy
    top_controller = TopLevelController(_ONNX_DIR=_ONNX_DIR,
                                         model=model)

    # top_controller.Nav_Policy.t_last_cmd = 0.0

    # # Set the main controll callback from Top Controller
    mujoco.set_mjcb_control(top_controller.Nav_Policy.step)

    return model, data



if __name__ == "__main__":
  
  viewer.launch(loader=main_function)
