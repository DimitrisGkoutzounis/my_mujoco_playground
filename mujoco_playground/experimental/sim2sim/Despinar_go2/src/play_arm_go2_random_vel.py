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
# import glfw

# from mujoco_playground._src.locomotion.go2 import go2_constants
# from mujoco_playground._src.locomotion.go2.base import get_assets

# This is mine! Place holder to add arm+go2
from mujoco_playground._src.dynamic_events import new_go2_constants
from mujoco_playground._src.dynamic_events.base  import get_assets

from Locomotion_Controller import Locomotion_Controller
from Navigator import Navigator
from NavigationPolicy import NavigationPolicy
from TopLevelController import TopLevelController

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent.parent / "onnx" # Modified this (_HERE.parent.parent), two folders back before access .onnx

def main_function(model=None, data=None):

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

    # renderer = mujoco.Renderer(model, height=480, width=640)
    # context =  renderer._mjr_context

    # Which is the Go2Env
    # perception_context = go2_env.get_context()

    # # ========== GLFW Setup ==========
    # if not glfw.init():
    #     raise RuntimeError("Could not initialize GLFW")

    # window = glfw.create_window(1244, 700, "MuJoCo Arm Control", None, None)
    # glfw.make_context_current(window)
    # glfw.swap_interval(1)

    # cam = mujoco.MjvCamera()
    # opt = mujoco.MjvOption()
    # scene = mujoco.MjvScene(model, maxgeom=2000)
    # context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    # perception_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)


    # Define params
    ctrl_dt = 0.02
    sim_dt =  model.opt.timestep # 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))

    
    # Top Level Controller - Controls EVERYTHING
    top_controller = TopLevelController(model=model, data=data)
    # Locomotion Controller from Trained Onnx Policy
    top_controller.locomotion_ctrl_ = Locomotion_Controller(
        locomotion_policy_path=(_ONNX_DIR / "go2_policy_galloping.onnx").as_posix(),
        # Pass only qpos regarding Go2's joints, NOT base's x,y,z,quat 
        default_angles=np.array(model.keyframe("home").qpos[top_controller.robot_go2.i_start_qpos:top_controller.robot_go2.i_end_qpos]),
        n_substeps=n_substeps,
        action_scale=0.5,
        vel_scale_x=1.5,
        vel_scale_y=0.8,
        vel_scale_rot=2 * np.pi,
        locomotion_cmd=np.zeros(3) # Define which Navigator-Boss the locomotion cotroller will have.
    )
    # TBD: Future definition of Navigation Policy
    top_controller.navigation_policy_ = NavigationPolicy(
            #TBD: FIX THOSE DEFAULT ANGELS
            default_angles=np.array(model.keyframe("home").qpos[top_controller.robot_go2.i_start_qpos:top_controller.robot_go2.i_end_qpos]), 
            # default_angles=np.array(model.keyframe("home").qpos[:]), 
            n_substeps=n_substeps,  #TBD: FIX THOSE DEFAULT N_SUBSTEPS
            action_scale=0.5,
        )

    # Init time variables
    top_controller.t_last_cmd = data.time
    top_controller.dt_new_cmd = 0.5 #0.5sec - update vel command per delta t

    # Set the main controll callback from Top Controller
    mujoco.set_mjcb_control(top_controller.control_callback)

    return model, data



if __name__ == "__main__":
  
  viewer.launch(loader=main_function)
