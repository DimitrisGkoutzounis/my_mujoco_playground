# TopLevelController class: basically...handles NavigationPolicy
from NavigationPolicy import NavigationPolicy
import numpy as np

class TopLevelController:
    def __init__(self, _ONNX_DIR, model):

        # Constructor NavigationPolicy
        self.Nav_Policy = NavigationPolicy(
            #TBD: FIX THOSE DEFAULT ANGELS
            default_angles= np.zeros(6),#np.array(model.keyframe("home").qpos[top_controller.robot_go2.i_start_qpos:top_controller.robot_go2.i_end_qpos]), 
            # default_angles=np.array(model.keyframe("home").qpos[:]), 
            n_substeps=5,  #TBD: FIX THOSE DEFAULT N_SUBSTEPS int(round(ctrl_dt / sim_dt))
            action_scale=0.5,

            ############### DO NOT CHANGE THOSE - START ##############
            # For Locomotion Controller
            _ONNX_DIR = _ONNX_DIR,
            model = model
            ############### DO NOT CHANGE THOSE - END ##############

            ############### ADD INIT PARAMS FOR Go2NavEnv(mjx_env.MjxEnv) ##############

        )
        print("TopLevelController: Initialized")

    def set_navigation_policy(self, Nav_Pol__):
        self.Nav_Policy = Nav_Pol__

    def control_callback(self, model, data):
        print("lala from top Cb")
        #### Placeholder for later call Nav_policy: step, reset