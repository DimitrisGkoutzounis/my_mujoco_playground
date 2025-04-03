# TopLevelController class: basically...handles NavigationPolicy
from NavigationPolicy import NavigationPolicy
import numpy as np

class TopLevelController:
    def __init__(self, n_substeps, _ONNX_DIR, model):

        # Constructor NavigationPolicy
        self.Nav_Policy = NavigationPolicy(
            #TBD: FIX THOSE DEFAULT ANGELS
            default_angles= np.zeros(6),#np.array(model.keyframe("home").qpos[top_controller.robot_go2.i_start_qpos:top_controller.robot_go2.i_end_qpos]), 
            # default_angles=np.array(model.keyframe("home").qpos[:]), 
            n_substeps=n_substeps,  #TBD: FIX THOSE DEFAULT N_SUBSTEPS
            action_scale=0.5,
            # For Locomotion Controller
            _ONNX_DIR = _ONNX_DIR,
            model = model
        )
        print("TopLevelController: Initialized")

    def set_navigation_policy(self, Nav_Pol__):
        self.Nav_Policy = Nav_Pol__

    def control_callback(self, model, data):
        print("lala from top Cb")
    #     # Set new vel cmd per dt_new_cmd
    #     if (data.time - self.t_last_cmd) >= self.dt_new_cmd:
    #         print("New cmd at:",data.time)

    #         # Update locomotion cmd, from whatever(in random) the Navigator decides
    #         # TBD: replace navigator from you policy decision
    # #################################################################
    #         # SOS HERE UNCOMMENT THIS TO MOVE Go2
    #         # self.locomotion_ctrl_.update_locomotion_cmd(self.Nav_Policy.Navigator.generate_command_norot()) 
    # #################################################################


    #         # Reset t_last_cmd 
    #         self.t_last_cmd = data.time

        # Move arm
        # self.arm.control_Cb(model=model, data=data)
        # Perceive
        # self.perception.get_rgbd_auto_AOI(model=model, data=data, perception_context=self.context)
        # Execute the locomotion cmd
        # self.locomotion_ctrl_.exec_locomotion_control(model=model, data=data, robot=self.robot_go2)
