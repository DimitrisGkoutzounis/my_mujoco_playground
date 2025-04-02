# TopLevelController class: basically...handles NavigationPolicy
from NavigationPolicy import NavigationPolicy
import numpy as np

class TopLevelController:
    def __init__(self):
        self.navigation_policy_ = NavigationPolicy()

    navigation_policy_ = None
    t_last_cmd = 0
    dt_new_cmd = 1.0

    def control_callback(self, model, data):
    
        # Set new vel cmd per dt_new_cmd
        if (data.time - self.t_last_cmd) >= self.dt_new_cmd:
            print("New cmd at:",data.time)
            # Update locomotion cmd, from whatever(in random) the Navigator decides
            # TBD: replace navigator from you policy decision
    #################################################################
            # SOS HERE UNCOMMENT THIS TO MOVE Go2
            # self.locomotion_ctrl_.update_locomotion_cmd(self.navigation_policy_.Navigator_.generate_command_norot()) 
    #################################################################

            # Reset t_last_cmd 
            self.t_last_cmd = data.time

        # Move arm
        # self.arm.control_Cb(model=model, data=data)
        # Perceive
        # self.perception.get_rgbd_auto_AOI(model=model, data=data, perception_context=self.context)
        # Execute the locomotion cmd
        # self.locomotion_ctrl_.exec_locomotion_control(model=model, data=data, robot=self.robot_go2)
