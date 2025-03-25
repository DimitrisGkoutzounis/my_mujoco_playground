# TopLevelController class: basically...handles everything
from NavigationPolicy import NavigationPolicy
from Locomotion_Controller import Locomotion_Controller
import numpy as np

class TopLevelController:
    def __init__(self):
        pass
    
    navigation_policy_ = None
    locomotion_ctrl_ = None
    t_last_cmd = 0
    dt_new_cmd = 1.0

    def control_callback(self, model, data):
        # nonlocal self.t_last_cmd

        # Set new vel cmd per dt_new_cmd
        if (data.time - self.t_last_cmd) >= self.dt_new_cmd:
            print("New cmd at:",data.time)
            # Update locomotion cmd, from whatever(in random) the Navigator decides
            # TBD: replace navigator from you policy decision
            self.locomotion_ctrl_.update_locomotion_cmd(self.navigation_policy_.Navigator_.generate_command_norot()) 
            # Reset t_last_cmd 
            self.t_last_cmd = data.time

        # Execute the locomotion cmd
        self.locomotion_ctrl_.exec_locomotion_control(model=model, data=data)