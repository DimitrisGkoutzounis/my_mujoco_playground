# TopLevelController class: basically...handles everything
from NavigationPolicy import NavigationPolicy
from Locomotion_Controller import Locomotion_Controller
import numpy as np

from mujoco_playground._src.dynamic_events.arm_mujoco.src.Robot  import RobotGo2
from mujoco_playground._src.dynamic_events.arm_mujoco.src.Arm  import Arm
from mujoco_playground._src.dynamic_events.arm_mujoco.src.Perception import Perception

class TopLevelController:
    def __init__(self): #, perception_cont
        # self.context = context
        # Objects from Arm and RobotGo2
        self.robot_go2 = RobotGo2()
        self.arm = Arm()
        self.perception = Perception()  #, perception_cont

    navigation_policy_ = None
    locomotion_ctrl_ = None
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
            self.locomotion_ctrl_.update_locomotion_cmd(self.navigation_policy_.Navigator_.generate_command_norot()) 
    #################################################################

            # Reset t_last_cmd 
            self.t_last_cmd = data.time

        # Move arm
        self.arm.control_Cb(model=model, data=data)
        # Perceive
        # self.perception.get_rgbd_auto_AOI(model=model, data=data, perception_context=self.context)
        # Execute the locomotion cmd
        self.locomotion_ctrl_.exec_locomotion_control(model=model, data=data, robot=self.robot_go2)
