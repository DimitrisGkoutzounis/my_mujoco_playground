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


import numpy as np
import time
from mujoco_playground.experimental.sim2sim.Dimitris.LocomotionPolicy import LocomotionPolicy
from mujoco_playground.experimental.sim2sim.Dimitris.TrajectoryGenerator import TrajectoryGenerator

class TrajectoryPolicy:
    def __init__(
        self,
        n_substeps: int = 5,
        locomotion_policy: LocomotionPolicy = None,
        trajectory_generator: TrajectoryGenerator = None,
    ):
        self._counter = 0
        self._n_substeps = n_substeps
        self.locomotion_policy = locomotion_policy
        self.trajectory_generator = trajectory_generator
        self.target = None
        self.init_pos = None
        self.current_pos = None

        self.last_update_time = time.time()  # initialize properly
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0

    def update_control(self):
        # Generate random velocity
        self.generate_velocity() #sets a random veliocity --> change with a new trajectory
        
        self.locomotion_policy.set_cmd_vel(self.cmd_vel_x, self.cmd_vel_y) #updates the random velocity --> change with the required velocities needed to trajectory

    def update_trajectory_control(self):
        
        self.locomotion_policy.set_cmd_vel(self.cmd_vel_x, self.cmd_vel_y)
        
    def random_controller(self, model, data):
        """Set random velocity commands"""
        current_time = time.time()


        if (current_time - self.last_update_time) >= 2.0: # 2 second has passed
            self.update_control()
            self.last_update_time = current_time
        # Always apply control
        if self.locomotion_policy:
            self.locomotion_policy.get_control(model, data)


    def trajectory_controller(self, model, data):
        '''guides the robot to follow a trajectory'''
        #set the control loop
        if self.locomotion_policy:
            self.locomotion_policy.get_control(model, data)
        
        current_time = time.time()
        #retrun x,y,yaw list
        self.set_current_pos(data)
        
        #compute the error from the target
        
        if(self.target is None):
            """Set initial target"""
            self.set_target()
        
        error_pos = np.linalg.norm(self.current_pos - self.target)
        if error_pos < 0.5:
            # Generate new trajectory
            self.set_target()
        
        else:
            # Compute the desired velocity
            self.cmd_vel_x, self.cmd_vel_y = self.compute_velocity(data)
            print("Target:", self.target)
            self.update_trajectory_control()
            
            
            
    def compute_velocity(self,data):
        """Simple PD controller to compute velocity command towards the target."""

        # Get current position and velocity from locomotion policy
        
        Kp = 1.0
        Kd = 0.2

        # Position error
        pos_error = np.array(self.target) - self.current_pos

        # Velocity command using PD
        vel_cmd = Kp * pos_error

        # Optional: Clip velocities to [-1, 1]
        vel_cmd = np.clip(vel_cmd, -1.0, 1.0).flatten()
        
        #set the cmd_vel
        self.cmd_vel_x = vel_cmd[0]
        self.cmd_vel_y = vel_cmd[1]

        return vel_cmd[0], vel_cmd[1]
    
    
    def generate_velocity(self):
        
        self.cmd_vel_x = np.around(np.random.uniform(-1, 1), 3)
        self.cmd_vel_y = np.around(np.random.uniform(-1, 1), 3)
        
    def set_target(self):
        # Set the target for the locomotion policy
        self.target = np.array(self.trajectory_generator._generate_simple_target())
        
    def set_current_pos(self,data):
        # Set the current position for the locomotion policy
        self.current_pos = np.array(self.locomotion_policy.current_pos(data))
        xy_cords = self.current_pos[:2]
        self.current_pos = xy_cords

    def get_velocity(self):
        return self.cmd_vel_x, self.cmd_vel_y