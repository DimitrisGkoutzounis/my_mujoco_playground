import numpy as np

# Navigator Class: Decides nav. cmds - Generates random navigation commands
class Navigator:
    def __init__(self, vel_scale_x=1.5, vel_scale_y=0.8, vel_scale_rot=2*np.pi):
        self.vel_scale_x=vel_scale_x
        self.vel_scale_y=vel_scale_y
        self.vel_scale_rot=vel_scale_rot
        self.command = np.zeros(3)

    # Random x,y,rotation
    def generate_command(self):
        # Random vel
        self.vel_scale_x = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_y = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_rot = np.around(np.random.uniform(-1, 1)*2*np.pi,3) # Random between 0-2pi, 3 dec
        # Pass it to command
        self.command = np.zeros(3)
        self.command[0] = self.vel_scale_x
        self.command[1] = self.vel_scale_y
        self.command[2] = self.vel_scale_rot

        return self.command
    
    # Random x,y: no rotation
    def generate_command_norot(self):
        # Random vel
        self.vel_scale_x = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_y = np.around(np.random.uniform(-1, 1),3) # Random between 0-1, 3 dec.
        self.vel_scale_rot = 0.0
        # Pass it to command
        self.command = np.zeros(3)
        self.command[0] = self.vel_scale_x
        self.command[1] = self.vel_scale_y
        self.command[2] = self.vel_scale_rot
        print("From generate_command_norot",self.command)
        return self.command