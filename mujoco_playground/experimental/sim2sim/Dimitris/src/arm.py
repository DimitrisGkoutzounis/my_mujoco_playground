import mujoco
import glfw
import numpy as np
import math
import time
import csv
import os

# ========== Paths ==========
XML_PATH = "../xml/scene.xml"
DATA_LOG = "data.csv"

# ========== Simulation Constants ==========
q_desired = np.array([-0.75, -1.57, 1.57, -0.37, -2.45, -2.45])
t_init = 3.0
kp = 15
freq = 2.0
Ax, Ay, Az = 0.01, 0.01, 0.05

# ========== Global Variables ==========
p0 = np.zeros(3)
q_out = np.zeros(6)
dq_out = np.zeros(6)

# ========== Load Model ==========
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ========== Logging ==========
csv_file = open(DATA_LOG, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "ep_x", "ep_y", "ep_z", "p_cx", "p_cy", "p_cz", "p_dx", "p_dy", "p_dz"])

# ========== GLFW + Visualization Setup ==========
if not glfw.init():
    raise RuntimeError("Could not initialize GLFW")

window = glfw.create_window(1244, 700, "MuJoCo Arm Control", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=2000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

# Set initial camera view
arr_view = [-90.477651, -40.102665, 3.209726, -0.047404, -0.001591, 0.330533] # [azimuth, elevation, distance, lookat_x, lookat_y, lookat_z]
cam.azimuth, cam.elevation, cam.distance = arr_view[:3]
cam.lookat[:] = arr_view[3:]

# ========== Controller ==========
def my_controller(model, data):
    global p0, q_out, dq_out

    t = data.time - t_init
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    if ee_body_id == -1:
        return

    if data.time < t_init:
        data.ctrl[:6] = q_desired
        q_out[:] = data.qpos[:6]
        p0[:] = data.xpos[ee_body_id]
    else:
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, ee_body_id)
        Jall = jacp
        J = Jall[:3, :6]  # Only the first 3 rows (linear velocity)
        # Pseudo-inverse
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        S_inv = np.array([1/s if s > 1e-2 else 0.0 for s in S])
        J_pinv = Vt.T @ np.diag(S_inv) @ U.T

        # Actual position and velocity
        p_c = data.xpos[ee_body_id]
        dp_c = data.cvel[ee_body_id][:3]  # local linear vel

        # Desired trajectory
        p_d = p0 + np.array([
            Ax * np.sin(freq * t),
            Ay * np.sin(freq * t),
            Az * np.sin(freq * t)
        ])
        dp_d = np.array([
            Ax * freq * np.cos(freq * t),
            Ay * freq * np.cos(freq * t),
            Az * freq * np.cos(freq * t)
        ])

        # Error and control
        ep = p_c - p_d
        dq_out = J_pinv @ (-kp * ep)
        q_out += dq_out * 0.002

        # Send joint commands
        data.ctrl[:6] = q_out

        # Log
        csv_writer.writerow([
            data.time,
            ep[0], ep[1], ep[2],
            p_c[0], p_c[1], p_c[2],
            p_d[0], p_d[1], p_d[2]
        ])



mouse = {
    "left": False,
    "middle": False,
    "right": False,
    "last_x": 0.0,
    "last_y": 0.0
}

def mouse_button_callback(window, button, action, mods):
    mouse["left"] = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    mouse["middle"] = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    mouse["right"] = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    mouse["mods"] = mods
    mouse["last_x"], mouse["last_y"] = glfw.get_cursor_pos(window)

def cursor_pos_callback(window, xpos, ypos):
    if not (mouse["left"] or mouse["middle"] or mouse["right"]):
        return

    dx = xpos - mouse["last_x"]
    dy = ypos - mouse["last_y"]
    mouse["last_x"] = xpos
    mouse["last_y"] = ypos

    width, height = glfw.get_window_size(window)
    dx /= height
    dy /= height

    # Determine camera action
    if mouse["right"]:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if (mouse["mods"] & glfw.MOD_SHIFT) else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif mouse["left"]:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if (mouse["mods"] & glfw.MOD_SHIFT) else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    elif mouse["middle"]:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
    else:
        return

    mujoco.mjv_moveCamera(model, action, dx, dy, scene, cam)

def scroll_callback(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

# Register callbacks
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_pos_callback)
glfw.set_scroll_callback(window, scroll_callback)


# ========== Main Loop ==========
while not glfw.window_should_close(window):
    my_controller(model, data)

    # Step simulation (60 FPS)
    sim_start = data.time
    while data.time - sim_start < 1.0 / 60.0:
        mujoco.mj_step(model, data)

    # Rendering
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, scene, context)

    # Swap + handle input
    glfw.swap_buffers(window)
    glfw.poll_events()

# ========== Cleanup ==========
csv_file.close()
glfw.terminate()