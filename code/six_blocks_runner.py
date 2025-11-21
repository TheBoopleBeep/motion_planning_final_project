import sys
import warnings
import numpy as np
import genesis as gs
from scenes import create_scene_6blocks, create_scene_stacked

# Suppress numpy deprecation warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

def move_while_closed(franka, scene, pos, ignore_collisions=True):
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    
    # get the end-effector link
    end_effector = franka.get_link("hand")
    
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=pos,
        quat=np.array([0, 1, 0, 0]),
    )
    
    # Check if IK succeeded
    if qpos is None:
        print(f"Warning: IK failed for position {pos}")
        return
    
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)
    
    # Check if path planning succeeded
    if path is None or len(path) == 0:
        print(f"Warning: Path planning failed for position {pos}")
        return
    
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof) 
        franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
        scene.step()

def move_while_open(franka, scene, pos):
    # get the end-effector link
    end_effector = franka.get_link("hand")
    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=pos,
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04  # gripper open
    
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)
    
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

def close_gripper(franka,scene):
    # grasp
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    # get the end-effector link
    end_effector = franka.get_link("hand")
    # reach
    qpos = franka.get_qpos()
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

    for i in range(40):
        scene.step()

def open_gripper(franka,scene):
    # open gripper
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    # get the end-effector link
    end_effector = franka.get_link("hand")
    # reach
    qpos = franka.get_qpos()
    qpos[-2:] = 0.04 
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_position(qpos[-2:], fingers_dof)
    for i in range(40):
        scene.step()


def STACK(block1, block2, blocks_state, franka,scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above block2
    move_while_closed(franka, scene, b2pos + np.array([0, 0, 0.23]))
    # #lower to block2
    move_while_closed(franka, scene, b2pos + np.array([0, 0, 0.19]))
    # #release
    open_gripper(franka,scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0, 0, 0.3]))
    
def PICK(block, blocks_state, franka,scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above block
    move_while_open(franka, scene, bpos + np.array([0, 0, 0.13]))
    # #lower to block
    move_while_open(franka, scene, bpos + np.array([0, 0, 0.08]))
    # #grasp
    close_gripper(franka,scene)
    # #move up
    move_while_closed(franka, scene, bpos + np.array([0, 0, 0.3]))
    

# Ensure Genesis is initialized before building scenes
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)

# build the scene using the factory
scene, franka, BlocksState = create_scene_6blocks()
#scene, franka, BlocksState = create_scene_stacked()

# set control gains
# Note: the following values are tuned for achieving best behavior with Franka
# Typically, each new robot would have a different set of parameters.
# Sometimes high-quality URDF or XML file would also provide this and will be parsed.

franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)
PICK("g", BlocksState, franka,scene)
STACK("g", "r", BlocksState, franka,scene)
PICK("b", BlocksState, franka,scene)
STACK("b", "g", BlocksState, franka,scene)
PICK("y", BlocksState, franka,scene)
STACK("y", "b", BlocksState, franka,scene)
PICK("m", BlocksState, franka,scene)
STACK("m", "y", BlocksState, franka,scene)
PICK("c", BlocksState, franka,scene)
STACK("c", "m", BlocksState, franka,scene)

