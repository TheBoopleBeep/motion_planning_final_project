import sys
import numpy as np
import genesis as gs
from scenes import create_scene_adjacency
from pddl_planner import plan_task, save_plan_to_file, load_plan_from_file
from motion_action_primitives import move_while_closed, move_while_open, close_gripper, open_gripper
import os
import math
from scipy.spatial.transform import Rotation as R

gripper_ee_offset = np.array([0.0027, 0.003, 0.04])

def PICK(block, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    bpos = np.asarray(b.get_pos(), dtype=np.float64)
    # offset forward because actual gripper is behind the end effector link
    #move above block
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.2]) + gripper_ee_offset)
    # #lower to block
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.07]) + gripper_ee_offset, tolerance=.008,attached_object=b,waypoints=70)
    # #move up - pass b as attached object
    move_while_closed(franka, scene, bpos + np.array([0.005, 0, 0.3])+ gripper_ee_offset, attached_object=b, tolerance=.01,waypoints=100)


def PLACE(block, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above block
    move_while_closed(franka, scene, np.array([0.3, .3, 0.23]), attached_object=b, waypoints=70)
    # #lower next to block
    move_while_closed(franka, scene,  np.array([.3, 0.3, 0.18]), attached_object=b,waypoints=70)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, bpos + np.array([0.3, .3, 0.3]),waypoints=70)

def PLACEEAST(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and east of block2, aligned with y axis
    move_while_closed(franka, scene, b2pos + np.array([0, 0.043, 0.3]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0, 0.043, 0.13]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1,ignore_collisions=True)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0, 0.05, 0.3]),attached_object=b1)

def PLACEWEST(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and west of block2, aligned with y axis
    move_while_closed(franka, scene, b2pos + np.array([0, -0.043, 0.3]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0, -0.043, 0.13]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1,ignore_collisions=True)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0, -.043, 0.3]))

def PLACENORTH(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and north of block2, aligned with x axis
    move_while_closed(franka, scene, b2pos + np.array([0.043, 0, 0.3]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0.043, 0, 0.13]), attached_object=b1,ignore_collisions=True)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0.043, 0, 0.3]))

def PLACESOUTH(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and south of block2, aligned with x axis
    move_while_closed(franka, scene, b2pos + np.array([-0.043, 0, 0.3]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([-0.043, 0, 0.13]), attached_object=b1,ignore_collisions=True)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([-.043, 0, 0.3]), attached_object=b1)

def PLACELONGDIAG(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and at long diagonal of block2
    move_while_closed(franka, scene, b2pos + np.array([0.088, 0.088, 0.23]), attached_object=b1,waypoints=200)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0.088, 0.088, 0.13]), attached_object=b1,waypoints=100,ignore_collisions=True)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0.088, 0.088, 0.3]),attached_object=b1,waypoints=50)


# TODO Added for testing
# ---------------------------
def PLACE_SPECIFIC(block, position, blocks_state, franka, scene):
    print("Place_Specific")
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    # bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.30]), attached_object=b)
    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.13]), attached_object=b, ignore_collisions=True)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]))

def PLACE_SPECIFIC_ORIENTED(block, position, quat, blocks_state, franka, scene):
    print("Place_Specific_Oriented")
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    # bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.30]), quat, attached_object=b,waypoints=200)
    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.13]), quat, attached_object=b, ignore_collisions=True, waypoints = 100)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]))

# ---------------------------

def execute_pddl_plan(plan_file, domain_file, problem_file, blocks_state, franka, scene):
    """
    Execute a PDDL plan by loading from file if it exists, or computing and saving it.
    
    Parameters:
        plan_file: Path to the JSON file for the plan
        domain_file: Path to the PDDL domain file (used if plan needs to be computed)
        problem_file: Path to the PDDL problem file (used if plan needs to be computed)
        blocks_state: Dictionary mapping block names to block entities
        franka: Robot adapter
        scene: Genesis scene
    """
    # Check if plan file exists
    if os.path.exists(plan_file):
        print(f"Loading precomputed plan from {plan_file}")
        solution = load_plan_from_file(plan_file)
    else:
        print(f"Plan file not found. Computing plan and saving to {plan_file}")
        solution = plan_task(domain_file, problem_file)
        if solution:
            save_plan_to_file(solution, plan_file)
        else:
            print("ERROR: No solution found!")
            return False
    
    if not solution:
        print("ERROR: No solution found!")
        return False
    
    # Execute each action in the plan
    for i, action in enumerate(solution, 1):
        action_str = str(action)
        
        # Parse action string (format: "action_name block1 [block2]")
        # Remove all parentheses and split by whitespace
        parts = action_str.replace('(', '').replace(')', '').split()
        action_name = parts[0].lower()
        
        if action_name == "pick":
            # pick block
            print("PICK %s" % parts[1])
            block = parts[1]
            PICK(block, blocks_state, franka, scene)
            
        elif action_name == "place":
            # place block
            print("PLACE %s" % parts[1])
            block = parts[1]

            ## TODO edited for testing
            # ---------------------------
            # pos = np.array([0.3, 0.3, 0.0]) # actual starting position of pattern
            theta_r = 0
            r = ((0.025/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180)))) / math.cos(54*(math.pi/180))
            pos = np.array([0.3+r*math.cos(theta_r), 0.3+r*math.sin(theta_r), 0.0])
            # pos = np.array([0.66, -0.4, 0.0])
            print("pos output from place: ",pos)
            quat = np.array(R.from_euler('xz', [math.pi,theta_r]).as_quat(scalar_first=True))

            PLACE_SPECIFIC_ORIENTED(block, pos, quat, blocks_state, franka, scene)

            # ---------------------------

            
        elif action_name == "placenorth":
            # placeNorth block1 block2
            print("PLACENORTH %s %s" % (parts[1], parts[2]))
            block1 = parts[1]
            block2 = parts[2]
            ## TODO edited for testing
            # ---------------------------

            theta_r = 0
            quat = np.array(R.from_euler('xz', [math.pi,theta_r]).as_quat(scalar_first=True))
            b2 = blocks_state[block2]
            b2pos = np.asarray(b2.get_pos(), dtype=np.float64)
            pos = b2pos + np.array([0.043, 0.0, 0.0])

            PLACE_SPECIFIC_ORIENTED(block1, pos, quat, blocks_state, franka, scene)

            # ---------------------------            
        elif action_name == "placeeast":
            # placeEast block1 block2
            print("PLACEEAST %s %s" % (parts[1], parts[2]))
            block1 = parts[1]
            block2 = parts[2]
            ## TODO edited for testing
            # ---------------------------

            theta_r = 0
            quat = np.array(R.from_euler('xz', [math.pi,theta_r]).as_quat(scalar_first=True))
            b2 = blocks_state[block2]
            b2pos = np.asarray(b2.get_pos(), dtype=np.float64)
            pos = b2pos + np.array([0, 0.043, 0.0])

            PLACE_SPECIFIC_ORIENTED(block1, pos, quat, blocks_state, franka, scene)

            # ---------------------------            
        elif action_name == "placesouth":
            # placeSouth block1 block2
            print("PLACESOUTH %s %s" % (parts[1], parts[2]))
            block1 = parts[1]
            block2 = parts[2]
            ## TODO edited for testing
            # ---------------------------


            theta_r = 0
            quat = np.array(R.from_euler('xz', [math.pi,theta_r]).as_quat(scalar_first=True))
            b2 = blocks_state[block2]
            b2pos = np.asarray(b2.get_pos(), dtype=np.float64)
            pos = b2pos + np.array([-0.043, 0.0, 0.0])

            PLACE_SPECIFIC_ORIENTED(block1, pos, quat, blocks_state, franka, scene)

            # ---------------------------            
        elif action_name == "placewest":
            # placeWest block1 block2
            print("PLACEWEST %s %s" % (parts[1], parts[2]))
            block1 = parts[1]
            block2 = parts[2]
            ## TODO edited for testing
            # ---------------------------

            theta_r = 0
            quat = np.array(R.from_euler('xz', [math.pi,theta_r]).as_quat(scalar_first=True))


            b2 = blocks_state[block2]
            b2pos = np.asarray(b2.get_pos(), dtype=np.float64)
            pos = b2pos + np.array([0, -0.043, 0.0])

            PLACE_SPECIFIC_ORIENTED(block1, pos, quat, blocks_state, franka, scene)

            # ---------------------------

            
        elif action_name == "placelongdiag":
            # placeLongDiag block1 block2
            block1 = parts[1]
            block2 = parts[2]
            print("PLACELONGDIAG %s %s" % (block1, block2))
            PLACELONGDIAG(block1, block2, blocks_state, franka, scene)

            
        else:
            print(f"ERROR: Unknown action: {action_name}")
    
    return True


def main():
    """
    Main function to run the adjacency problem.
    Sets up the scene, configures the robot, and executes the PDDL plan.
    """
    # Ensure Genesis is initialized before building scenes
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
    else:
        gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)

    # Build the scene using the factory
    scene, franka, BlocksState = create_scene_adjacency()
    #scene, franka, BlocksState = create_scene_stacked()

    # Set control gains
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
    # Execute PDDL plan for adjacency problem
    execute_pddl_plan('adjacency_plan.json', 'adjacency_task.pddl', 'adjacency_problem.pddl', BlocksState, franka, scene)


if __name__ == "__main__":
    main()
