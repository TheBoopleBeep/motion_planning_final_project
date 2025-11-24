import sys
import warnings
import numpy as np
import genesis as gs
from scenes import create_scene_6blocks, create_scene_stacked
from pddl_planner import plan_task


def move_while_closed(franka, scene, pos, ignore_collisions=True, max_retries=30):
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    
    # get the end-effector link
    end_effector = franka.get_link("hand")
    
    # Get current gripper position to maintain during movement
    current_gripper_pos = franka.get_qpos()[-2:]
    
    # Retry loop for IK and path planning
    for attempt in range(max_retries):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=np.array([0, 1, 0, 0]),
        )
        
        # Check if IK succeeded
        if qpos is None:
            print(f"Attempt {attempt+1}/{max_retries}: IK failed for position {pos}")
            continue
        
        # Set gripper to current closed position
        qpos[-2:] = current_gripper_pos
        
        path = franka.plan_path(qpos_goal=qpos, num_waypoints=200, ignore_collisions=ignore_collisions, timeout=10.0)
        
        # Check if path planning succeeded
        if path is None or len(path) == 0:
            print(f"Attempt {attempt+1}/{max_retries}: Path planning failed for position {pos}")
            continue
        
        # Validate path before executing - check if final path qpos matches goal qpos
        final_qpos = path[-1]
        qpos_diff = np.linalg.norm(final_qpos[:7] - qpos[:7])
        
        if qpos_diff > 0.1:
            print(f"Attempt {attempt+1}/{max_retries}: Path validation failed - final qpos differs from goal by {qpos_diff:.3f}")
            continue
        
        # Try to execute the path - if it fails due to bad path, retry
        try:
            # First, ensure gripper is firmly closed before moving
            current_arm_pos = franka.get_qpos()[:-2]
            for _ in range(5):
                franka.control_dofs_position(current_arm_pos, motors_dof)
                franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
                scene.step()
            
            for i, waypoint in enumerate(path):
                # Control arm with position from waypoint
                franka.control_dofs_position(waypoint[:-2], motors_dof)
                # Apply strong closing force to maintain grip
                franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
                scene.step()
            
            # Hold position briefly at the end to stabilize grip
            final_arm_pos = path[-1][:-2]
            for _ in range(10):
                franka.control_dofs_position(final_arm_pos, motors_dof)
                franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
                scene.step()
            
            # Final position check
            actual_final_pos = end_effector.get_pos()
            final_dist = np.linalg.norm(actual_final_pos - pos)
            if final_dist > 0.15:
                print(f"\n!!! Final position check failed - retrying !!!")
                print(f"Target: {pos}, Actual: {actual_final_pos}, Distance: {final_dist:.3f}m")
                raise ValueError("Bad path - final position too far from target")
            
            # Success!
            return
            
        except ValueError as e:
            # Path was bad, continue to next retry attempt
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            continue
    
    # If we get here, all retries failed
    print(f"ERROR: Failed to find valid path after {max_retries} attempts for position {pos}")
    raise RuntimeError(f"Path planning failed for position {pos}")

def move_while_open(franka, scene, pos, max_retries=10):
    # get the end-effector link
    end_effector = franka.get_link("hand")
    
    # Retry loop for IK and path planning
    for attempt in range(max_retries):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=np.array([0, 1, 0, 0]),
        )
        
        if qpos is None:
            print(f"Attempt {attempt+1}/{max_retries}: IK failed for position {pos}")
            continue
            
        qpos[-2:] = 0.04  # gripper open
        
        path = franka.plan_path(qpos_goal=qpos, num_waypoints=200, timeout=10.0)
        
        if path is None or len(path) == 0:
            print(f"Attempt {attempt+1}/{max_retries}: Path planning failed for position {pos}")
            continue
        
        # Validate path before executing - check if final path qpos matches goal qpos
        # Attempts to fix weird bug where the arm moves back to the home position
        final_qpos = path[-1]
        qpos_diff = np.linalg.norm(final_qpos[:7] - qpos[:7])
        
        if qpos_diff > 0.1:
            print(f"Attempt {attempt+1}/{max_retries}: Path validation failed - final qpos differs from goal by {qpos_diff:.3f}")
            continue
        
        # Try to execute the path - if it fails due to bad path, retry
        try:
            for i, waypoint in enumerate(path):
                franka.control_dofs_position(waypoint)
                scene.step()
            
            # Final position check
            actual_final_pos = end_effector.get_pos()
            final_dist = np.linalg.norm(actual_final_pos - pos)
            if final_dist > 0.15:
                print(f"\n!!! Final position check failed - retrying !!!")
                print(f"Target: {pos}, Actual: {actual_final_pos}, Distance: {final_dist:.3f}m")
                raise ValueError("Bad path - final position too far from target")
            
            # Success!
            return
            
        except ValueError as e:
            # Path was bad, continue to next retry attempt
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            continue
    
    # If we get here, all retries failed
    # Genesis kind of sucks
    print(f"ERROR: Failed to find valid path after {max_retries} attempts for position {pos}")
    raise RuntimeError(f"Path planning failed for position {pos}")

def close_gripper(franka, scene):
    # Close gripper with force control
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    
    # Hold arm position steady
    arm_qpos = franka.get_qpos()[:-2]
    
    for i in range(150):
        # Keep arm steady, apply closing force to fingers
        franka.control_dofs_position(arm_qpos, motors_dof)
        franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
        scene.step()

def open_gripper(franka, scene):
    # Open gripper with position control
    qpos = franka.get_qpos()
    qpos[-2:] = 0.04  # Open position
    
    for i in range(40):
        # Control all DOFs together with position control
        franka.control_dofs_position(qpos)
        scene.step()


def STACK(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above block2
    move_while_closed(franka, scene, b2pos + np.array([0, 0, 0.23]))
    # #lower to block2
    move_while_closed(franka, scene, b2pos + np.array([0, 0, 0.17]))
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0, 0, 0.3]))
    
def PICK(block, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above block
    move_while_open(franka, scene, bpos + np.array([0, 0, 0.18]))
    # #lower to block
    move_while_open(franka, scene, bpos + np.array([0, 0, 0.1]))
    # #grasp
    close_gripper(franka, scene)
    # #move up
    move_while_closed(franka, scene, bpos + np.array([0, 0, 0.3]))


def execute_pddl_plan(domain_file, problem_file, blocks_state, franka, scene):
    """
    Takes PDDL domain and problem files, generates a plan, and executes it.
    
    Parameters:
        domain_file: Path to PDDL domain file
        problem_file: Path to PDDL problem file
        blocks_state: Dictionary mapping block names to block entities
        franka: Robot adapter
        scene: Genesis scene
    """
    # Generate plan using PDDL planner
    solution = plan_task(domain_file, problem_file)
    
    if not solution:
        print("ERROR: No solution found!")
        return False
    
    # Execute each action in the plan
    for i, action in enumerate(solution, 1):
        action_str = str(action)
        
        # Parse action string (format: "action_name block1 [block2]")
        # Remove all parentheses and split by whitespace
        parts = action_str.replace('(', '').replace(')', '').split()
        action_name = parts[0]
        
        if action_name == "pick":
            # pick block
            block = parts[1]
            PICK(block, blocks_state, franka, scene)
            
        elif action_name == "stack":
            # stack block1 block2
            block1 = parts[1]
            block2 = parts[2]
            STACK(block1, block2, blocks_state, franka, scene)
            
        elif action_name == "unstack":
            # unstack block1 block2 (pick block1 from block2)
            block1 = parts[1]
            block2 = parts[2]
            PICK(block1, blocks_state, franka, scene)
            
        elif action_name == "place":
            # place block (not used in typical tower building)
            block = parts[1]
            print(f"WARNING: Place action not fully implemented for block {block}")
            
        else:
            print(f"ERROR: Unknown action: {action_name}")
    
    return True
    

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

# Execute PDDL plan
execute_pddl_plan('2_towers_domain.pddl', '2_towers_problem.pddl', BlocksState, franka, scene)


