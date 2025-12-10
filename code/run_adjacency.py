import sys
import numpy as np
import genesis as gs
from scenes import create_scene_adjacency
from pddl_planner import plan_task, save_plan_to_file, load_plan_from_file
from motion_action_primitives import move_while_closed, move_while_open, close_gripper, open_gripper
from state_estimation import StateEstimation
import os
GRIPPER_EE_OFFSET = np.array([0.00, 0.003, 0.04])

def PICK(block, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    bpos = np.asarray(b.get_pos(), dtype=np.float64)
    # offset forward because actual gripper is behind the end effector link
    #move above block
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.2]) + GRIPPER_EE_OFFSET, tolerance = .01)
    # #lower to block
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.07]) + GRIPPER_EE_OFFSET, tolerance=.001,attached_object=b)
    # #move up - pass b as attached object
    move_while_closed(franka, scene, bpos + np.array([0.005, 0, 0.3])+ GRIPPER_EE_OFFSET, attached_object=b, tolerance=.001)


def PLACE(block, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above block
    move_while_closed(franka, scene, np.array([0.4, .4, 0.23]), attached_object=b)
    # #lower next to block
    move_while_closed(franka, scene,  np.array([.4, 0.4, 0.18]), attached_object=b)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, bpos + np.array([0.4, .4, 0.3]))

def PLACEEAST(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and east of block2, aligned with y axis
    move_while_closed(franka, scene, b2pos + np.array([0, 0.043, 0.35]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0, 0.043, 0.13]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1,)
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
    move_while_closed(franka, scene, b2pos + np.array([0, -0.043, 0.35]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0, -0.043, 0.13]), quat=np.array([0, -0.7071, 0.7071, 0]), attached_object=b1,)
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
    move_while_closed(franka, scene, b2pos + np.array([0.043, 0, 0.35]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0.043, 0, 0.13]), attached_object=b1)
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
    move_while_closed(franka, scene, b2pos + np.array([-0.043, 0, 0.35]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([-0.043, 0, 0.13]), attached_object=b1)
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
    move_while_closed(franka, scene, b2pos + np.array([0.088, 0.088, 0.23]), attached_object=b1)
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0.088, 0.088, 0.13]), attached_object=b1)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0.088, 0.088, 0.3]),attached_object=b1)


def execute_action(action_str, blocks_state, franka, scene):
    """
    Execute a single PDDL action for adjacency task.
    
    Parameters:
        action_str: Action string (e.g., "(pick b1)" or "(placeNorth b1 b2)")
        blocks_state: Dictionary mapping block names to block entities
        franka: Robot adapter
        scene: Genesis scene
    
    Returns:
        dict with 'success' key indicating if action executed successfully
    """
    # Parse action string
    parts = action_str.replace('(', '').replace(')', '').split()
    action_name = parts[0].lower()
    
    print(f"Executing: {action_str}")
    
    try:
        if action_name == "pick":
            PICK(parts[1], blocks_state, franka, scene)
        elif action_name == "place":
            PLACE(parts[1], blocks_state, franka, scene)
        elif action_name == "placenorth":
            PLACENORTH(parts[1], parts[2], blocks_state, franka, scene)
        elif action_name == "placeeast":
            PLACEEAST(parts[1], parts[2], blocks_state, franka, scene)
        elif action_name == "placesouth":
            PLACESOUTH(parts[1], parts[2], blocks_state, franka, scene)
        elif action_name == "placewest":
            PLACEWEST(parts[1], parts[2], blocks_state, franka, scene)
        elif action_name == "placelongdiag":
            PLACELONGDIAG(parts[1], parts[2], blocks_state, franka, scene)
        else:
            print(f"ERROR: Unknown action: {action_name}")
            return {'success': False}
        
        return {'success': True}
        
    except Exception as e:
        print(f"✗ Action failed: {action_str}")
        print(f"   Error: {e}")
        return {'success': False}


def execute_with_replanning(domain_file, blocks_state, franka, scene, max_iterations=50):
    """
    Execute adjacency task with replanning at every step.
    
    Parameters:
        domain_file: Path to the PDDL domain file
        blocks_state: Dictionary mapping block names to block entities
        franka: Robot adapter
        scene: Genesis scene
        max_iterations: Maximum number of actions to execute (safety limit)
    
    Returns:
        True if goal was achieved, False otherwise
    """
    estimator = StateEstimation("adjacency")
    
    # Initialize block info once
    estimator.set_scene_info(franka, blocks_state)
    
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        print(f"\nReplanning (iteration {iteration})...")
        
        # Replan from current state (always replan since no state update functions)
        plan = estimator.replan_from_current_state(franka, blocks_state, domain_file)
        
        # Check planner result
        if plan is None:
            print("\n✗ Planner failed to find a solution")
            return False
        elif len(plan) == 0:
            print("\nGoal achieved!")
            return True
        else:
            print(f"Plan generated with {len(plan)} actions")
        
        # Execute first action from plan
        action = plan[0]
        result = execute_action(action, blocks_state, franka, scene)
        
        if not result['success']:
            print("\nAction execution failed")
            return False
        
        # Wait for scene to settle
        for _ in range(10):
            scene.step()
        
        # Update and print current state
        estimator.set_scene_info(franka, blocks_state)
        current_state = estimator.parse_state_to_pddl()
        print("\nCurrent estimated state:")
        for fact in current_state:
            print(f"  {fact}")
    
    print(f"\nMaximum iterations ({max_iterations}) reached")
    return False


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
    
    # Execute with replanning at every step
    success = execute_with_replanning(
        domain_file='adjacency/adjacency_task.pddl',
        blocks_state=BlocksState,
        franka=franka,
        scene=scene,
        max_iterations=50
    )
    
    if success:
        print("\nTask completed successfully!")
    else:
        print("\nTask failed to complete")


if __name__ == "__main__":
    main()
