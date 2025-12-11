import sys
import numpy as np
import genesis as gs
import os
from scenes import create_scene_special_tower
from fast_downward_planner import plan_task, load_plan_from_file, save_plan_to_file
from motion_action_primitives import move_while_closed, move_while_open, close_gripper, open_gripper
from state_estimation import StateEstimation
import math
from scipy.spatial.transform import Rotation as R


# Constants
GRIPPER_EE_OFFSET = np.array([0.00, 0.003, 0.04])

# Global state for stacking
baseblock = None
stacked_blocks = 0


def STACK(block1, block2, blocks_state, franka, scene):
    """
    Stack block1 on top of block2.
    Alternates orientation between stacks for stability.
    """
    global baseblock, stacked_blocks
    
    # Get block entities and positions
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)
    
    # Alternate orientation: 45-degree rotation on even stacks, normal on odd
    quat = np.array([0, -0.921, -0.390, 0]) if stacked_blocks % 2 == 0 else np.array([0, 1, 0, 0])
    
    # Track base block position for alignment
    if baseblock is None:
        baseblock = b2
    
    basepos = np.asarray(baseblock.get_pos(), dtype=np.float64)
    
    # Update base position if target block has moved significantly
    if np.linalg.norm(b2pos - basepos) > 0.1:
        basepos = np.asarray(b2.get_pos(), dtype=np.float64)
    
    # Move above target block
    move_while_closed(franka, scene, np.array([basepos[0], basepos[1], b2pos[2] + 0.2]), 
                     attached_object=b1, waypoints=-1, quat=quat)
    for _ in range(30):
        scene.step()
    
    # Calculate final placement position with alignment correction
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    diff = basepos - b1pos
    
    if stacked_blocks % 2 == 0:
        # 45-degree rotation placement
        des_pos = np.array([diff[0] + basepos[0] - 0.000, diff[1] + basepos[1] - 0.00323, b2pos[2] + 0.111]) + GRIPPER_EE_OFFSET
    else:
        # Normal placement
        des_pos = np.array([diff[0] + basepos[0] - 0.001265, diff[1] + basepos[1] - 0.00327, b2pos[2] + 0.111]) + GRIPPER_EE_OFFSET
    
    # Move to final position and place
    move_while_closed(franka, scene, des_pos, tolerance=0.001, attached_object=b1, waypoints=40, quat=quat)
    
    # Release block
    open_gripper(franka, scene)
    for _ in range(20):
        scene.step()
    
    # Move gripper up
    move_while_open(franka, scene, b2pos + np.array([0, 0, 0.25]), waypoints=20)
    stacked_blocks += 1
    
def PICK(block, blocks_state, franka, scene):
    """
    Pick up a block from its current position.
    """
    b = blocks_state[block]
    bpos = np.asarray(b.get_pos(), dtype=np.float64)
    
    # Move above block (offset accounts for gripper geometry)
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.25]) + GRIPPER_EE_OFFSET, waypoints=-1, waypoint_scale=.5)
    
    # Lower to block
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.07]) + GRIPPER_EE_OFFSET, 
                   tolerance=0.001, attached_object=b, waypoints=30)
    
    # Grasp block
    close_gripper(franka, scene)
    
    # Lift block
    move_while_closed(franka, scene, bpos + np.array([0.005, 0, 0.25]) + GRIPPER_EE_OFFSET, 
                     attached_object=b, tolerance=0.01, waypoints=80)

def PLACE(block, blocks_state, franka, scene):
    """
    Place a held block onto the table at a designated location.
    """
    b = blocks_state[block]
    bpos = np.asarray(b.get_pos(), dtype=np.float64)
    
    # Move to placement location above table
    move_while_closed(franka, scene, np.array([0.3, 0.3, 0.23]), attached_object=b, waypoints=70)
    
    # Lower to table
    move_while_closed(franka, scene, np.array([0.3, 0.3, 0.18]), attached_object=b, waypoints=70)
    
    # Release block
    open_gripper(franka, scene)
    
    # Move gripper up
    move_while_open(franka, scene, bpos + np.array([0.3, 0.3, 0.3]), waypoints=70)

def open_gripper_slight(franka, scene):
    """
    Open the gripper with position control while keeping arm stable.
    
    Parameters:
        franka: Robot adapter
        scene: Genesis scene
    """
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    q_init = np.array(franka.get_qpos(), dtype=np.float64)
    target_gripper_pos = np.array([0.025, 0.025], dtype=np.float64)
    
    # Stabilize arm position
    for _ in range(5):
        franka.control_dofs_position(q_init[:7], motors_dof)
        franka.control_dofs_position(q_init[-2:], fingers_dof)
        scene.step()
    
    # Open gripper to target position
    for _ in range(30):
        franka.control_dofs_position(q_init[:7], motors_dof)
        franka.control_dofs_position(target_gripper_pos, fingers_dof)
        scene.step()


def PLACE_SPECIFIC_ORIENTED(block, blocks_state, franka, scene):
    print("Place_Specific_Oriented")
    b = blocks_state[block]

    r = ((0.04/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180)))) / math.cos(54*(math.pi/180))
    theta_block = 0
    position = np.array([0.3+r*math.cos(theta_block), 0.3+r*math.sin(theta_block), 0.0])
    quat = np.array(R.from_euler('xyz', [math.pi,0.0,theta_block]).as_quat(scalar_first=True))


    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.25]), quat=quat, attached_object=b)

    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.13]), quat=quat, attached_object=b)

    # #release
    open_gripper_slight(franka, scene)

    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]), quat=quat, attached_object=b)

    open_gripper(franka, scene)

def PLACE_RELATIVE_ORIENTED(block1, block2, blocks_state, franka, scene):
    print("Place_Specific_Oriented")
    place_b = blocks_state[block1]
    ref_b = blocks_state[block2]

    [position, quat] = compute_relative(ref_b, False)
    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.25]), quat=quat, attached_object=place_b)

    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.13]), quat=quat, attached_object=place_b)

    # #release
    open_gripper_slight(franka, scene)

    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]), quat=quat, attached_object=place_b)

    open_gripper(franka, scene)

def STACK_RELATIVE_ORIENTED(block1, block2, blocks_state, franka, scene):
    print("Place_Specific_Oriented")
    place_b = blocks_state[block1]
    ref_b = blocks_state[block2]

    [position, quat] = compute_relative(ref_b, True)

    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.25]), quat=quat, attached_object=place_b)

    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.13]), quat=quat, attached_object=place_b)

    # #release
    open_gripper_slight(franka, scene)

    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]), quat=quat, attached_object=place_b)

    open_gripper(franka, scene)

def compute_relative(ref_block_state, stack=False):
    """
    Take a reference block and compute the relative position and 
    orientation the associated block should be placed in
    
    Parameters:
        ref_block_state: block state of the reference block

    Returns:
        target_position: the target position of the new block np.array([x,y,z])
        target_quat: quaternion for the place block np.array([w,x,y,z])
    """

    ref_block_pos = np.asarray(ref_block_state.get_pos(), dtype=np.float64)

    ref_block_quat = np.asarray(ref_block_state.get_quat(), dtype=np.float64)

    # Calculate position and quaternion. See Jason's notes if curious on how.
    block_width = 0.04 # adjustable value to create extra space if desired
    w = ((block_width/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180))))
    theta_dif = 72*(math.pi/180)  # 72 degrees in radians
    r =  (w/2) / math.sin(theta_dif/2)

    theta2 = math.acos((w/2)/r)*(180/math.pi)
    beta = R.from_quat(ref_block_quat, scalar_first=True).as_euler('xyz')[2]
    beta = beta*(180/math.pi)
    if beta < 0:
        beta = 360 + beta # To account for angles considered negative.

    print("beta: ",beta)
    
    if (beta-theta2) < 90:
        lamb = 180 - beta
        phi = 180 - lamb - theta2
        x_dif = -math.cos(phi*(math.pi/180))*w
        y_dif = -math.sin(phi*(math.pi/180))*w

    elif (beta-theta2) < 180:
        lamb = 270 - beta
        phi = 180 - lamb - theta2
        x_dif = math.sin(phi*(math.pi/180))*w
        y_dif = -math.cos(phi*(math.pi/180))*w

    elif (beta-theta2) < 270:
        lamb = 360 - beta
        phi = 180 - lamb - theta2
        x_dif = math.cos(phi*(math.pi/180))*w
        y_dif = math.sin(phi*(math.pi/180))*w

    elif (beta-theta2) < 360:
        lamb = 450 - beta
        phi = 180 - lamb - theta2
        x_dif = -math.sin(phi*(math.pi/180))*w
        y_dif = math.cos(phi*(math.pi/180))*w

    else:
        raise Exception("beta: ", beta, " - theta2: ", theta2," is an unnexpected value.")
    
    # print("beta: ",beta," - theta2: ",theta2," = ",beta-theta2)
    # print("phi: ",phi, " lambda: ",lamb," beta: ",beta)
    # print("x_dif: ",x_dif," y_dif: ",y_dif)

    # If not stacking block onto the second layer
    if stack == False:
        target_position = np.array([ref_block_pos[0]+x_dif, ref_block_pos[1]+y_dif, ref_block_pos[2]])

        # print("target_position for place_relative_oriented: ",target_position)
        target_quat = np.array(R.from_euler('xyz', [math.pi,0.0,theta_dif]).as_quat(scalar_first=True))
        # print("target_quat before: ",target_quat)

    else: # block is stacking onto the second layer. Increase height and adjust values
        target_position = np.array([ref_block_pos[0]+(x_dif/2), ref_block_pos[1]+(y_dif/2), ref_block_pos[2]+0.04])

        print("target_position for place_relative_oriented: ",target_position)
        target_quat = np.array(R.from_euler('xyz', [math.pi,0.0,theta_dif/2]).as_quat(scalar_first=True)) 
        # print("target_quat before: ",target_quat)

    # quaternion multiplication to combine orientations
    target_quat[0] = ref_block_quat[0] * target_quat[0] - ref_block_quat[1] * target_quat[1] - ref_block_quat[2] * target_quat[2] - ref_block_quat[3] * target_quat[3]
    target_quat[1] = ref_block_quat[0] * target_quat[1] + ref_block_quat[1] * target_quat[0] + ref_block_quat[2] * target_quat[3] - ref_block_quat[3] * target_quat[2]
    target_quat[2] = ref_block_quat[0] * target_quat[2] - ref_block_quat[1] * target_quat[3] + ref_block_quat[2] * target_quat[0] + ref_block_quat[3] * target_quat[1]
    target_quat[3] = ref_block_quat[0] * target_quat[3] + ref_block_quat[1] * target_quat[2] - ref_block_quat[2] * target_quat[1] + ref_block_quat[3] * target_quat[0]

    return [target_position, target_quat]


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
        elif action_name == "place_specific_oriented":
            PLACE_SPECIFIC_ORIENTED(parts[1], blocks_state, franka, scene)
        elif action_name == "place_relative_oriented":
            PLACE_RELATIVE_ORIENTED(parts[1], parts[2], blocks_state, franka, scene)
        elif action_name == "stack_relative_oriented":
            STACK_RELATIVE_ORIENTED(parts[1], parts[2], blocks_state, franka, scene)
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
    estimator = StateEstimation("special_tower")
    
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


def replan_tower(franka, blocks_state, domain_file, output_problem_file=None):
    """
    Replan from the current tower state using Fast Downward.
    
    Parameters:
        franka: Robot adapter
        blocks_state: Dictionary mapping block names to block entities
        domain_file: Path to the PDDL domain file (e.g., 'ten_blocks/10_block_domain.pddl')
        output_problem_file: Optional path to save the generated problem file
                            If None, a temporary file will be created and deleted
    
    Returns:blocks_state, fr
        List of action strings representing the new plan, or None if no solution found
        
    Example usage:
        # Define the goal: stack blocks in order g on r, b on g, y on b
        goal = [('on', 'g', 'r'), ('on', 'b', 'g'), ('on', 'y', 'b')]
        new_plan = replan_tower(franka, BlocksState, 'ten_blocks/10_block_domain.pddl', goal)
    """

    # Get block names from blocks_state dictionary
    block_names = list(blocks_state.keys())
    
    # Initialize state estimator
    estimator = StateEstimation("special_tower")
    
    # Set initial scene information
    estimator.set_scene_info(franka, blocks_state)
    
    # Replan from current state
    new_plan = estimator.replan_from_current_state(
        franka=franka,
        blocks_state=blocks_state,
        domain_file=domain_file,
        output_problem_file=output_problem_file
    )
    
    return new_plan


def main():
    """
    Main function to run the 10 blocks tower building task.
    Sets up the scene, configures the robot, and executes with replanning.
    """
    # Initialize Genesis simulator
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
    else:
        gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)

    # Build scene with blocks and robot
    scene, franka, BlocksState = create_scene_special_tower()

    # Configure robot control parameters (tuned for Franka Panda)

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

    
    # Execute with replanning after each action
    success = execute_with_replanning(
        domain_file='special_tower/special_tower_domain.pddl',
        blocks_state=BlocksState,
        franka=franka,
        scene=scene,
        max_iterations=100
    )
    
    if success:
        print("\nTask completed successfully!")
    else:
        print("\nTask failed to complete")


if __name__ == "__main__":
    main()
