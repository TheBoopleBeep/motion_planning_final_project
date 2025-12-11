import sys
import numpy as np
import genesis as gs
import os
from scenes import create_scene_10blocks
from fast_downward_planner import plan_task, load_plan_from_file, save_plan_to_file
from motion_action_primitives import move_while_closed, move_while_open, close_gripper, open_gripper
from state_estimation import StateEstimation

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

def execute_action(action_str, blocks_state, franka, scene, estimator=None, verify_state=False):
    """
    Execute a single PDDL action with optional state verification.
    
    Parameters:
        action_str: Action string (e.g., "(pick r)" or "(stack g r)")
        blocks_state: Dictionary mapping block names to block entities
        franka: Robot adapter
        scene: Genesis scene
        estimator: Optional StateEstimation object for state verification
        verify_state: Whether to verify state matches predicted state
    
    Returns:
        dict with keys:
            - 'success': bool indicating if action executed successfully
            - 'state_comparison': dict with comparison results or None
    """
    # Parse action string
    parts = action_str.replace('(', '').replace(')', '').split()
    action_name = parts[0]
    
    print(f"Executing: {action_str}")
    
    # Compute predicted state for verification
    predicted_facts = None
    if verify_state and estimator is not None:
        current_facts = set(estimator.parse_state_to_pddl())
        
        if action_name == "pick":
            predicted_facts = estimator.comp_PICK(parts[1], current_facts)
        elif action_name == "stack":
            predicted_facts = estimator.comp_STACK(parts[1], parts[2], current_facts)
        elif action_name == "unstack":
            predicted_facts = estimator.comp_PICK(parts[1], current_facts)
    
    try:
        # Execute the appropriate action
        if action_name == "pick":
            PICK(parts[1], blocks_state, franka, scene)
        elif action_name == "stack":
            STACK(parts[1], parts[2], blocks_state, franka, scene)
        elif action_name == "unstack":
            PICK(parts[1], blocks_state, franka, scene)
        elif action_name == "place":
            PLACE(parts[1], blocks_state, franka, scene)
        else:
            print(f"ERROR: Unknown action: {action_name}")
            return {'success': False, 'state_comparison': None}
        
        # Verify state if requested
        state_comparison = None
        if verify_state and estimator is not None and predicted_facts is not None:
            state_comparison = estimator.compare_states(predicted_facts, franka, blocks_state)
        
        return {'success': True, 'state_comparison': state_comparison}
        
    except Exception as e:
        print(f"✗ Action failed: {action_str}")
        print(f"   Error: {e}")
        return {'success': False, 'state_comparison': None}


def execute_with_replanning(domain_file, blocks_state, franka, scene, max_iterations=100):
    """
    Execute a task with replanning only when state mismatches are detected.
    
    Parameters:
        domain_file: Path to the PDDL domain file
        blocks_state: Dictionary mapping block names to block entities
        franka: Robot adapter
        scene: Genesis scene
        max_iterations: Maximum number of actions to execute (safety limit)
    
    Returns:
        True if goal was achieved, False otherwise
    """


    estimator = StateEstimation("tower")
    
    # Initialize block info once
    estimator.set_scene_info(franka, blocks_state)
    
    iteration = 0
    plan = None  # Current plan
    plan_index = 0  # Track position in current plan
    need_replan = True  # Flag to trigger replanning
    
    while iteration < max_iterations:
        iteration += 1
        
        # Replan only if needed (first iteration, state mismatch, or action failure)
        if need_replan:
            print(f"\nReplanning (iteration {iteration})...")
            
            # Replan (this internally updates all state)
            plan = estimator.replan_from_current_state(franka, blocks_state, domain_file)
            plan_index = 0  # Reset to start of new plan
            need_replan = False  # Reset flag
            
            # Check planner result
            if plan is None:
                print("\nPlanner failed to find a solution")
                return False
            elif len(plan) == 0:
                print("\nGoal achieved!")
                return True
            else:
                print(f"Plan generated with {len(plan)} actions")
        
        # Execute next action with state verification
        action = plan[plan_index]
        result = execute_action(action, blocks_state, franka, scene, estimator=estimator, verify_state=True)
        
        # Determine if replanning is needed
        if not result['success']:
            need_replan = True
        elif result['state_comparison'] and not result['state_comparison']['matches']:
            need_replan = True
        else:
            plan_index += 1
            if plan_index >= len(plan):
                need_replan = True
        
        # Wait a bit for the scene to settle
        for _ in range(10):
            scene.step()
    
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
    
    Returns:
        List of action strings representing the new plan, or None if no solution found
        
    Example usage:
        # Define the goal: stack blocks in order g on r, b on g, y on b
        goal = [('on', 'g', 'r'), ('on', 'b', 'g'), ('on', 'y', 'b')]
        new_plan = replan_tower(franka, BlocksState, 'ten_blocks/10_block_domain.pddl', goal)
    """

    # Get block names from blocks_state dictionary
    block_names = list(blocks_state.keys())
    
    # Initialize state estimator
    estimator = StateEstimation("tower")
    
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
    scene, franka, BlocksState = create_scene_10blocks()

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
        domain_file='ten_blocks/10_block_domain.pddl',
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
