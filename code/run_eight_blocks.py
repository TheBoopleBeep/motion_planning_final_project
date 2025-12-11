import sys
import numpy as np
import genesis as gs
import os
from scenes import create_scene_8blocks
from pddl_planner import plan_task, load_plan_from_file, save_plan_to_file
from motion_action_primitives import move_while_closed, move_while_open, close_gripper, open_gripper

gripper_ee_offset = np.array([0.003, 0.003, 0.03])
baseblock = None
def STACK(block1, block2, blocks_state, franka, scene):
    global baseblock
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above block2 - pass b1 as attached object
    move_while_closed(franka, scene, b2pos + np.array([0, 0, 0.18]), tolerance=.01, attached_object=b1,waypoints=70)
    for _ in range(30):
        scene.step()
    #relative movement to account for minor inaccuracies
    b1 = blocks_state[block1]
    # b2 = blocks_state[block2]
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    if baseblock is None:
        baseblock = b2
        print(f"Setting baseblock to {block2}")
    basepos = np.asarray(baseblock.get_pos(), dtype=np.float64)
    print(f"Baseblock position: {basepos}, stacking on block {block2} at position {b2pos}")
    diff = basepos - b1pos
    des_pos = np.array([diff[0] + basepos[0]-.005, diff[1]+basepos[1]-.0015, b2pos[2]+.12]) + gripper_ee_offset
    print(diff)
    print(des_pos)
    move_while_closed(franka, scene, np.array([des_pos[0], des_pos[1], des_pos[2]]) , tolerance=.01, attached_object=b1, waypoints=50)
    # despos = np.array([basepos[0], basepos[1], b2pos[2]])+ np.array([0.0035, -0.0029, 0.12])
    # move_while_closed(franka, scene, despos + gripper_ee_offset , waypoints=120, tolerance=.0075, attached_object=b1)
    # #release
    open_gripper(franka, scene)
    # Let gripper physically open and block settle before next move
    for _ in range(30):
        scene.step()
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0, 0, 0.3]))
    
def PICK(block, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    bpos = np.asarray(b.get_pos(), dtype=np.float64)
    # offset forward because actual gripper is behind the end effector link
    #move above block
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.2]) + gripper_ee_offset, waypoints=20)
    # #lower to block
    move_while_open(franka, scene, bpos + np.array([0.005, 0, 0.07]) + gripper_ee_offset, tolerance=.006,attached_object=b, waypoints=60)
    # #close gripper
    close_gripper(franka, scene)
    # #move up - pass b as attached object
    move_while_closed(franka, scene, bpos + np.array([0.005, 0, 0.3])+ gripper_ee_offset, attached_object=b, tolerance=.01)

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


def main():
    """
    Main function to run the 8 blocks problem.
    Sets up the scene, configures the robot, and executes the PDDL plan.
    """
    # Ensure Genesis is initialized before building scenes
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
    else:
        gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)

    # Build the scene using the factory
    scene, franka, BlocksState = create_scene_8blocks()

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

    # Execute PDDL plan (compute if needed, otherwise load from file)
    execute_pddl_plan('8_block_plan.json', '8_block_domain.pddl', '8_block_problem.pddl', BlocksState, franka, scene)


if __name__ == "__main__":
    main()
