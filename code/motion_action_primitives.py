import sys
import warnings
import numpy as np
import math
import genesis as gs
from planning import PlannerInterface
from scenes import create_scene_6blocks, create_scene_stacked
from pddl_planner import plan_task
from scipy.spatial.transform import Rotation as R



def move_while_closed(franka, scene, pos, quat=None, ignore_collisions=True, max_retries=30, tolerance=.01, waypoints=200, planner='BITstar', attached_object=None):
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    
    # get the end-effector link
    end_effector = franka.get_link("hand")
    
    # Create OMPL planner interface
    planner_interface = PlannerInterface(franka, scene)
    
    # Get current gripper position to maintain during movement
    current_gripper_pos = franka.get_qpos()[-2:]
    
    # Default quaternion if not provided
    if quat is None:
        quat = np.array([0, 1, 0, 0])

    
    # Store original position for variance
    original_pos = np.array(pos, dtype=np.float64)
    
    # Retry loop for IK and path planning
    for attempt in range(max_retries):
        # Add small random variance (< 1mm) to position when replanning (after first attempt)
        if attempt > 0:
            variance = np.random.uniform(-0.001, 0.001, size=3)  # +/- 1mm
            pos_with_variance = original_pos + variance
            print(f"  Adding variance: {variance * 1000} mm")
        else:
            pos_with_variance = original_pos
        
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_with_variance,
            quat=quat,
        )
        
        # Check if IK succeeded
        if qpos is None:
            print(f"Attempt {attempt+1}/{max_retries}: IK failed for position {original_pos}")
            continue
        
        # Set gripper to current closed position
        qpos[-2:] = current_gripper_pos

        # Debug: print current and goal qpos
        current_qpos = franka.get_qpos()
        print(f"\n[DEBUG] Planning attempt {attempt+1}")
        print(f"  Current qpos: {current_qpos[:7]}")  # First 7 are arm joints
        print(f"  Goal qpos: {qpos[:7]}")
        print(f"  Goal position: {pos_with_variance}")
        print(f"  Planner: {planner}")
        print(f"  Attached object: {attached_object}")
        
        # Try OMPL planner
        path = None
        try:
            path = planner_interface.plan_path(
                qpos_goal=qpos,
                num_waypoints=waypoints,
                planner=planner,
                timeout=20.0,
                attached_object=attached_object,
                ignore_collisions=ignore_collisions
            )
            if path and len(path) > 0:
                print(f" OMPL planning succeeded")
            else:
                print(f"OMPL returned empty path")
        except Exception as e:
            import traceback
            print(f"OMPL planning exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            path = None
        
        # Check if path planning succeeded
        if path is None or len(path) == 0:
            print(f"Attempt {attempt+1}/{max_retries}: Path planning failed for position {pos}")
            # quat = np.array(quat[2], quat[0], quat[1], quat[3])  # quat swap. Alternate ee angle on failed plans
            continue
        
        # Validate path before executing - check if final path qpos matches goal qpos
        final_qpos = path[-1]
        qpos_diff = np.linalg.norm(final_qpos[:6] - qpos[:6])
        
        if qpos_diff > 0.1:
            print(f"Attempt {attempt+1}/{max_retries}: Path validation failed - final qpos differs from goal by {qpos_diff:.3f}")
            continue
        
        # Try to execute the path - if it fails due to bad path, retry
        try:
            # First, ensure gripper is firmly closed before moving
            current_arm_pos = franka.get_qpos()[:-2]
            for _ in range(5):
                franka.control_dofs_position(current_arm_pos, motors_dof)
                franka.control_dofs_force(np.array([-3.0, -3.0]), fingers_dof)
                scene.step()
            
            for i, waypoint in enumerate(path):
                # Control arm with position from waypoint
                franka.control_dofs_position(waypoint[:-2], motors_dof)
                # Apply strong closing force to maintain grip
                franka.control_dofs_force(np.array([-3.0, -3.0]), fingers_dof)
                scene.step()
            
            # Hold position until arm reaches desired pose
            final_arm_pos = path[-1][:-2]
            step_count = 0
            while True:
                franka.control_dofs_force(np.array([-3.0, -3.0]), fingers_dof)
                scene.step()
                
                actual_final_pos = end_effector.get_pos()
                final_dist = np.linalg.norm(actual_final_pos - original_pos)
                step_count += 1
                if step_count % 10 == 0:
                    print(f"move_while_closed convergence: step {step_count}, distance={final_dist:.6f}")
                if final_dist <= tolerance or step_count >= 50:
                    break
            
            # Check if final position is within 0.001m of desired position
            actual_final_pos = end_effector.get_pos()
            final_dist = np.linalg.norm(actual_final_pos - original_pos)
            print(f"Final position error: {final_dist:.6f}m")
            
            if final_dist > tolerance:
                print(f"Position error {final_dist:.6f}m exceeds %f threshold, replanning...", tolerance)
                # Continue to next attempt to replan
                continue
            
            # Success!
            print(f"Successfully reached target within tolerance")
            return
            
        except ValueError as e:
            # Path was bad, continue to next retry attempt
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            continue
    
    # If we get here, all retries failed
    print(f"ERROR: Failed to find valid path after {max_retries} attempts for position {original_pos}")
    raise RuntimeError(f"Path planning failed for position {original_pos}")

def move_while_open(franka, scene, pos, max_retries=10, tolerance=.011, waypoints=200, planner='BITstar', attached_object=None):
    # get the end-effector link
    end_effector = franka.get_link("hand")
    
    # Create OMPL planner interface
    planner_interface = PlannerInterface(franka, scene)
    
    # Store original position for variance
    original_pos = np.array(pos, dtype=np.float64)
    
    # Retry loop for IK and path planning
    for attempt in range(max_retries):
        # Add small random variance (< 1mm) to position when replanning (after first attempt)
        if attempt > 0:
            variance = np.random.uniform(-0.001, 0.001, size=3)  # +/- 1mm
            pos_with_variance = original_pos + variance
            print(f"  Adding variance: {variance * 1000} mm")
        else:
            pos_with_variance = original_pos
        
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_with_variance,
            quat=np.array([0, 1, 0, 0]),
        )
        
        if qpos is None:
            print(f"Attempt {attempt+1}/{max_retries}: IK failed for position {original_pos}")
            continue
            
        qpos[-2:] = 0.04  # gripper open
        
        # Debug: print current and goal qpos
        current_qpos = franka.get_qpos()
        print(f"\n[DEBUG] Planning attempt {attempt+1} (move_while_open)")
        print(f"  Current qpos: {current_qpos[:7]}")
        print(f"  Goal qpos: {qpos[:7]}")
        print(f"  Goal position: {pos_with_variance}")
        print(f"  Planner: {planner}")
        
        # Try OMPL planner (no attached object for open gripper)
        path = None
        try:
            path = planner_interface.plan_path(
                qpos_goal=qpos,
                num_waypoints=waypoints,
                planner=planner,
                timeout=20.0,
                attached_object=attached_object,
                ignore_collisions=False
            )
            if path and len(path) > 0:
                print(f"  ✓ OMPL planning succeeded: {len(path)} waypoints")
            else:
                print(f"  ✗ OMPL returned empty path")
        except Exception as e:
            import traceback
            print(f"  ✗ OMPL planning exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            path = None
        
        if path is None or len(path) == 0:
            print(f"Attempt {attempt+1}/{max_retries}: Path planning failed for position {original_pos}")
            continue
        
        # Validate path before executing - check if final path qpos matches goal qpos
        # Attempts to fix weird bug where the arm moves back to the home position
        final_qpos = path[-1]
        qpos_diff = np.linalg.norm(final_qpos[:6] - qpos[:6])
        
        if qpos_diff > 0.008:
            print(f"Attempt {attempt+1}/{max_retries}: Path validation failed - final qpos differs from goal by {qpos_diff:.3f}")
            continue
        
        # Try to execute the path - if it fails due to bad path, retry
        try:
            for i, waypoint in enumerate(path):
                franka.control_dofs_position(waypoint)
                scene.step()
            
            # Hold position until arm reaches desired pose
            final_waypoint = path[-1]
            step_count = 0
            while True:
                scene.step()
                
                actual_final_pos = end_effector.get_pos()
                final_dist = np.linalg.norm(actual_final_pos - original_pos)
                step_count += 1
                if step_count % 10 == 0:
                    print(f"move_while_open convergence: step {step_count}, distance={final_dist:.6f}")
                if final_dist <= tolerance or step_count >= 110:
                    break
            
            # Check if final position is within 0.001m of desired position
            actual_final_pos = end_effector.get_pos()
            final_dist = np.linalg.norm(actual_final_pos - original_pos)
            print(f"Final position error: {final_dist:.6f}m")
            
            if final_dist > tolerance:
                print(f"Position error {final_dist:.6f}m exceeds 0.001m threshold, replanning...")
                # Continue to next attempt to replan
                continue

            # Success!
            print(f"Successfully reached target within 0.001m tolerance")
            return
    
        except ValueError as e:
            # Path was bad, continue to next retry attempt
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            continue
    
    # If we get here, all retries failed
    # Genesis kind of sucks
    print(f"ERROR: Failed to find valid path after {max_retries} attempts for position {original_pos}")
    raise RuntimeError(f"Path planning failed for position {original_pos}")

def close_gripper(franka, scene):
    # Close gripper with force control
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    
    # Hold arm position steady
    arm_qpos = franka.get_qpos()[:-2]
    
    # Apply strong force to close gripper, matching move_while_closed behavior
    max_steps = 50
    for i in range(max_steps):
        # Keep arm steady, apply strong closing force to fingers
        franka.control_dofs_position(arm_qpos, motors_dof)
        franka.control_dofs_force(np.array([-2.0, -2.0]), fingers_dof)
        scene.step()

def open_gripper(franka, scene):
    # Open gripper with position control
    qpos = franka.get_qpos()
    qpos[-2:] = 0.04  # Open position
    target_gripper_pos = np.array([0.04, 0.04])
    
    # Control until gripper reaches desired position
    max_steps = 80
    for i in range(max_steps):
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
    
# def PICK(block, blocks_state, franka, scene):
#     b = blocks_state[block]
#     # Convert Genesis positions to proper numpy arrays
#     bpos = np.asarray(b.get_pos(), dtype=np.float64)

#     #move above block
#     move_while_open(franka, scene, bpos + np.array([0, 0, 0.18]))
#     # #lower to block
#     move_while_open(franka, scene, bpos + np.array([0, 0, 0.1]))
#     # #grasp
#     close_gripper(franka, scene)
#     # #move up
#     move_while_closed(franka, scene, bpos + np.array([0, 0, 0.3]))

def PICK(block, blocks_state, franka, scene):
    gripper_ee_offset = np.array([0.0027, 0.003, 0.04])

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
    move_while_closed(franka, scene, np.array([0.3, 0.3, 0.23]), attached_object=b, waypoints=70)
    # #lower next to block
    move_while_closed(franka, scene,  np.array([0.3, 0.3, 0.18]), attached_object=b,waypoints=70)
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
    move_while_closed(franka, scene, b2pos + np.array([0.8, 0, 0.23]), quat=np.array([0.7071, 0.7071, 0, 0]))
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([.8, 0, 0.17]), quat=np.array([0.7071, 0.7071, 0, 0]))
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([.8, 0, 0.3]))

def PLACEWEST(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and west of block2, aligned with y axis
    move_while_closed(franka, scene, b2pos + np.array([-0.8, 0, 0.23]), quat=np.array([0.7071, 0.7071, 0, 0]))
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([-.8, 0, 0.17]), quat=np.array([0.7071, 0.7071, 0, 0]))
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([-.8, 0, 0.3]))

def PLACENORTH(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and north of block2, aligned with x axis
    move_while_closed(franka, scene, b2pos + np.array([0, 0.8, 0.23]))
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0, 0.8, 0.17]))
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0, 0.8, 0.3]))

def PLACESOUTH(block1, block2, blocks_state, franka, scene):
    b1 = blocks_state[block1]
    b2 = blocks_state[block2]
    # Convert Genesis positions to proper numpy arrays
    b1pos = np.asarray(b1.get_pos(), dtype=np.float64)
    b2pos = np.asarray(b2.get_pos(), dtype=np.float64)

    #move above and south of block2, aligned with x axis
    move_while_closed(franka, scene, b2pos + np.array([0, -0.8, 0.23]))
    # #lower next to block2
    move_while_closed(franka, scene, b2pos + np.array([0, -0.8, 0.17]))
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, b2pos + np.array([0, -0.8, 0.3]))

def PLACE_SPECIFIC(block, position, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    # bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.30]), attached_object=b, waypoints=70)
    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.14]), attached_object=b, waypoints=70)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]))

def PLACE_SPECIFIC_ORIENTED(block, position, quat, blocks_state, franka, scene):
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    # bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.30]), quat, attached_object=b, waypoints=70)
    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.14]), quat, attached_object=b, waypoints=70)
    # #release
    open_gripper(franka, scene)
    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]))


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

        elif action_name == "placespecific":
            # place block at a specific position
            block = parts[1]
            # Define target positions for each block
            target_positions = {
                # for Two Tower Task
                'b': np.array([0.4, -0.50, 0.0]),
                'c': np.array([0.6, -0.25, 0.0]),

            }
            if block == 'b':
                position = target_positions['b']
            elif block == 'c':
                position = target_positions['c']
            else:
                print(f"ERROR: Unknown block for placespecific: {block}")
                continue
            PLACE_SPECIFIC(block, position, blocks_state, franka, scene)
        
        elif action_name == "placespecificoriented":
            # place block at a specific position with a specific quaternion orientation
            # Perhaps it's a little excessive to write all of this out. I can speed this up later.
            block = parts[1]
            # Define target positions for each block

            # Radius of circle connecting block centers in special tower task
            r = ((0.025/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180)))) / math.cos(54*(math.pi/180))
            theta_r = 0*(math.pi/180)
            theta_b = 72*(math.pi/180)+math.pi
            theta_y = 144*(math.pi/180)+math.pi
            theta_g = 216*(math.pi/180)+math.pi
            theta_o = 288*(math.pi/180)+math.pi
            block_state = blocks_state[block]
            print("block:", block)
            ref_block = parts[2]
            block_pos = np.asarray(block_state.get_pos(), dtype=np.float64)
            block_quat = np.asarray(block_state.get_quat(), dtype=np.float64)
            print("block_orient:", block_quat)
            print("block_state:", block_state)

            target_positions = {
                # For Special Tower Task
                'r1': np.array([0.5+r*math.cos(theta_r), -0.5+r*math.sin(theta_r), 0.0]),
                'g1': np.array([0.5+r*math.cos(theta_g), -0.5+r*math.sin(theta_g), 0.0]),
                'b1': np.array([0.5+r*math.cos(theta_b), -0.5+r*math.sin(theta_b), 0.0]),
                'o1': np.array([0.5+r*math.cos(theta_o), -0.5+r*math.sin(theta_o), 0.0]),
                'y1': np.array([0.5+r*math.cos(theta_y), -0.5+r*math.sin(theta_y), 0.0]),
                'r2': np.array([0.4, -0.2, 0.0]), # Not made yet, will just require a angle change and height change
                'g2': np.array([0.4, -0.2, 0.0]),
                'b2': np.array([0.4, -0.2, 0.0]),
                'o2': np.array([0.4, -0.2, 0.0]),
                'y2': np.array([0.4, -0.2, 0.0]),
            }
            target_orientations = {
                # [w, x, y, z] unit quaternions
                'r1': np.array(R.from_euler('xz', [math.pi,theta_r]).as_quat(scalar_first=True)),
                'g1': np.array(R.from_euler('xz', [math.pi,theta_g]).as_quat(scalar_first=True)),
                'b1': np.array(R.from_euler('xz', [math.pi,theta_b]).as_quat(scalar_first=True)),
                'o1': np.array(R.from_euler('xz', [math.pi,theta_o]).as_quat(scalar_first=True)),
                'y1': np.array(R.from_euler('xz', [math.pi,theta_y]).as_quat(scalar_first=True)),
                'r2': np.array([0.0, 0.0, 0.0, 1.0]),
                'g2': np.array([0.0, 0.0, 0.0, 1.0]),
                'b2': np.array([0.0, 0.0, 0.0, 1.0]),
                'o2': np.array([0.0, 0.0, 0.0, 1.0]),
                'y2': np.array([0.0, 0.0, 0.0, 1.0]),
            }

            if block == 'r1':
                position = target_positions['r1']
                quat = target_orientations['r1']
            elif block == 'r2':
                position = target_positions['r2']
                quat = target_orientations['r2']
            elif block == 'g1':
                position = target_positions['g1']
                quat = target_orientations['g1']
            elif block == 'g2':
                position = target_positions['g2']
                quat = target_orientations['g2']
            elif block == 'b1':
                position = target_positions['b1']
                quat = target_orientations['b1']
            elif block == 'b2':
                position = target_positions['b2']
                quat = target_orientations['b2']
            elif block == 'o1':
                position = target_positions['o1']
                quat = target_orientations['o1']
            elif block == 'o2':
                position = target_positions['o2']
                quat = target_orientations['o2']
            elif block == 'y1':
                position = target_positions['y1']
                quat = target_orientations['y1']
            elif block == 'y2':
                position = target_positions['y2']
                quat = target_orientations['y2']
            else:
                print(f"ERROR: Unknown block for placespecific: {block}")
                continue
            PLACE_SPECIFIC_ORIENTED(block, position, quat, blocks_state, franka, scene)

        elif action_name == "place_relative_oriented":
            block = parts[1]

            block_state = blocks_state[block]
            # print("block:", block)
            ref_block = parts[2]
            block_pos = np.asarray(block_state.get_pos(), dtype=np.float64)
            block_quat = np.asarray(block_state.get_quat(), dtype=np.float64)
            # print("block_orient:", block_quat)
            # print("block_state:", block_state)

            # 0.02m is the half-width of a block, add more to create extra space
            r = ((0.02/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180)))) / math.cos(54*(math.pi/180))
            theta_dif = 72*(math.pi/180)  # 72 degrees in radians

            target_position = np.array([block_pos[0]+r*math.cos(theta_dif), block_pos[1]+r*math.sin(theta_dif), block_pos[2]]),
            target_quat = np.array(R.from_euler('xz', [math.pi,theta_dif]).as_quat(scalar_first=True))
            
            # quaternion multiplication to combine orientations
            target_quat[0] = block_quat[0] * target_quat[0] - block_quat[1] * target_quat[1] - block_quat[2] * target_quat[2] - block_quat[3] * target_quat[3]
            target_quat[1] = block_quat[0] * target_quat[1] + block_quat[1] * target_quat[0] + block_quat[2] * target_quat[3] - block_quat[3] * target_quat[2]
            target_quat[2] = block_quat[0] * target_quat[2] - block_quat[1] * target_quat[3] + block_quat[2] * target_quat[0] + block_quat[3] * target_quat[1]
            target_quat[3] = block_quat[0] * target_quat[3] + block_quat[1] * target_quat[2] - block_quat[2] * target_quat[1] + block_quat[3] * target_quat[0]

            PLACE_SPECIFIC_ORIENTED(block, target_position, target_quat, blocks_state, franka, scene)

        
        else:
            print(f"ERROR: Unknown action: {action_name}")
    
    return True
