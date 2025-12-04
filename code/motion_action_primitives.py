import sys
import warnings
import numpy as np
import genesis as gs
from planning import PlannerInterface



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
                attached_object=attached_object
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
                franka.control_dofs_force(np.array([-2.0, -2.0]), fingers_dof)
                scene.step()
            
            for i, waypoint in enumerate(path):
                # Control arm with position from waypoint
                franka.control_dofs_position(waypoint[:-2], motors_dof)
                # Apply strong closing force to maintain grip
                franka.control_dofs_force(np.array([-2.0, -2.0]), fingers_dof)
                scene.step()
            
            # Hold position until arm reaches desired pose
            final_arm_pos = path[-1][:-2]
            step_count = 0
            while True:
                franka.control_dofs_force(np.array([-2.0, -2.0]), fingers_dof)
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
                attached_object=attached_object
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

