import sys
import warnings
import numpy as np
import genesis as gs
from planning import PlannerInterface

# Constants
GRIPPER_CLOSE_FORCE = -2.0
GRIPPER_OPEN_POSITION = 0.04
JACOBIAN_CONVERGENCE_GAIN = 4.0
MAX_JOINT_CORRECTION = 0.02  # radians
DEFAULT_ORIENTATION = np.array([0, 1, 0, 0])


def _apply_jacobian_correction(franka, scene, end_effector, target_pos, tolerance,
                               motors_dof, fingers_dof, max_iterations=10, gripper_closed=True):
    """
    Apply Jacobian-based position correction to reach target position.
    
    Parameters:
        franka: Robot adapter
        scene: Genesis scene
        end_effector: End effector link
        target_pos: Target position to reach
        tolerance: Position tolerance in meters
        motors_dof: Motor DOF indices
        fingers_dof: Finger DOF indices
        max_iterations: Maximum correction iterations
        gripper_closed: Whether gripper should remain closed
    """
    for iteration in range(max_iterations):
        # Get current state
        current_pos = np.array(end_effector.get_pos(), dtype=np.float64)
        current_qpos = np.array(franka.get_qpos(), dtype=np.float64)
        
        # Compute position error
        pos_error = target_pos - current_pos
        error_norm = np.linalg.norm(pos_error)
        
        print(f"  Jacobian iteration {iteration+1}/{max_iterations}: error={error_norm:.6f}m")
        
        if error_norm <= tolerance:
            print(f"  \u2713 Jacobian correction succeeded")
            return
        
        # Get spatial Jacobian
        J_spatial = franka.get_jacobian(end_effector)
        J_spatial_np = np.array(J_spatial, dtype=np.float64)
        
        # Extract position Jacobian (first 3 rows, first 7 columns for arm joints)
        J_pos = J_spatial_np[:3, :7]
        
        # Compute joint correction: Δq = α * J† * Δx
        J_pinv = np.linalg.pinv(J_pos)
        delta_q = J_pinv @ pos_error * JACOBIAN_CONVERGENCE_GAIN
        
        # Limit maximum joint correction for safety
        delta_q_norm = np.linalg.norm(delta_q)
        if delta_q_norm > MAX_JOINT_CORRECTION:
            delta_q = delta_q * (MAX_JOINT_CORRECTION / delta_q_norm)
        
        # Apply correction
        corrected_qpos = current_qpos.copy()
        corrected_qpos[:7] += delta_q
        
        # Execute correction while maintaining gripper state
        for _ in range(5):
            franka.control_dofs_position(corrected_qpos[:-2], motors_dof)
            if gripper_closed:
                franka.control_dofs_force(np.array([GRIPPER_CLOSE_FORCE, GRIPPER_CLOSE_FORCE]), fingers_dof)
            scene.step()
    
    # Settle and check final error
    for _ in range(10):
        scene.step()
    
    final_pos = np.array(end_effector.get_pos(), dtype=np.float64)
    final_error = np.linalg.norm(final_pos - target_pos)
    
    if final_error <= tolerance:
        print(f"  \u2713 Jacobian correction succeeded after {max_iterations} iterations")
    else:
        print(f"  \u26a0 Jacobian correction insufficient (error: {final_error:.6f}m), continuing...")


def move_while_closed(franka, scene, pos, quat=None, ignore_collisions=False, max_retries=30, 
                     tolerance=0.001, waypoints=-1, planner='BITstar', attached_object=None):
    """
    Move robot with gripper closed to target position, maintaining grip on held object.
    
    Parameters:
        franka: Robot adapter
        scene: Genesis scene
        pos: Target position (x, y, z)
        quat: Target orientation quaternion (defaults to standard orientation)
        ignore_collisions: Whether to ignore collision checking
        max_retries: Maximum number of planning attempts
        tolerance: Position tolerance in meters
        waypoints: Number of waypoints (-1 for automatic)
        planner: OMPL planner to use ('BITstar', etc.)
        attached_object: Object held by gripper for collision checking
    """
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    
    end_effector = franka.get_link("hand")
    planner_interface = PlannerInterface(franka, scene)
    
    # Maintain current gripper position during movement
    current_gripper_pos = franka.get_qpos()[-2:]
    
    # Use default orientation if not specified
    if quat is None:
        quat = DEFAULT_ORIENTATION
    
    original_pos = np.array(pos, dtype=np.float64)
    
    # Retry loop for IK and path planning
    for attempt in range(max_retries):
        # Compute inverse kinematics
        qpos = franka.inverse_kinematics(link=end_effector, pos=pos, quat=quat)
        
        if qpos is None:
            print(f"Attempt {attempt+1}/{max_retries}: IK failed")
            continue
        
        # Maintain gripper position
        qpos[-2:] = current_gripper_pos

        # Debug output
        current_qpos = franka.get_qpos()
        print(f"\n[DEBUG] Planning attempt {attempt+1}")
        print(f"  Current qpos: {current_qpos[:7]}")
        print(f"  Goal qpos: {qpos[:7]}")
        print(f"  Goal position: {pos}")
        print(f"  Attached object: {attached_object}")
        
        # Plan path with OMPL
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
                print(f"  ✓ OMPL planning succeeded: {len(path)} waypoints")
            else:
                print(f"  ✗ OMPL returned empty path")
        except Exception as e:
            import traceback
            print(f"  ✗ OMPL planning exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            path = None
        
        if path is None or len(path) == 0:
            print(f"Attempt {attempt+1}/{max_retries}: Path planning failed")
            continue
        
        # Validate path endpoint matches goal
        final_qpos = path[-1]
        qpos_diff = np.linalg.norm(final_qpos[:6] - qpos[:6])
        
        if qpos_diff > 0.1:
            print(f"Attempt {attempt+1}/{max_retries}: Path validation failed (qpos diff: {qpos_diff:.3f})")
            continue
        
        # Execute the path
        try:
            # Ensure gripper is firmly closed before moving
            current_arm_pos = franka.get_qpos()[:-2]
            for _ in range(5):
                franka.control_dofs_position(current_arm_pos, motors_dof)
                franka.control_dofs_force(np.array([GRIPPER_CLOSE_FORCE, GRIPPER_CLOSE_FORCE]), fingers_dof)
                scene.step()
            
            # Follow path while maintaining grip
            for waypoint in path:
                franka.control_dofs_position(waypoint[:-2], motors_dof)
                franka.control_dofs_force(np.array([GRIPPER_CLOSE_FORCE, GRIPPER_CLOSE_FORCE]), fingers_dof)
                scene.step()
            
            # Hold position until convergence
            step_count = 0
            max_convergence_steps = 50
            while step_count < max_convergence_steps:
                franka.control_dofs_force(np.array([GRIPPER_CLOSE_FORCE, GRIPPER_CLOSE_FORCE]), fingers_dof)
                scene.step()
                
                actual_final_pos = end_effector.get_pos()
                final_dist = np.linalg.norm(actual_final_pos - original_pos)
                step_count += 1
                
                if step_count % 10 == 0:
                    print(f"  Convergence: step {step_count}, distance={final_dist:.6f}m")
                
                if final_dist <= tolerance:
                    break
            
            # Check final position
            actual_final_pos = end_effector.get_pos()
            final_dist = np.linalg.norm(actual_final_pos - original_pos)
            print(f"  Final position error: {final_dist:.6f}m")
            
            # Apply Jacobian-based correction if needed
            if final_dist > tolerance:
                print(f"  Position error exceeds tolerance, using Jacobian correction...")
                _apply_jacobian_correction(
                    franka, scene, end_effector, original_pos, tolerance,
                    motors_dof, fingers_dof, max_iterations=10, gripper_closed=True
                )
                return
            
            print(f"  ✓ Successfully reached target")
            return
            
        except ValueError as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            continue
    
    # All retries exhausted
    print(f"ERROR: Failed to find valid path after {max_retries} attempts")
    raise RuntimeError(f"Path planning failed for position {original_pos}")

def move_while_open(franka, scene, pos, max_retries=10, tolerance=0.001, waypoints=-1, 
                   planner='BITstar', attached_object=None, waypoint_scale = .75, ignore_collisions=False):
    """
    Move robot with gripper open to target position.
    
    Parameters:
        franka: Robot adapter
        scene: Genesis scene
        pos: Target position (x, y, z)
        max_retries: Maximum number of planning attempts
        tolerance: Position tolerance in meters
        waypoints: Number of waypoints (-1 for automatic)
        planner: OMPL planner to use
        attached_object: Optional object for collision checking
    """
    motors_dof = np.arange(7)
    end_effector = franka.get_link("hand")
    planner_interface = PlannerInterface(franka, scene)
    original_pos = np.array(pos, dtype=np.float64)
    
    for attempt in range(max_retries):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=DEFAULT_ORIENTATION
        )
        
        if qpos is None:
            print(f"Attempt {attempt+1}/{max_retries}: IK failed")
            continue
            
        qpos[-2:] = GRIPPER_OPEN_POSITION
        
        # Debug output
        current_qpos = franka.get_qpos()
        print(f"\n[DEBUG] Planning attempt {attempt+1} (open gripper)")
        print(f"  Current qpos: {current_qpos[:7]}")
        print(f"  Goal qpos: {qpos[:7]}")
        
        # Plan path
        path = None
        try:
            path = planner_interface.plan_path(
                qpos_goal=qpos,
                num_waypoints=waypoints,
                planner=planner,
                timeout=20.0,
                attached_object=attached_object,
                ignore_collisions=ignore_collisions,
                waypoint_scale=waypoint_scale
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
            print(f"Attempt {attempt+1}/{max_retries}: Path planning failed")
            continue
        
        # Validate path endpoint
        final_qpos = path[-1]
        qpos_diff = np.linalg.norm(final_qpos[:6] - qpos[:6])
        
        if qpos_diff > 0.008:
            print(f"Attempt {attempt+1}/{max_retries}: Path validation failed (qpos diff: {qpos_diff:.3f})")
            continue
        
        # Execute the path
        try:
            for waypoint in path:
                franka.control_dofs_position(waypoint)
                scene.step()
            
            # Hold position until convergence
            step_count = 0
            max_convergence_steps = 30
            while step_count < max_convergence_steps:
                scene.step()
                
                actual_final_pos = end_effector.get_pos()
                final_dist = np.linalg.norm(actual_final_pos - original_pos)
                step_count += 1
                
                if step_count % 10 == 0:
                    print(f"  Convergence: step {step_count}, distance={final_dist:.6f}m")
                
                if final_dist <= tolerance:
                    break
            
            # Check final position
            actual_final_pos = end_effector.get_pos()
            final_dist = np.linalg.norm(actual_final_pos - original_pos)
            print(f"  Final position error: {final_dist:.6f}m")
            
            # Apply Jacobian-based correction if needed
            if final_dist > tolerance:
                print(f"  Position error exceeds tolerance, using Jacobian correction...")
                _apply_jacobian_correction(
                    franka, scene, end_effector, original_pos, tolerance,
                    motors_dof, fingers_dof=None, max_iterations=5, gripper_closed=False
                )
                return
            
            print(f"  ✓ Successfully reached target")
            return
            
        except ValueError as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            return
    
    # All retries exhausted
    print(f"ERROR: Failed to find valid path after {max_retries} attempts")
    raise RuntimeError(f"Path planning failed for position {original_pos}")

def close_gripper(franka, scene):
    """
    Close the gripper with force control while keeping arm stable.
    
    Parameters:
        franka: Robot adapter
        scene: Genesis scene
    """
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    q_init = np.array(franka.get_qpos(), dtype=np.float64)
    
    # Stabilize arm position
    for _ in range(5):
        franka.control_dofs_position(q_init[:7], motors_dof)
        franka.control_dofs_position(q_init[-2:], fingers_dof)
        scene.step()
    
    # Close gripper with strong force
    for _ in range(30):
        franka.control_dofs_position(q_init[:7], motors_dof)
        franka.control_dofs_force(np.array([GRIPPER_CLOSE_FORCE, GRIPPER_CLOSE_FORCE]), fingers_dof)
        scene.step()

def open_gripper(franka, scene):
    """
    Open the gripper with position control while keeping arm stable.
    
    Parameters:
        franka: Robot adapter
        scene: Genesis scene
    """
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    q_init = np.array(franka.get_qpos(), dtype=np.float64)
    target_gripper_pos = np.array([GRIPPER_OPEN_POSITION, GRIPPER_OPEN_POSITION], dtype=np.float64)
    
    # # Stabilize arm position
    # for _ in range(5):
    #     franka.control_dofs_position(q_init[:7], motors_dof)
    #     franka.control_dofs_position(q_init[-2:], fingers_dof)
    #     scene.step()
    
    # Open gripper to target position
    for _ in range(30):
        franka.control_dofs_position(q_init[:7], motors_dof)
        franka.control_dofs_position(target_gripper_pos, fingers_dof)
        scene.step()

