import sys
import warnings
import numpy as np
import math
from pddl_planner import plan_task
from scipy.spatial.transform import Rotation as R
from time import sleep

import genesis as gs
from planning import PlannerInterface

# Constants
GRIPPER_CLOSE_FORCE = -3.0
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
                     tolerance=0.01, waypoints=-1, planner='BITstar', attached_object=None):
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

def move_while_open(franka, scene, pos, quat=None, max_retries=10, tolerance=0.011, waypoints=-1, 
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
    
    if quat is None:
        quat = DEFAULT_ORIENTATION
    
    for attempt in range(max_retries):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=quat
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

def control_to_pos(franka, scene, pos, quat=None, ignore_collisions=True, max_retries=30, tolerance=.01, waypoints=200, planner='BITstar', attached_object=None):
    print("Running control_to_pos() to: ",pos)
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

        # # Add small random variance (< 1mm) to position when replanning (after first attempt)
        # if attempt > 0:
        #     variance = np.random.uniform(-0.001, 0.001, size=3)  # +/- 1mm
        #     pos_with_variance = original_pos + variance
        #     print(f"  Adding variance: {variance * 1000} mm")

        # else:
        #     pos_with_variance = original_pos
        
        # print("link: ",end_effector)
        # print("pos: ",pos_with_variance)
        # print("quat: ", quat)

        final_qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=quat,
        )
        
        # Check if IK succeeded
        if final_qpos is None:
            print(f"Attempt {attempt+1}/{max_retries}: IK failed for position {original_pos}")
            continue
        else:
            exit
        
    # Set gripper to current closed position
    final_qpos[-2:] = current_gripper_pos

    # Debug: print current and goal final_qpos
    current_qpos = franka.get_qpos()
    print(f"\n[DEBUG] Planning attempt {attempt+1}")
    print(f"  Current qpos: {current_qpos[:7]}")  # First 7 are arm joints
    print(f"  Goal qpos: {final_qpos[:7]}")
    print(f"  Goal position: {pos}")
    print(f"  Planner: {planner}")
    print(f"  Attached object: {attached_object}")

    # First, ensure gripper is firmly closed before moving
    current_arm_pos = franka.get_qpos()[:-2]
    # for _ in range(5):
    #     franka.control_dofs_position(current_arm_pos, motors_dof)
    #     franka.control_dofs_force(np.array([-3.0, -3.0]), fingers_dof)
    #     scene.step()

    # trajectory_steps = int(qpos_diff * 100) # set trajectory time depending on starting difference

    trajectory_steps = 100 # TODO Hard value for testing, remove once done.

    travel_time = 5 # seconds
    cur_time = 0 # seconds
    past_pose = np.zeros(7,)
    integral_pos_e = np.zeros(7,)

    time_step = travel_time / trajectory_steps

    # Convert tensors to numpy arrays for trajectory creation
    final_qpos_np = np.asarray(final_qpos[:7], dtype=np.float64)  # Only arm joints
    current_arm_pos_np = np.asarray(current_arm_pos, dtype=np.float64)
    
    trajectory = create_trajectory(final_qpos_np, current_arm_pos_np, travel_time, trajectory_steps)
    # print("trajectory: ", trajectory)
    # Gets this far error free

    for des_state in trajectory:
        franka.control_dofs_force(np.array([-3.0, -3.0]), fingers_dof)

        print("des_state: ", des_state)

        cur_qpos = franka.get_qpos()[:7]
        print("cur_qpos: ",cur_qpos)
        output_pose, past_pose, integral_pos_e = control(des_state, cur_qpos, time_step, cur_time, past_pose, integral_pos_e)
        print("output_pose: ",output_pose)
        # `output_pose` is a 7-element numpy array for the arm joints.
        # Log the full arm pose and send it directly to the arm motors.
        print("output_pose (arm 7-DOF):", output_pose)

        franka.control_dofs_position(output_pose, motors_dof)  # Move arm without altering grip
        scene.step()
        sleep(time_step) # Probably blocking, is that alright?
        cur_time += time_step

    pose_dif = abs(output_pose - final_qpos_np)
    print("pose_dif: ", pose_dif)
    pose_dif_norm = np.linalg.norm(pose_dif)
    print("pose_dif_norm: ", pose_dif_norm)
    # if abs(output_pose - final_qpos) < 0.1


def control(target_state, cur_pose, time_step, cur_time, past_pose, integral_pos_e):
    """
    Given a target state, control to that state.
    
    :param target_state: np.array of size (1,21), position: all joint position vals then vel, acc. numpy or tensor
    :param cur_pose: current pose np.array of size (1,7), 
    :param time_step: size of a time step, in seconds
    :param cur_time: current time in the trajectory
    :param past_pose: the previous pose

    output desired positions.
    """
    
    # Convert cur_pose to numpy array if it's a tensor
    cur_pose_np = np.asarray(cur_pose, dtype=np.float64)
    
    # Ensure target_state is numpy array
    target_state = np.asarray(target_state, dtype=np.float64)

    pos_e = target_state[:7] - cur_pose_np # 0-7
    # print("target_state[1:14]",target_state[1:14])
    # print("cur_pose_np: ",cur_pose_np," past_pose: ",past_pose," time_step: ",time_step)
    vel_e = target_state[7:14] - (cur_pose_np - past_pose) / time_step 
    integral_pos_e = pos_e * cur_time

    kp = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) # labeled joints 1-7
    kd = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) # labeled joints 1-7
    ki = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) # labeled joints 1-7
    max_integral = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Anti-windup limit

    target_change = kp*pos_e + kd*vel_e + ki*integral_pos_e

    output_pose = cur_pose_np + target_change

    return output_pose, cur_pose_np, integral_pos_e


def M(t = float):
            return np.array([[1, t, t**2, t**3, t**4, t**5],
                    [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
                    [0, 0, 2, 6*t, 12*t**2, 20*t**3]])


def create_trajectory(target_pose, start_pose, travel_time, total_steps):
    """
    Create a line trajectory from one pose to another. 
    Assuming desired final velocity and acceleration is 0.
    Assuming initial velocity and acceleration is 0.
    
    :param target_pose: target pose for each joint angle in radians as 7 long numpy list
    :param start_pose: same as above, as a target
    :param travel_time: travel time in seconds
    :param steps: total steps in the trajectory

    :return trajectory: np.ndarray depicting a line trajectory
    """
    
    # Ensure all inputs are numpy arrays
    target_pose = np.asarray(target_pose, dtype=np.float64)
    start_pose = np.asarray(start_pose, dtype=np.float64)

    vel_i = np.zeros(7,)
    # print("vel_i: ",vel_i)
    acc_i = np.zeros(7,)
    vel_f = np.zeros(7,)
    acc_f = np.zeros(7,)

    time_steps = np.linspace(0, travel_time, total_steps)

    T = np.concatenate((M(0),M(travel_time)))
    # print("T: ",T)

    T_inv = np.linalg.inv(T)

    trajectory_list = []  # Use list instead of array to avoid type issues

    for t in time_steps:
        # Build conditions matrix: shape (6, 7) for 6 boundary conditions and 7 joints
        conditions = np.array([
            start_pose,
            vel_i,
            acc_i,
            target_pose,
            vel_f,
            acc_f
        ])

        coeffs = T_inv @ conditions # Coefficients for each power in the quintic matrix, shape (6, 7)

        q_vals = M(t) @ coeffs  # Shape (3, 7): [position, velocity, acceleration]

        t_pose = q_vals[0, :]  # Position row
        t_vel = q_vals[1, :]   # Velocity row
        t_acc = q_vals[2, :]   # Acceleration row

        trajectory_list.append(np.hstack([t_pose, t_vel, t_acc]))
    
    state_array = np.array(trajectory_list)
    
    return state_array


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
    print("Place_Specific")
    b = blocks_state[block]
    # Convert Genesis positions to proper numpy arrays
    # bpos = np.asarray(b.get_pos(), dtype=np.float64)

    #move above target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.30]), attached_object=b)
    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.13]), attached_object=b)
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
    move_while_closed(franka, scene, position + np.array([0, 0, 0.25]), quat=quat, attached_object=b)

    # #lower to target position
    move_while_closed(franka, scene, position + np.array([0, 0, 0.13]), quat=quat, attached_object=b)

    # #release
    open_gripper_slight(franka, scene)

    # #move up
    move_while_open(franka, scene, position + np.array([0, 0, 0.3]), quat=quat, attached_object=b)

    open_gripper(franka, scene)





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

        elif action_name == "place_specific":
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
                print(f"ERROR: Unknown block for place_specific: {block}")
                continue
            PLACE_SPECIFIC(block, position, blocks_state, franka, scene)
        
        elif action_name == "place_specific_oriented":
            # place block at a specific position with a specific quaternion orientation
            # Perhaps it's a little excessive to write all of this out. I can speed this up later.
            block = parts[1]
            # Define target positions for each block

            # Radius of circle connecting block centers in special tower task
            r = ((0.025/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180)))) / math.cos(54*(math.pi/180))
            theta_r = 0*(math.pi/180)
            theta_b = 72*(math.pi/180)
            theta_y = 144*(math.pi/180)
            theta_g = 216*(math.pi/180)
            theta_o = 288*(math.pi/180)

            print("theta_o, degrees: ",theta_o*(180/math.pi))
            block_state = blocks_state[block]
            print("block:", block)
            ref_block = parts[2]
            block_pos = np.asarray(block_state.get_pos(), dtype=np.float64)
            block_quat = np.asarray(block_state.get_quat(), dtype=np.float64)
            print("block_orient:", block_quat)
            print("block_state:", block_state)

            target_positions = {
                # For Special Tower Task
                # only r1 and r2 needed, rest left for testing.
                'r1': np.array([0.3+r*math.cos(theta_r), 0.3+r*math.sin(theta_r), 0.0]),
                'g1': np.array([0.3+r*math.cos(theta_g), 0.3+r*math.sin(theta_g), 0.0]),
                'b1': np.array([0.3+r*math.cos(theta_b), 0.3+r*math.sin(theta_b), 0.0]),
                'o1': np.array([0.3+r*math.cos(theta_o), 0.3+r*math.sin(theta_o), 0.0]),
                'y1': np.array([0.3+r*math.cos(theta_y), 0.3+r*math.sin(theta_y), 0.0]),
                'r2': np.array([0.4, -0.2, 0.0]), 
                'g2': np.array([0.4, -0.2, 0.0]),
                'b2': np.array([0.4, -0.2, 0.0]),
                'o2': np.array([0.4, -0.2, 0.0]),
                'y2': np.array([0.4, -0.2, 0.0])
            }
            target_orientations = {
                # [w, x, y, z] unit quaternions
                'r1': np.array(R.from_euler('xyz', [math.pi,0.0,theta_r]).as_quat(scalar_first=True)),
                'g1': np.array(R.from_euler('xyz', [math.pi,0.0,theta_g]).as_quat(scalar_first=True)),
                'b1': np.array(R.from_euler('xyz', [math.pi,0.0,theta_b]).as_quat(scalar_first=True)),
                'o1': np.array(R.from_euler('xyz', [math.pi,0.0,theta_o]).as_quat(scalar_first=True)),
                'y1': np.array(R.from_euler('xyz', [math.pi,0.0,theta_y]).as_quat(scalar_first=True)),
                'r2': np.array([0.0, 0.0, 0.0, 1.0]),
                'g2': np.array([0.0, 0.0, 0.0, 1.0]),
                'b2': np.array([0.0, 0.0, 0.0, 1.0]),
                'o2': np.array([0.0, 0.0, 0.0, 1.0]),
                'y2': np.array([0.0, 0.0, 0.0, 1.0])
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
                print(f"ERROR: Unknown block for place_specific: {block}")
                continue
            print("quat for place_specific_oriented: ",quat)

            quat_as_euler = R.from_quat(quat, scalar_first=True).as_euler('xyz')

            print("place_specific_oriented quat as xyz euler in degrees: ",quat_as_euler*(180/math.pi))
            PLACE_SPECIFIC_ORIENTED(block, position, quat, blocks_state, franka, scene)

        elif action_name == "place_relative_oriented":
            block = parts[2]

            block_state = blocks_state[block]
            # print("block:", block)
            ref_block = parts[1]
            block_pos = np.asarray(block_state.get_pos(), dtype=np.float64)
            print("block_pos: ",block_pos)
            block_quat = np.asarray(block_state.get_quat(), dtype=np.float64)
            print("block quat: ",block_quat)
            # print("block_orient:", block_quat)
            # print("block_state:", block_state)

            # Determine new block position
            # See Jason's notes on the math + thought process behind this
            block_half_width = 0.04 # adjustable value to create extra space if desired
            w = ((block_half_width/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180))))
            # 0.02m is the half-width of a block, add more to create extra space
            theta_dif = 72*(math.pi/180)  # 72 degrees in radians
            r =  (w/2) / math.sin(theta_dif/2)
  

            theta2 = math.acos((w/2)/r)*(180/math.pi)
            beta = R.from_quat(block_quat, scalar_first=True).as_euler('xyz')[2]
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
            
            print("beta: ",beta," - theta2: ",theta2," = ",beta-theta2)
            print("phi: ",phi, " lambda: ",lamb," beta: ",beta)
            print("x_dif: ",x_dif," y_dif: ",y_dif)

            target_position = np.array([block_pos[0]+x_dif, block_pos[1]+y_dif, block_pos[2]])

            print("target_position for place_relative_oriented: ",target_position)
            target_quat = np.array(R.from_euler('xyz', [math.pi,0.0,theta_dif]).as_quat(scalar_first=True))
            # print("target_quat before: ",target_quat)

            # quaternion multiplication to combine orientations
            target_quat[0] = block_quat[0] * target_quat[0] - block_quat[1] * target_quat[1] - block_quat[2] * target_quat[2] - block_quat[3] * target_quat[3]
            target_quat[1] = block_quat[0] * target_quat[1] + block_quat[1] * target_quat[0] + block_quat[2] * target_quat[3] - block_quat[3] * target_quat[2]
            target_quat[2] = block_quat[0] * target_quat[2] - block_quat[1] * target_quat[3] + block_quat[2] * target_quat[0] + block_quat[3] * target_quat[1]
            target_quat[3] = block_quat[0] * target_quat[3] + block_quat[1] * target_quat[2] - block_quat[2] * target_quat[1] + block_quat[3] * target_quat[0]

            # print("target_quat after: ",target_quat)
            PLACE_SPECIFIC_ORIENTED(block, target_position, target_quat, blocks_state, franka, scene)

        elif action_name == "place_stacked_relative_oriented":
            # same code as above with halfed distance, angle, and a higher drop distance.
            # Can make the previous stuff a function with an output need be, but for now lets make this work.

            block = parts[2]

            block_state = blocks_state[block]
            # print("block:", block)
            ref_block = parts[1]
            block_pos = np.asarray(block_state.get_pos(), dtype=np.float64)
            print("block_pos: ",block_pos)
            block_quat = np.asarray(block_state.get_quat(), dtype=np.float64)
            print("block quat: ",block_quat)
            # print("block_orient:", block_quat)
            # print("block_state:", block_state)

            # Determine new block position
            # See Jason's notes on the math + thought process behind this
            block_half_width = 0.04 # adjustable value to create extra space if desired
            w = ((block_half_width/math.cos(math.pi/4)) / (math.cos(9*(math.pi/180))))
            # 0.02m is the half-width of a block, add more to create extra space
            theta_dif = 72*(math.pi/180)  # 72 degrees in radians
            r =  (w/2) / math.sin(theta_dif/2)
  

            theta2 = math.acos((w/2)/r)*(180/math.pi)
            beta = R.from_quat(block_quat, scalar_first=True).as_euler('xyz')[2]
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
            
            print("beta: ",beta," - theta2: ",theta2," = ",beta-theta2)
            print("phi: ",phi, " lambda: ",lamb," beta: ",beta)
            print("x_dif: ",x_dif," y_dif: ",y_dif)

            target_position = np.array([block_pos[0]+(x_dif/2), block_pos[1]+(y_dif/2), block_pos[2]+0.04]) # First change from code above

            print("target_position for place_relative_oriented: ",target_position)
            target_quat = np.array(R.from_euler('xyz', [math.pi,0.0,theta_dif/2]).as_quat(scalar_first=True)) # Second change from code above. That's it.
            # print("target_quat before: ",target_quat)

            # quaternion multiplication to combine orientations
            target_quat[0] = block_quat[0] * target_quat[0] - block_quat[1] * target_quat[1] - block_quat[2] * target_quat[2] - block_quat[3] * target_quat[3]
            target_quat[1] = block_quat[0] * target_quat[1] + block_quat[1] * target_quat[0] + block_quat[2] * target_quat[3] - block_quat[3] * target_quat[2]
            target_quat[2] = block_quat[0] * target_quat[2] - block_quat[1] * target_quat[3] + block_quat[2] * target_quat[0] + block_quat[3] * target_quat[1]
            target_quat[3] = block_quat[0] * target_quat[3] + block_quat[1] * target_quat[2] - block_quat[2] * target_quat[1] + block_quat[3] * target_quat[0]

            # print("target_quat after: ",target_quat)
            PLACE_SPECIFIC_ORIENTED(block, target_position, target_quat, blocks_state, franka, scene)


        else:
            print(f"ERROR: Unknown action: {action_name}")
    
    return True
