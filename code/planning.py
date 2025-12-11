import genesis as gs
import numpy as np
import torch
from typing import Any, Optional, List

from genesis.utils.misc import tensor_to_array
from robot_adapter import RobotAdapter


def _ensure_adapter(robot: Any, scene: Any) -> RobotAdapter:
    """
    Wrap raw Genesis robot in RobotAdapter if needed.
    
    This keeps PlannerInterface backward-compatible with callers that may
    still pass the raw Genesis entity.
    
    Parameters:
        robot: Robot entity (RobotAdapter or raw Genesis robot)
        scene: Genesis scene
    
    Returns:
        RobotAdapter instance
    """
    if isinstance(robot, RobotAdapter):
        return robot
    return RobotAdapter(robot, scene)

class PlannerInterface:
    """
    Interface for OMPL-based motion planning with Genesis robots.
    """
    
    def __init__(self, robot: Any, scene: Any):
        """
        Initialize planner interface.
        
        Parameters:
            robot: Robot entity (RobotAdapter or raw Genesis robot)
            scene: Genesis scene
        """
        # Ensure we have a RobotAdapter for stable interface
        self.robot = _ensure_adapter(robot, scene)
        self.scene = scene
        self.attached_object = None
        self.ignore_collisions = False

    def diagnose_bounds_violation(self, si, state):
        """
        Diagnose which joints violate bounds in the given state.
        
        Parameters:
            si: OMPL SpaceInformation object
            state: OMPL state to check
        """
        violated_bounds = []
        for i_q in range(self.robot.n_qs):
            val = state[i_q]
            low = si.getStateSpace().getBounds().low[i_q]
            high = si.getStateSpace().getBounds().high[i_q]
            if val < low or val > high:
                violated_bounds.append((i_q, val, low, high))
        gs.logger.warning(f"State violates bounds on joints: {violated_bounds}")

    def diagnose_valid_violation(self, state):
        """
        Diagnose collision violations for the given state.
        
        Parameters:
            state: OMPL state to check
        """
        self.robot.set_qpos(self._ompl_state_to_tensor(state))
        collision_pairs = self.robot.detect_collision()
        
        if collision_pairs.any() and len(collision_pairs) > 0:
            bad_links = set()
            for a, b in collision_pairs:
                bad_links.add(self.scene.rigid_solver.geoms[a].link.name)
                bad_links.add(self.scene.rigid_solver.geoms[b].link.name)
            gs.logger.warning(f"State causes collisions between links: {sorted(bad_links)}")

    def plan_path(
            self,
            qpos_goal,
            qpos_start=None,
            timeout=5.0,
            smooth_path=True,
            num_waypoints=100,
            attached_object=None,
            planner="RRTConnect",
            goal_tolerance=0.0005,
            ignore_collisions=False,
            waypoint_scale=0.75,
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters:
            qpos_goal: Target joint configuration
            qpos_start: Starting joint configuration (None = use current state)
            timeout: Maximum planning time in seconds
            smooth_path: Whether to smooth the path after finding solution
            num_waypoints: Number of waypoints to interpolate (-1 = auto-calculate)
            attached_object: Object held by gripper for collision checking
            planner: OMPL planner name ('RRTConnect', 'BITstar', etc.)
            goal_tolerance: Distance threshold for goal matching (0.0 = exact)
            ignore_collisions: Whether to skip collision checking
            waypoint_scale: Scaling factor for auto-calculated waypoints

        Returns:
            List of waypoints (each is a joint configuration tensor)
        """

        # Validate and import OMPL
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou
            ou.setLogLevel(ou.LOG_WARN)  # Changed from LOG_ERROR to see more diagnostic output
        except ImportError:
            gs.raise_exception(
                "Failed to import OMPL. Install instructions: "
                "https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html#optional-motion-planning"
            )

        supported_planners = ["PRM", "RRT", "RRTConnect", "RRTstar", "EST", "FMT", "BITstar", "ABITstar"]
        if planner not in supported_planners:
            gs.raise_exception(f"Planner '{planner}' not supported. Available: {supported_planners}")

        if self.robot._solver.n_envs > 0:
            gs.raise_exception("Motion planning not supported for batched environments")

        if self.robot.n_qs != self.robot.n_dofs:
            gs.raise_exception("Motion planning not supported for robots with free joints")

        # Get current configuration for restoration later
        qpos_cur = self.robot.get_qpos()

        # Process start and goal configurations
        if qpos_start is None:
            qpos_start = self.robot.get_qpos()
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (self.robot.n_qs,) or qpos_goal.shape != (self.robot.n_qs,):
            gs.raise_exception("Invalid shape for qpos_start or qpos_goal")

        # Process joint limits
        q_limit_lower = np.asarray(self.robot.q_limit[0], dtype=float)
        q_limit_upper = np.asarray(self.robot.q_limit[1], dtype=float)
        
        # Clamp to bounds (handles floating-point precision issues)
        qpos_start = np.clip(qpos_start, q_limit_lower, q_limit_upper)
        qpos_goal = np.clip(qpos_goal, q_limit_lower, q_limit_upper)
        
        # Auto-calculate waypoints if requested
        if num_waypoints is None or num_waypoints == -1:
            num_waypoints = self.calculate_waypoints_from_path_length(
                qpos_start, qpos_goal, waypoint_scale=waypoint_scale
            )
            print(f"Auto-calculated waypoints: {num_waypoints}")

        # Setup OMPL
        space = ob.RealVectorStateSpace(self.robot.n_qs)
        bounds = ob.RealVectorBounds(self.robot.n_qs)

        for i_q in range(self.robot.n_qs):
            bounds.setLow(i_q, float(q_limit_lower[i_q]))
            bounds.setHigh(i_q, float(q_limit_upper[i_q]))
        space.setBounds(bounds)
        
        ss = og.SimpleSetup(space)
        self.attached_object = attached_object
        self.ignore_collisions = ignore_collisions
        
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_ompl_state_valid))
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))

        # Create start and goal states
        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(self.robot.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        
        # Validate start and goal states
        si = ss.getSpaceInformation()
        if not si.satisfiesBounds(state_start.get()):
            gs.logger.warning("Start state out of bounds")
            self.diagnose_bounds_violation(si, state_start.get())

        if not si.satisfiesBounds(state_goal.get()):
            gs.logger.warning("Goal state out of bounds")
            self.diagnose_bounds_violation(si, state_goal)

        if not si.isValid(state_start.get()):
            gs.logger.warning("Start state invalid (collisions or constraints violated)")
            self.diagnose_valid_violation(state_start)

        if not si.isValid(state_goal.get()):
            gs.logger.warning("Goal state invalid (collisions or constraints violated)")
            self.diagnose_valid_violation(state_goal)

        # Configure OMPL problem
        if goal_tolerance > 0.0:
            ss.setStartAndGoalStates(state_start, state_goal, goal_tolerance)
        else:
            ss.setStartAndGoalStates(state_start, state_goal)
        ss.setup()

        # Solve
        solved = ss.solve(timeout)
        waypoints = []
        
        if solved:
            gs.logger.info("Path solution found")
            path = ss.getSolutionPath()
            
            if smooth_path:
                ss.simplifySolution()

            path.interpolate(num_waypoints)
            print(f"Path contains {path.getStateCount()} waypoints")
            waypoints = self._ompl_states_to_tensor_list(path.getStates())
        else:
            gs.logger.warning("Path planning failed - returning empty path")

        # Restore original configuration
        self.robot.set_qpos(qpos_cur)
        return waypoints

    def _is_ompl_state_valid(self, state):
        """
        Check if an OMPL state is valid (collision-free).
        
        Parameters:
            state: OMPL state to validate
        
        Returns:
            True if state is valid, False otherwise
        """
        if self.ignore_collisions:
            return True
            
        self.robot.set_qpos(self._ompl_state_to_tensor(state))
        collision_pairs = self.robot.detect_collision()

        if not len(collision_pairs):
            return True

        if not self.attached_object:
            return False

        return self._is_collision_with_attached_object(collision_pairs)

    def _is_collision_with_attached_object(self, collision_pairs):
        """
        Check if collisions are acceptable (gripper holding attached object).
        
        Parameters:
            collision_pairs: List of collision pairs to check
        
        Returns:
            True if all collisions are gripper-to-attached-object, False otherwise
        """
        finger_names = {"left_finger", "right_finger", "hand"}
        
        for a, b in collision_pairs:
            name_a = self.scene.rigid_solver.geoms[a].link.name
            name_b = self.scene.rigid_solver.geoms[b].link.name
            
            # Allow collisions between gripper and attached object
            gripper_to_object = (
                (name_a in finger_names and b == self.attached_object.idx) or
                (name_b in finger_names and a == self.attached_object.idx)
            )
            
            if not gripper_to_object:
                return False
        
        return True

    def _ompl_states_to_tensor_list(self, states) -> List[torch.Tensor]:
        """
        Convert list of OMPL states to list of PyTorch tensors.
        
        Parameters:
            states: List of OMPL states
        
        Returns:
            List of tensor configurations
        """
        return [self._ompl_state_to_tensor(state) for state in states]

    def _ompl_state_to_tensor(self, state) -> torch.Tensor:
        """
        Convert single OMPL state to PyTorch tensor.
        
        Parameters:
            state: OMPL state
        
        Returns:
            Joint configuration tensor
        """
        tensor = torch.empty(self.robot.n_qs, dtype=gs.tc_float, device=gs.device)
        for i in range(self.robot.n_qs):
            tensor[i] = state[i]
        return tensor

    @staticmethod
    def calculate_waypoints_from_path_length(
        qpos_start,
        qpos_goal,
        waypoints_per_radian=50,
        min_waypoints=30,
        max_waypoints=500,
        waypoint_scale=0.75
    ):
        """
        Calculate appropriate number of waypoints based on joint-space path length.
        
        Parameters:
            qpos_start: Starting joint configuration
            qpos_goal: Goal joint configuration
            waypoints_per_radian: Base number of waypoints per radian
            min_waypoints: Minimum waypoints regardless of path length
            max_waypoints: Maximum waypoints to prevent excessive computation
            waypoint_scale: Scaling factor to adjust waypoint density
            
        Returns:
            Calculated number of waypoints
        """
        # Calculate joint-space distance (L2 norm)
        path_length = np.linalg.norm(np.array(qpos_goal) - np.array(qpos_start))
        
        # Calculate waypoints: path_length * waypoints_per_radian * scale
        num_waypoints = int(path_length * waypoints_per_radian * waypoint_scale)
        
        # Clamp to bounds
        num_waypoints = max(min_waypoints, min(num_waypoints, max_waypoints))
        
        return num_waypoints