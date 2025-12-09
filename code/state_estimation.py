from dataclasses import dataclass
import numpy as np
import genesis
import os
import tempfile
from fast_downward_planner import plan_task, plan_task_from_string


@dataclass
class GripperState:
    """State tracking for the robot gripper."""
    pos: np.ndarray = None
    empty: bool = True
    holding: str = None
    
    def __post_init__(self):
        if self.pos is None:
            self.pos = np.array([0.0, 0.0, 0.0])


@dataclass
class TowerCube:
    """Representation of a block in the tower."""
    name: str
    position: np.ndarray
    on: str = None
    clear: bool = True
    ontable: bool = True


# Global gripper state instance
gripperState = GripperState()


class StateEstimation:
    """
    Manages state estimation for blocks world, including block positions,
    stacking relationships, and gripper state.
    """
    
    def __init__(self, problem):
        self.problem = problem
        self.block_info = []

    def set_scene_info(self, franka, blocks_state):
        """
        Initialize or update scene information from robot and blocks.
        
        Parameters:
            franka: Robot adapter
            blocks_state: Dictionary mapping block names to block entities
        
        Returns:
            tuple: (block_poses dict, end effector pose)
        """
        # Clear existing block info
        self.block_info = []
        
        # Update gripper position
        end_effector = franka.get_link("hand")
        EE_pose = end_effector.get_pos()
        gripperState.pos = np.array(EE_pose, dtype=np.float64)
        
        # Create block info for each block
        block_poses = {}
        for block_name in self.problem.block_names:
            if block_name in blocks_state:
                block = blocks_state[block_name]
                position = np.array(block.get_pos(), dtype=np.float64)
                new_cube = TowerCube(name=block_name, position=position)
                self.block_info.append(new_cube)
                block_poses[block_name] = position
            else:
                print(f"WARNING: Block {block_name} not found in blocks_state.")
        
        # Update gripper state after setting block positions
        self.update_gripper_state()
        
        return block_poses, EE_pose

    def estimate_tower(self):
        """
        Determine stacking relationships between all blocks.
        """
        for i in range(len(self.block_info)):
            for j in range(len(self.block_info)):
                if i != j:
                    self.update_stacked(self.block_info[i], self.block_info[j])


    def update_stacked(self, top_block, bottom_block):
        """
        Check if top_block is stacked on bottom_block and update their states.
        
        Parameters:
            top_block: TowerCube potentially on top
            bottom_block: TowerCube potentially on bottom
        """
        # Check horizontal alignment (x and y within threshold)
        horizontal_threshold = 0.03  # 3 cm tolerance
        horizontal_aligned = (
            abs(top_block.position[0] - bottom_block.position[0]) < horizontal_threshold and
            abs(top_block.position[1] - bottom_block.position[1]) < horizontal_threshold
        )
        
        # Check vertical alignment (z difference equals block height)
        block_height = 0.04  # 4 cm block height
        vertical_threshold = 0.002  # 2 mm tolerance
        z_diff = top_block.position[2] - bottom_block.position[2]
        vertical_aligned = abs(z_diff - block_height) < vertical_threshold
        
        # Update stacking relationship if aligned
        if horizontal_aligned and vertical_aligned:
            top_block.on = bottom_block.name
            bottom_block.clear = False
            top_block.ontable = False
    
    def update_gripper_state(self):
        """
        Update gripper state by checking distance to all blocks.
        Sets holding state if gripper is close enough to a block.
        """
        # Reset to empty state
        gripperState.empty = True
        gripperState.holding = None
        
        # Check if gripper is holding any block
        holding_threshold = 0.2  # 20 cm threshold
        for block in self.block_info:
            dist = np.linalg.norm(gripperState.pos - block.position)
            if dist < holding_threshold:
                gripperState.empty = False
                gripperState.holding = block.name
                return
    

    def parse_state_to_pddl(self):
        """
        Convert current state to PDDL predicates.
        
        Returns:
            list: PDDL predicate strings
        """
        pddl_facts = []
        
        # Add block predicates
        for block in self.block_info:
            if block.ontable:
                pddl_facts.append(f"(ontable {block.name})")
            if block.on is not None:
                pddl_facts.append(f"(on {block.name} {block.on})")
            if block.clear:
                pddl_facts.append(f"(clear {block.name})")
        
        # Add gripper predicates
        if gripperState.empty:
            pddl_facts.append("(handempty)")
        else:
            pddl_facts.append(f"(holding {gripperState.holding})")
        
        return pddl_facts

    def comp_PICK(self, block_name, current_facts):
        """
        Compute how predicates change after a PICK action.
        
        PICK action preconditions: (clear ?x), (ontable ?x), (handempty)
        PICK action effects: 
            - Add: (holding ?x), NOT (handempty)
            - Remove: (clear ?x), (ontable ?x), (handempty)
        
        Parameters:
            block_name: Name of the block being picked
            current_facts: Set of current PDDL facts (strings)
            
        Returns:
            Set of predicted facts after the action
        """
        predicted_facts = set(current_facts.copy())
        
        # Remove facts
        predicted_facts.discard(f"(clear {block_name})")
        predicted_facts.discard(f"(ontable {block_name})")
        predicted_facts.discard("(handempty)")
        
        # Add facts
        predicted_facts.add(f"(holding {block_name})")
        
        return predicted_facts
    
    def comp_STACK(self, block_x, block_y, current_facts):
        """
        Compute how predicates change after a STACK action.
        
        STACK action preconditions: (holding ?x), (clear ?y)
        STACK action effects:
            - Add: (on ?x ?y), (clear ?x), (handempty)
            - Remove: (holding ?x), (clear ?y)
        
        Parameters:
            block_x: Name of the block being stacked (the one held)
            block_y: Name of the block to stack on
            current_facts: Set of current PDDL facts (strings)
            
        Returns:
            Set of predicted facts after the action
        """
        predicted_facts = set(current_facts.copy())
        
        # Remove facts
        predicted_facts.discard(f"(holding {block_x})")
        predicted_facts.discard(f"(clear {block_y})")
        
        # Add facts
        predicted_facts.add(f"(on {block_x} {block_y})")
        predicted_facts.add(f"(clear {block_x})")
        predicted_facts.add("(handempty)")
        
        return predicted_facts
    
    def compare_states(self, predicted_facts, franka, blocks_state):
        """
        Compare predicted state (from action model) with actual observed state.
        
        Parameters:
            predicted_facts: Set of predicted PDDL facts after an action
            franka: Robot adapter
            blocks_state: Dictionary mapping block names to block entities
            
        Returns:
            dict with keys:
                - 'matches': bool - True if states match
                - 'missing_in_actual': set - facts predicted but not observed
                - 'extra_in_actual': set - facts observed but not predicted
                - 'predicted': set - predicted facts
                - 'actual': set - actual observed facts
        """
        # Update current state from scene
        for block_info in self.block_info:
            if block_info.name in blocks_state:
                block = blocks_state[block_info.name]
                block_info.position = np.array(block.get_pos(), dtype=np.float64)
                # Reset state for fresh estimation
                block_info.on = None
                block_info.clear = True
                block_info.ontable = True
        
        # Update gripper state
        end_effector = franka.get_link("hand")
        EE_pose = end_effector.get_pos()
        gripperState.pos = np.array(EE_pose, dtype=np.float64)
        gripperState.empty = True
        gripperState.holding = None
        self.update_gripper_state()
        
        # Estimate current tower configuration
        self.estimate_tower()
        
        # Get actual state
        actual_facts = set(self.parse_state_to_pddl())
        
        # Compare
        missing_in_actual = predicted_facts - actual_facts
        extra_in_actual = actual_facts - predicted_facts
        matches = len(missing_in_actual) == 0 and len(extra_in_actual) == 0
        
        return {
            'matches': matches,
            'missing_in_actual': missing_in_actual,
            'extra_in_actual': extra_in_actual,
            'predicted': predicted_facts,
            'actual': actual_facts
        }

    def replan_from_current_state(self, franka, blocks_state, domain_file, goal_config, output_problem_file=None):
        """
        Rerun the Fast Downward planner from the current tower state.
        
        Parameters:
            franka: Robot adapter
            blocks_state: Dictionary mapping block names to block entities
            domain_file: Path to the PDDL domain file
            goal_config: List of tuples representing goal (e.g., [('on', 'g', 'r'), ('on', 'b', 'g')])
            output_problem_file: Optional path to save the generated problem file
            
        Returns:
            List of action strings representing the new plan, or None if no solution found
        """
        # Update block positions from blocks_state
        for block_info in self.block_info:
            if block_info.name in blocks_state:
                block = blocks_state[block_info.name]
                block_info.position = np.array(block.get_pos(), dtype=np.float64)
                # Reset state for fresh estimation
                block_info.on = None
                block_info.clear = True
                block_info.ontable = True
        
        # Update gripper position
        end_effector = franka.get_link("hand")
        EE_pose = end_effector.get_pos()
        gripperState.pos = np.array(EE_pose, dtype=np.float64)
        
        # Update gripper state (checks which block is held)
        self.update_gripper_state()
        
        # Estimate current tower configuration
        self.estimate_tower()
        
        # Generate PDDL problem file from current state
        problem_content = self._generate_pddl_problem(goal_config)
        
        # Optionally save the problem file for debugging/inspection
        if output_problem_file is not None:
            with open(output_problem_file, 'w') as f:
                f.write(problem_content)
            print(f"Generated PDDL problem file: {output_problem_file}")
        
        print("Current state:")
        for block in self.block_info:
            print(f"  {block.name}: on={block.on}, clear={block.clear}, ontable={block.ontable}")
        
        # Print gripper state
        if gripperState.empty:
            print(f"  Gripper: empty")
        else:
            print(f"  Gripper: holding {gripperState.holding}")
        
        # Run Fast Downward planner
        print("\nRunning Fast Downward planner...")
        solution = plan_task_from_string(domain_file, problem_content)
        
        if solution is None:
            print("❌ No solution found")
        
        return solution
    
    def _generate_pddl_problem(self, goal_config):
        """
        Generate a PDDL problem file string from current state and goal.
        
        Parameters:
            goal_config: List of tuples representing goal (e.g., [('on', 'g', 'r'), ('on', 'b', 'g')])
            
        Returns:
            String containing the PDDL problem definition
        """
        # Get block names
        block_names = [block.name for block in self.block_info]
        
        # Start building PDDL problem
        pddl = "(define (problem replanned-blocks)\n"
        pddl += "  (:domain blocks-world)\n\n"
        
        # Objects
        pddl += "  (:objects " + " ".join(block_names) + " - block)\n\n"
        
        # Initial state
        pddl += "  (:init\n"
        init_facts = self.parse_state_to_pddl()
        for fact in init_facts:
            pddl += f"    {fact}\n"
        pddl += "  )\n\n"
        
        # Goal
        pddl += "  (:goal (and\n"
        for goal_fact in goal_config:
            if len(goal_fact) == 3:  # e.g., ('on', 'b1', 'b2')
                pddl += f"    ({goal_fact[0]} {goal_fact[1]} {goal_fact[2]})\n"
            elif len(goal_fact) == 2:  # e.g., ('clear', 'b1')
                pddl += f"    ({goal_fact[0]} {goal_fact[1]})\n"
        pddl += "  ))\n"
        pddl += ")\n"
        
        return pddl


    