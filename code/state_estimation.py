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

@dataclass
class adjacentCube:
    """Representation of a block in the adjacency task."""
    name: str
    position: np.ndarray
    NNeighbor: str = None
    SNeighbor: str = None
    ENeighbor: str = None
    WNeighbor: str = None
    longDiag: str = None
    placed: bool = False

@dataclass
class specialCube:
    """Representation of a block in special tasks."""
    name: str
    position: np.ndarray
    ### Whatever else

# Global gripper state instance
gripperState = GripperState()


class StateEstimation:
    """
    Manages state estimation for blocks world, including block positions,
    stacking relationships, and gripper state.
    """
    
    def __init__(self, problem):
        self.problem = problem
        self.blocknames = []
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
        # Reset keys (you know in case something weird happens)
        self.blocknames = list(blocks_state.keys())
        # Update gripper position
        end_effector = franka.get_link("hand")
        EE_pose = end_effector.get_pos()
        gripperState.pos = np.array(EE_pose, dtype=np.float64)
        
        # Create block info for each block
        block_poses = {}
        for block_name in self.blocknames:
            if block_name in blocks_state:
                block = blocks_state[block_name]
                position = np.array(block.get_pos(), dtype=np.float64)
                
                # Create appropriate cube type based on problem
                if self.problem == "tower":
                    new_cube = TowerCube(name=block_name, position=position)
                elif self.problem == "adjacency":
                    new_cube = adjacentCube(name=block_name, position=position)
                elif self.problem == 'special':
                    new_cube = specialCube(name=block_name, position=position)
                else:
                    print(f"ERROR: Unknown problem type '{self.problem}'")
                    continue
                
                self.block_info.append(new_cube)
                block_poses[block_name] = position

            else:
                print(f"WARNING: Block {block_name} not found in blocks_state.")
        
        # Update gripper state after setting block positions
        self.update_gripper_state()
        
        return block_poses, EE_pose
    ####
    ## TOWER TASK STATE ESTIMATION METHODS
    ####

    def estimate_tower(self):
        """
        Determine stacking relationships between all blocks.
        """
        # Reset stacking relationships for all blocks
        for block in self.block_info:
            block.on = None
            block.clear = True
            block.ontable = True
        
        # Determine stacking relationships
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
        match self.problem:
            case "tower":
                for block in self.block_info:
                    if block.ontable:
                        pddl_facts.append(f"(ontable {block.name})")
                    if block.on is not None:
                        pddl_facts.append(f"(on {block.name} {block.on})")
                    if block.clear:
                        pddl_facts.append(f"(clear {block.name})")
            case "adjacency":
                for block in self.block_info:
                    if block.NNeighbor is not None:
                        pddl_facts.append(f"(Nneighbor {block.name} {block.NNeighbor})")
                    if block.SNeighbor is not None:
                        pddl_facts.append(f"(Sneighbor {block.name} {block.SNeighbor})")
                    if block.ENeighbor is not None:
                        pddl_facts.append(f"(Eneighbor {block.name} {block.ENeighbor})")
                    if block.WNeighbor is not None:
                        pddl_facts.append(f"(Wneighbor {block.name} {block.WNeighbor})")
                    if block.longDiag is not None:
                        pddl_facts.append(f"(longDiag {block.name} {block.longDiag})")
                    if block.placed:
                        pddl_facts.append(f"(placed {block.name})")
            case 'special':
                ##IMPLEMENT SPECIAL HERE
                pass
            case _:
                print(f"ERROR: Unknown problem type. Cannot parse state to PDDL.'{self.problem}'")
                return []
        
        # Add gripper predicates
        if gripperState.empty:
            pddl_facts.append("(handempty)")
        else:
            pddl_facts.append(f"(holding {gripperState.holding})")
        
        return pddl_facts

    ####
    ## TOWER TASK EXPECTED STATE UPDATE METHODS
    ####
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
    
    ####
    ## ADJACENCY TASK STATE ESTIMATION METHODS
    ####
    def adjacent_state_estimation(self):
        """
        Update adjacency relationships between all blocks.
        Make sure to call set_scene_info before running
        """
        print("\n\nHOWWWW\n\n")
        #reset old info
        self.estimate_adjacency()
        self.update_gripper_state()
        self.check_placed()

    def estimate_adjacency(self):
        """
        Determine adjacency relationships between all blocks.
        """
        # Reset all neighbor relationships before recalculating
        for block in self.block_info:
            block.NNeighbor = None
            block.SNeighbor = None
            block.ENeighbor = None
            block.WNeighbor = None
            block.longDiag = None
        
        # Calculate neighbor relationships
        for i in range(len(self.block_info)):
            for j in range(i,len(self.block_info)):
                if i != j:
                    self.find_relative_placement(self.block_info[i], self.block_info[j])

    def find_relative_placement(self, block_a, block_b):
        """
        Check relative placement of block_a with respect to block_b and update their states.
        
        Parameters:
            block_a: adjacentCube to check
            block_b: adjacentCube to check against
        """
        
        relative_pose = block_a.position - block_b.position
        x_diff, y_diff, _ = relative_pose

        cardinal_dist_thresh = 0.07  # 7 cm
        perp_dist_error = .03  # 3 cm
        long_diag_offset = .088 # 8.8 cm
        long_diag_thresh = 0.02  # 2 cm
        # block_a north of block_b means block_a is the north neighbor of block_b
        if  x_diff < cardinal_dist_thresh and x_diff > 0.02 and abs(y_diff) < perp_dist_error:
            block_a.NNeighbor = block_b.name
            block_b.SNeighbor = block_a.name
        # block_a south of block_b
        elif x_diff > -cardinal_dist_thresh and x_diff < -0.02 and abs(y_diff) < perp_dist_error:
            block_a.SNeighbor = block_b.name
            block_b.NNeighbor = block_a.name
        # block_a east of block_b
        elif y_diff < cardinal_dist_thresh and y_diff > 0.02 and abs(x_diff) < perp_dist_error:
            block_a.ENeighbor = block_b.name
            block_b.WNeighbor = block_a.name
        # block_a west of block_b
        elif y_diff > -cardinal_dist_thresh and y_diff < -0.02 and abs(x_diff) < perp_dist_error:
            block_a.WNeighbor = block_b.name
            block_b.ENeighbor = block_a.name
        # block_a is NE diagonal of block_b
        elif abs(x_diff - long_diag_offset) < long_diag_thresh and abs(y_diff - long_diag_offset) < long_diag_thresh:
            block_a.longDiag = block_b.name
        # block_a is SW diagonal of block_b (so block_b is NE diagonal of block_a)
        elif abs(x_diff + long_diag_offset) < long_diag_thresh and abs(y_diff + long_diag_offset) < long_diag_thresh:
            block_b.longDiag = block_a.name
            

    def check_placed(self):
        """
        Check if a block is in the target position and update its placed status.
        """
        for block in self.block_info:
            if np.linalg.norm(block.position - np.array([.4, .4, .02])) < 0.01:
                block.placed = True
            else:
                block.placed = False
    
        


    ####
    ## PDDL PLANNING AND STATE COMPARISON METHODS
    ####
    def compare_states(self, predicted_facts, franka):
        """
        Compare predicted state (from action model) with actual observed state.
        needs accompanying state update functions
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
        
        # Update gripper state
        end_effector = franka.get_link("hand")
        EE_pose = end_effector.get_pos()
        gripperState.pos = np.array(EE_pose, dtype=np.float64)
        gripperState.empty = True
        gripperState.holding = None
        self.update_gripper_state()

        
        # Estimate current tower configuration
        match self.problem:
            case "tower":
                self.estimate_tower()
                return
            case "adjacency":
                self.adjacent_state_estimation()
                return
            case 'special':
                ##IMPLEMENT SPECIAL HERE
                return
            case _:
                #assume adjacency
                self.adjacent_state_estimation()

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

    
    def _generate_pddl_problem(self, goal_config):
        """
        Generate a PDDL problem file string from current state and goal.
        
        Parameters:
            goal_config: List of tuples representing goal (e.g., [('on', 'g', 'r'), ('on', 'b', 'g')])
            
        Returns:
            String containing the PDDL problem definition
        """
        # Get block names and save as instance variable
        self.blocknames = [block.name for block in self.block_info]
        
        # Start building PDDL problem with task-specific naming
        if self.problem == "adjacency":
            pddl = "(define (problem adjacent-blocks)\n"
            pddl += "  (:domain adjacent-blocks-world-noclear)\n\n"
        else:
            pddl = "(define (problem replanned-blocks)\n"
            pddl += "  (:domain blocks-world)\n\n"
        
        # Objects
        pddl += "  (:objects " + " ".join(self.blocknames) + " - block)\n\n"
        
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
    

    def replan_from_current_state(self, franka, blocks_state, domain_file, output_problem_file=None):
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
        # Update scene info
        self.set_scene_info(franka, blocks_state)
        
        # Update gripper position
        end_effector = franka.get_link("hand")
        EE_pose = end_effector.get_pos()
        gripperState.pos = np.array(EE_pose, dtype=np.float64)
        
        # Update gripper state (checks which block is held)
        self.update_gripper_state()
        
        # Estimate state and set goal config based on problem type
        if self.problem == "tower":
            self.estimate_tower()
            # Define goal: build 12-block tower (matching 10_block_problem.pddl)
            goal_config = [
                ('on', 'g', 'r'),
                ('on', 'b', 'g'),
                ('on', 'y', 'b'),
                ('on', 'm', 'y'),
                ('on', 'c', 'm'),
                ('on', 'o', 'c'),
                ('on', 'p', 'o'),
                ('on', 'k', 'p'),
                ('on', 's', 'k'),
                ('on', 'l', 's'),
                ('on', 'n', 'l'),
                ('on', 'bb', 'n')
            ]
        elif self.problem == "adjacency":
            self.adjacent_state_estimation()
            goal_config = [
                ('Nneighbor', 'b2', 'b1'),
                ('Eneighbor', 'b3', 'b1'),
                ('longDiag', 'b4', 'b1'),
                ('Wneighbor', 'b5', 'b4'),
                ('Sneighbor', 'b6', 'b4')
            ]
        else:
            # Default to adjacency
            self.adjacent_state_estimation()
            goal_config = []
        
        # Generate PDDL problem file from current state
        problem_content = self._generate_pddl_problem(goal_config)
        
        # Save problem file for debugging
        debug_file = 'debug_problem.pddl'
        with open(debug_file, 'w') as f:
            f.write(problem_content)
        print(f"Generated PDDL problem saved to: {debug_file}")
        
        # Optionally save the problem file for debugging/inspection
        if output_problem_file is not None:
            with open(output_problem_file, 'w') as f:
                f.write(problem_content)
            print(f"Generated PDDL problem file: {output_problem_file}")
        
        print("Current state:")
        for block in self.block_info:
            if self.problem == "tower":
                print(f"  {block.name}: on={block.on}, clear={block.clear}, ontable={block.ontable}")
            elif self.problem == "adjacency":
                print(f"  {block.name}: NNeighbor={block.NNeighbor}, SNeighbor={block.SNeighbor}, ENeighbor={block.ENeighbor}, WNeighbor={block.WNeighbor}, longDiag={block.longDiag}, placed={block.placed}")
            else:
                print(f"  {block.name}: NNeighbor={block.NNeighbor}, SNeighbor={block.SNeighbor}, ENeighbor={block.ENeighbor}, WNeighbor={block.WNeighbor}, longDiag={block.longDiag}, placed={block.placed}")
        # Print gripper state
        if gripperState.empty:
            print(f"Gripper: empty")
        else:
            print(f"Gripper: holding {gripperState.holding}")
        
        # Run Fast Downward planner
        print("\nRunning planner...")
        solution = plan_task_from_string(domain_file, problem_content)
        
        if solution is None:
            print("No solution found")
        
        return solution


    