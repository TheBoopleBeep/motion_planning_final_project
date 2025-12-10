import os
import json
import subprocess
import tempfile
import re

def save_plan_to_file(solution, output_file):
    """Save a PDDL solution to a JSON file."""
    if solution:
        # Convert solution actions to strings for JSON serialization
        plan_data = [str(action) for action in solution]
        with open(output_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
        print(f"Plan saved to {output_file}")
    else:
        print("No solution to save")

def load_plan_from_file(input_file):
    """Load a PDDL solution from a JSON file."""
    if not os.path.exists(input_file):
        print(f"Plan file {input_file} not found")
        return None
    
    with open(input_file, 'r') as f:
        plan_data = json.load(f)
    
    print(f"Loaded plan from {input_file} with {len(plan_data)} steps")
    return plan_data

def parse_fast_downward_plan(plan_file):
    """Parse the plan output file from Fast Downward."""
    actions = []
    if not os.path.exists(plan_file):
        return None
    
    with open(plan_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith(';') or not line:
                continue
            # Remove cost information if present (e.g., "(action) (1)")
            action = re.sub(r'\s*\(\d+\)$', '', line)
            actions.append(action)
    
    return actions if actions else None

def plan_task_from_string(domain_file, problem_string, search_algorithm='astar(lmcut())', alias=None):
    """
    Use Fast Downward to solve a planning task with problem provided as a string.
    
    Parameters:
        domain_file: Path to PDDL domain file
        problem_string: PDDL problem definition as a string
        search_algorithm: Fast Downward search configuration string
                         Examples:
                         - 'astar(lmcut())' - A* with LM-Cut heuristic (optimal)
                         - 'astar(blind())' - A* with blind heuristic
                         - 'lazy_greedy([ff()])' - Greedy best-first with FF heuristic (fast)
                         - 'eager_greedy([ff()])' - Eager greedy with FF heuristic
        alias: Use predefined alias instead of search_algorithm
               Examples: 'lama-first', 'seq-sat-lama-2011', 'seq-opt-bjolp'
    
    Returns:
        List of action strings representing the plan, or None if no solution found
    """
    # Create a temporary file for the problem string
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as temp_problem:
        temp_problem.write(problem_string)
        temp_problem_path = temp_problem.name
    
    try:
        # Call the regular plan_task function with the temporary file
        solution = plan_task(domain_file, temp_problem_path, search_algorithm, alias)
        return solution
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_problem_path):
            os.unlink(temp_problem_path)


def plan_task(domain_file, problem_file, search_algorithm='astar(lmcut())', alias=None):
    """
    Use Fast Downward to solve a planning task.
    
    Parameters:
        domain_file: Path to PDDL domain file
        problem_file: Path to PDDL problem file
        search_algorithm: Fast Downward search configuration string
                         Examples:
                         - 'astar(lmcut())' - A* with LM-Cut heuristic (optimal)
                         - 'astar(blind())' - A* with blind heuristic
                         - 'lazy_greedy([ff()])' - Greedy best-first with FF heuristic (fast)
                         - 'eager_greedy([ff()])' - Eager greedy with FF heuristic
        alias: Use predefined alias instead of search_algorithm
               Examples: 'lama-first', 'seq-sat-lama-2011', 'seq-opt-bjolp'
    
    Returns:
        List of action strings representing the plan, or None if no solution found
    """
    # Try to find Fast Downward installation
    fd_paths = [
        os.path.join('..',os.path.dirname(__file__), 'fast-downward-24.06.1', 'fast-downward.py')
    ]
    
    fd_command = None
    for path in fd_paths:
        try:
            if path.endswith('.py'):
                # Python script - need to run with python
                result = subprocess.run(['python3', path, '--help'], 
                                      capture_output=True, 
                                      timeout=5)
                fd_command = ['python3', path]
                break
            else:
                result = subprocess.run([path, '--help'], 
                                      capture_output=True, 
                                      timeout=5)
                fd_command = [path]
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if fd_command is None:
        print("ERROR: Fast Downward not found. Tried:")
        return None
    
    # Convert to absolute paths so Fast Downward can find them from temp directory
    domain_file = os.path.abspath(domain_file)
    problem_file = os.path.abspath(problem_file)
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        sas_file = os.path.join(temp_dir, 'output.sas')
        plan_file = os.path.join(temp_dir, 'sas_plan')
        
        # Build Fast Downward command
        if alias:
            # Use predefined alias
            cmd = fd_command + ['--alias', alias, domain_file, problem_file]
        else:
            # Use custom search configuration
            cmd = fd_command + [domain_file, problem_file, '--search', search_algorithm]
        
        try:
            # Run Fast Downward with suppressed output
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300  # 5 minute timeout
            )
            
            # Check if plan was found
            if os.path.exists(plan_file):
                solution = parse_fast_downward_plan(plan_file)
                if solution:
                    return solution
                else:
                    return []
            else:
                return None
                
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            return None

def main():
    """Test the Fast Downward planner."""
    # Example usage with different configurations
    print("Testing Fast Downward planner...")
    
    # Try with optimal A* search using LM-Cut heuristic
    solution = plan_task('adjacency_task.pddl', 'adjacency_problem.pddl', 
                        search_algorithm='astar(lmcut())')
    
    # Alternative: Use LAMA-first alias (fast, satisficing)
    # solution = plan_task('adjacency_task.pddl', 'adjacency_problem.pddl', 
    #                     alias='lama-first')
    
    if solution:
        print(f"\nFound solution with {len(solution)} actions")
    else:
        print("\nNo solution found")

if __name__ == "__main__":
    main()
