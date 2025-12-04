import os
import json
from pyperplan import grounding
from pyperplan.pddl.parser import Parser
from pyperplan.search.a_star import astar_search
from pyperplan.heuristics.relaxation import hMaxHeuristic

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

def plan_task(domain_file, problem_file):
    #load and parse PDDL files
    parser = Parser(domain_file, problem_file)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    #define task
    task = grounding.ground(problem)
    #solve using A*
    heuristic = hMaxHeuristic(task)
    solution = astar_search(task, heuristic)
    if solution:
        print(f"\n✓ Solution found with {len(solution)} steps\n")
        for i, action in enumerate(solution, 1):
            print(f"  {i}. {action}")
    else:
        print("\n No solution found")
    return solution
def main():
    solution = plan_task('adjacency_task_noclear.pddl', 'adjacency_problem_noclear.pddl')

    if solution:
        print(f"\n✓ Solution found with {len(solution)} steps\n")
        for i, action in enumerate(solution, 1):
            print(f"  {i}. {action}")
    else:
        print("\n No solution found")

if __name__ == "__main__":
    main()