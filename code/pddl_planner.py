from pyperplan import grounding
from pyperplan.pddl.parser import Parser
from pyperplan.search.a_star import astar_search
from pyperplan.heuristics.relaxation import hMaxHeuristic

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

    return solution

solution = plan_task('6_block_domain.pddl', '6_block_problem.pddl')

if solution:
    print(f"\n✓ Solution found with {len(solution)} steps\n")
    for i, action in enumerate(solution, 1):
        print(f"  {i}. {action}")
else:
    print("\n No solution found")

