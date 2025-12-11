import numpy as np
from state_estimation import state_estimation

# Test comp_PICK
print("Testing comp_PICK:")
current_facts = {
    "(ontable r)", "(ontable g)", "(clear r)", "(clear g)", "(handempty)"
}
print(f"Current state: {current_facts}")

class DummyProblem:
    def __init__(self):
        self.block_names = ['r', 'g']

estimator = state_estimation(DummyProblem())
predicted = estimator.comp_PICK('r', current_facts)
print(f"After (pick r): {predicted}")
print()

# Test comp_STACK
print("Testing comp_STACK:")
current_facts = {
    "(ontable g)", "(clear g)", "(holding r)"
}
print(f"Current state: {current_facts}")
predicted = estimator.comp_STACK('r', 'g', current_facts)
print(f"After (stack r g): {predicted}")
print()

# Test chaining
print("Testing action sequence:")
current = {
    "(ontable r)", "(ontable g)", "(clear r)", "(clear g)", "(handempty)"
}
print(f"Initial: {current}")
after_pick = estimator.comp_PICK('g', current)
print(f"After (pick g): {after_pick}")
after_stack = estimator.comp_STACK('g', 'r', after_pick)
print(f"After (stack g r): {after_stack}")
