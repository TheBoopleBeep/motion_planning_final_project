import sys
import numpy as np
import genesis as gs

# Initialize Genesis
gs.init(backend=gs.cpu, logging_level='Error')

from scenes import create_scene_10blocks
from state_estimation import state_estimation

# Build the scene
scene, franka, BlocksState = create_scene_10blocks()

# Set control gains
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# Create state estimator
class DummyProblem:
    def __init__(self, block_names):
        self.block_names = block_names

block_names = list(BlocksState.keys())
dummy_problem = DummyProblem(block_names)
estimator = state_estimation(dummy_problem)

# Set scene info
print("Setting scene info...")
estimator.set_scene_info(franka, BlocksState)
print(f"Found {len(estimator.block_info)} blocks")

# Test replanning
goal_config = [('on', 'g', 'r')]
print("\nTesting replanning with simple goal: g on r")
plan = estimator.replan_from_current_state(
    franka=franka,
    blocks_state=BlocksState,
    domain_file='ten_blocks/10_block_domain.pddl',
    goal_config=goal_config
)

if plan:
    print(f"\n✓ Plan generated with {len(plan)} steps:")
    for i, action in enumerate(plan, 1):
        print(f"  {i}. {action}")
else:
    print("\n✗ No plan generated")
