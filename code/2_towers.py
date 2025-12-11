import sys
import warnings
import numpy as np
import genesis as gs
from scenes import create_scene_6blocks, create_scene_stacked
from pddl_planner import plan_task
from motion_action_primitives import execute_pddl_plan

def main():
    """
    Main function to run the 2 towers problem.
    Sets up the scene, configures the robot, and executes the PDDL plan.
    """
    # Ensure Genesis is initialized before building scenes
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
    else:
        gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)

    # build the scene using the factory
    scene, franka, BlocksState = create_scene_6blocks()
    #scene, franka, BlocksState = create_scene_stacked()

    # set control gains
    # Note: the following values are tuned for achieving best behavior with Franka
    # Typically, each new robot would have a different set of parameters.
    # Sometimes high-quality URDF or XML file would also provide this and will be parsed.

    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # Execute PDDL plan
    execute_pddl_plan('2_towers/2_towers_domain.pddl', '2_towers/2_towers_problem.pddl', BlocksState, franka, scene)

if __name__ == "__main__":
    # No need for extra arguments in this code.
    main()


