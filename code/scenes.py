"""Scene factory helpers.

Provide functions to create common demo scenes. Each factory returns a
tuple (scene, franka, blocks_state, end_effector) to be used by demos.
"""
from typing import Any, Dict, Tuple
import random
import time
random.seed(time.time())

import numpy as np
import genesis as gs
from robot_adapter import RobotAdapter


def _build_base_scene(camera_pos=(3, -1, 1.5), camera_lookat=(0.0, 0.0, 0.5)) -> gs.Scene:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=8),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=True,
    )
    return scene

def _elevate_robot_base(franka: Any) -> None:
    """Slightly raise robot base to avoid initial collisions."""
    base_pos = np.asarray(franka.get_pos(), dtype=float)
    new_pos = base_pos.copy()
    new_pos[2] += 0.01
    franka.set_pos(new_pos) 

def _rand_xy(base, noise=0.05):
        dx = random.uniform(-noise, noise)
        dy = random.uniform(-noise, noise)
        return (base[0] + dx, base[1] + dy, base[2])

def create_scene_6blocks() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create the default demo scene (layout 1).

    Returns:
        scene, franka_adapter, blocks_state, end_effector
    """
    scene = _build_base_scene()

    # basic geometry
    plane = scene.add_entity(gs.morphs.Plane())
    # add some random noise up to 5 cm in x/y


    posR = _rand_xy((0.65, 0.0, 0.02))
    posG = _rand_xy((0.65, 0.2, 0.02))
    posB = _rand_xy((0.65, 0.4, 0.02))
    posY = _rand_xy((0.45, 0.0, 0.02))
    posM = _rand_xy((0.45, 0.2, 0.02))
    posC = _rand_xy((0.45, 0.4, 0.02))

    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posM),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posC),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    # build scene (construct physics/visuals)
    scene.build()

    # initial robot pose (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # slightly raise robot base to avoid initial collisions
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state


def create_scene_10blocks() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create a scene for the special tower task

    Returns:
        scene, franka_adapter, blocks_state, end_effector
    """
    scene = _build_base_scene()

    # basic geometry
    plane = scene.add_entity(gs.morphs.Plane())
    # add some random noise up to 5 cm in x/y

    posR1 = _rand_xy((0.65, 0.0, 0.02))
    posG1 = _rand_xy((0.65, 0.2, 0.02))
    posB1 = _rand_xy((0.65, -0.2, 0.02))
    posY1 = _rand_xy((0.45, 0.0, 0.02))
    posO1 = _rand_xy((0.45, -0.2, 0.02))
    posR2 = _rand_xy((0.85, 0.0, 0.02))
    posG2 = _rand_xy((0.85, 0.2, 0.02))
    posB2 = _rand_xy((0.85, -0.2, 0.02))
    posY2 = _rand_xy((0.25, 0.0, 0.02))
    posO2 = _rand_xy((0.25, -0.2, 0.02))

    cubeR1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR1),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG1),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB1),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY1),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeO1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posO1),
        surface=gs.options.surfaces.Plastic(color=(1, 0.5, 0.0)),
    )
    cubeR2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR2),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG2),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB2),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY2),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeO2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posO2),
        surface=gs.options.surfaces.Plastic(color=(1, 0.5, 0.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    # build scene (construct physics/visuals)
    scene.build()

    # initial robot pose (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # slightly raise robot base to avoid initial collisions
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r1": cubeR1, "g1": cubeG1, "b1": cubeB1, "o1": cubeO1, "y1": cubeY1, "r2": cubeR2, "g2": cubeG2, "b2": cubeB2, "o2": cubeO2, "y2": cubeY2}

    return scene, franka, blocks_state


def create_scene_stacked() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create an alternative demo scene (layout 2) with cube positions. one on top of the other."""
    scene = _build_base_scene(camera_pos=(2.5, -1.2, 1.2), camera_lookat=(0.6, 0.0, 0.2))

    plane = scene.add_entity(gs.morphs.Plane())

    # slightly different positions
    startx, starty, _ = _rand_xy((0.45, 0.0, 0.02), noise=0.2) 
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.02)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.06)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.10)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.14)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )

    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.18)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.22)),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)
    scene.build()

    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state

def create_scene_adjacency() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create an alternative demo scene (layout 2) with cube positions. one on top of the other."""
    scene = _build_base_scene(camera_pos=(2.5, -1.2, 1.2), camera_lookat=(0.6, 0.0, 0.2))

    plane = scene.add_entity(gs.morphs.Plane())

    # slightly different positions
    pos1 = np.array([.3,  0 , 0.02])
    pos2 = np.array([.45, .0, 0.02])
    pos3 = np.array([.6, .0 , 0.02])
    pos4 = np.array([.3, -.2 , 0.02])
    pos5 = np.array([.45, -.2 , 0.02])
    pos6 = np.array([.6, -.2 , 0.02])
    cube1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos1),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cube2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos2),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cube3 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos3),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )

    cube4 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos4),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cube5 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos5),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cube6 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=pos6),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)
    scene.build()

    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"b1": cube1, "b2": cube2, "b3": cube3, "b4": cube4, "b5": cube5, "b6": cube6}

    return scene, franka, blocks_state