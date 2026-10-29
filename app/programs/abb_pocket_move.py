"""
Moving one block from one pocket of the pocket plate to another with the
ABB IRB 1200-7/0.7 robot and the Zimmer GEH6060 parallel gripper.

The gripper keeps the same orientation the whole time (tool z down, jaws along
the rail), so the wrist hardly turns and the cable to the gripper is not
wound up. Before executing, the program checks the planned joint travel and
refuses to move if a joint turns further than `max_joint_turn_deg`.

Pockets are numbered row by row starting next to the rail (see abb_pick.py):
1 = (339.9, 180.9), 2 = (419.9, 180.9), 3 = (339.9, 260.9), ...
"""

import math
import os
import sys

import nova
import rerun as rr
import rerun.blueprint as rrb
from nova import api, run_program
from nova.actions import io_write, joint_ptp, linear, wait
from nova.cell import virtual_controller
from nova.types import MotionSettings, Pose
from pydantic import Field

# Joint position of the robot in the Omniverse scene (degrees)
SCENE_JOINTS_DEG = (7.8, 75.5, -30.8, 3.7, 87.7, 4.3)

BLOCK_HEIGHT = 50

# Pocket centers (x, y), numbered 1..6 row by row starting next to the rail
POCKETS = [(x, y) for y in (180.9, 260.9, 340.9) for x in (339.9, 419.9)]
POCKET_FLOOR_Z = 29.5

# Let the viewer find the rerun executable of this venv, even if the venv is not activated
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

# Keep a handle on the viewer so the program can stream the live robot state into it
rerun_viewer = nova.viewers.Rerun()


# Configure the robot program
@nova.program(
    id="abb_pocket_move",  # Unique identifier of the program. If not provided, the function name will be used.
    name="Move Block Between Pockets",  # Readable name of the program
    viewer=rerun_viewer,
    preconditions=nova.ProgramPreconditions(
        controllers=[
            virtual_controller(
                name="abby",
                manufacturer=api.models.Manufacturer.ABB,
                type="abb-irb1200_07_7",
                position=[math.radians(a) for a in SCENE_JOINTS_DEG] + [0.0],
            )
        ],
        cleanup_controllers=False,
    ),
)
async def start(
    ctx: nova.ProgramContext,
    from_pocket: int = Field(
        default=6, ge=1, le=6, description="Pocket (1-6) to pick from"
    ),
    to_pocket: int = Field(
        default=2, ge=1, le=6, description="Pocket (1-6) to place into"
    ),
    grip_depth: float = Field(
        default=20,
        gt=0,
        lt=BLOCK_HEIGHT,
        description="How far the fingertips reach down from the block top in mm",
    ),
    hover_height: float = Field(
        default=100,
        gt=0,
        description="Height of the fingertips above the grip point while moving in mm",
    ),
    tool_length: float = Field(
        default=104, gt=0, description="Distance from flange to fingertips in mm"
    ),
    gripper_io: str = Field(
        default="digital_out[0]",
        description="Digital output that closes the gripper (on) and opens it (off)",
    ),
    max_joint_turn_deg: float = Field(
        default=90,
        gt=0,
        description="Refuse to move if any joint turns further than this away from home (protects the cable)",
    ),
):
    """Pick the block from one pocket and place it into another."""
    if from_pocket == to_pocket:
        raise ValueError("Pick and place pocket must be different")

    cell = ctx.cell
    controller = await cell.controller("abby")
    cycle = ctx.cycle(extra={"app": "visual-studio-code"})

    travel = MotionSettings(tcp_velocity_limit=150)
    approach = MotionSettings(tcp_velocity_limit=50)
    gentle = MotionSettings(tcp_velocity_limit=25)

    motion_group = controller[0]
    home_joints = tuple(math.radians(a) for a in SCENE_JOINTS_DEG)
    tcp = "Gripper"
    await motion_group.ensure_virtual_tcp(
        api.models.RobotTcp(
            id=tcp,
            name=tcp,
            position=[0, 0, tool_length],
            orientation=[0, 0, 0],
        )
    )

    def gripper_at(x: float, y: float, z: float) -> Pose:
        # Tool z points down, jaws (flange y) close along the rail (world y).
        # Turned 180 deg about tool z (same grip, the jaws are symmetric) so A6 stays near home
        return Pose((x, y, z, math.pi, 0, 0)) @ Pose((0, 0, 0, 0, 0, math.pi))

    grip_z = POCKET_FLOOR_Z + BLOCK_HEIGHT - grip_depth
    fx, fy = POCKETS[from_pocket - 1]
    tx, ty = POCKETS[to_pocket - 1]
    grip = gripper_at(fx, fy, grip_z)
    place = gripper_at(tx, ty, grip_z + 1)  # 1 mm above the floor
    up = Pose((0, 0, -hover_height, 0, 0, 0))  # Against tool z = straight up
    slow_zone = Pose((0, 0, -15, 0, 0, 0))

    actions = [
        joint_ptp(home_joints, settings=travel),
        io_write(gripper_io, False),  # Open the gripper
        linear(grip @ up, settings=travel),
        linear(grip @ slow_zone, settings=approach),
        linear(grip, settings=gentle),
        io_write(gripper_io, True),  # Close the gripper
        wait(0.5),
        linear(grip @ up, settings=approach),  # Lift straight out of the pocket
        linear(place @ up, settings=travel),
        linear(place @ slow_zone, settings=approach),
        linear(place, settings=gentle),  # Set it down gently
        io_write(gripper_io, False),  # Open the gripper
        wait(0.5),
        linear(place @ up, settings=approach),
        joint_ptp(home_joints, settings=travel),
    ]

    # Start the cycle
    await cycle.start()

    # Plan the movements (shows in 3D viewer or creates an rrd file)
    joint_trajectory = await motion_group.plan(actions, tcp)

    # Protect the cable: no joint may turn far away from home
    for i, home in enumerate(home_joints):
        turn = max(
            abs(math.degrees(p[i] - home)) for p in joint_trajectory.joint_positions
        )
        print(f"A{i + 1} turns up to {turn:.0f} deg away from home")
        if turn > max_joint_turn_deg:
            raise RuntimeError(
                f"A{i + 1} would turn {turn:.0f} deg (limit {max_joint_turn_deg:.0f})"
            )

    # Stream the live robot state into Rerun and follow it
    bridge = rerun_viewer.get_bridge()
    if bridge is not None:
        await bridge.start_streaming(motion_group)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Spatial3DView(name="3D Nova", background=[20, 22, 35]),
                rrb.TimePanel(timeline="time", play_state="following"),
            )
        )

    print(f"Moving the block from pocket {from_pocket} to pocket {to_pocket}...")
    try:
        await motion_group.execute(joint_trajectory, tcp, actions=actions)
    finally:
        if bridge is not None:
            await bridge.stop_streaming()

    # Finish the cycle
    await cycle.finish()
    print("Block moved!")


if __name__ == "__main__":
    run_program(start)
