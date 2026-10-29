"""
Picking two blocks out of a rail and placing them into a pocket plate with a
virtual ABB IRB 1200-7/0.7 robot and a Zimmer GEH6060 parallel gripper.

The robot behaves a little like a curious helper:
- Wakes up and waves hello
- Peeks at each block (a small nod) before it picks it
- Grips the block, lifts it straight out of the rail and hops over to the plate
- Sets the block down gently into its pocket and nods proudly
- Wiggles happily when both blocks are placed and returns home

Geometry (robot base coordinates in mm, z = 0 is the table top):
- The blocks stand in a rail running along the y-axis. Each block is 49 (x) by
  65 (y) by 50 (z) mm; the rail walls reach up to z = 108.6
- The pocket plate holds 2 x 3 pockets (70 x 65 mm) with the pocket floor at z = 29.5.
  Pockets are numbered row by row starting next to the rail:
  1 = (339.9, 180.9), 2 = (419.9, 180.9), 3 = (339.9, 260.9), ...
- The "Gripper" TCP sits at the fingertips, 104 mm in front of the flange.
  The jaws close along the flange y-axis, which is kept along the rail, so the
  fingers slide down the free rail channel next to the block
"""

import math
import os
import sys

import rerun as rr
import rerun.blueprint as rrb

import nova
from nova import api, run_program
from nova.actions import circular, io_write, joint_ptp, linear, wait
from nova.cell import virtual_controller
from nova.types import MotionSettings, Pose
from pydantic import Field

# Joint position of the robot in the Omniverse scene (degrees)
SCENE_JOINTS_DEG = (7.8, 75.5, -30.8, 3.7, 87.7, 4.3)

# Blocks in the rail: center (x, y) and bottom z
BLOCKS = [(374.1, -284.1, 28.5), (373.8, 68.0, 28.8)]
BLOCK_HEIGHT = 50
RAIL_TOP_Z = 108.6

# Pocket centers (x, y), numbered 1..6 row by row starting next to the rail
POCKETS = [(x, y) for y in (180.9, 260.9, 340.9) for x in (339.9, 419.9)]
POCKET_FLOOR_Z = 29.5


# Let the viewer find the rerun executable of this venv, even if the venv is not activated
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

# Keep a handle on the viewer so the program can stream the live robot state into it
rerun_viewer = nova.viewers.Rerun()


# Configure the robot program
@nova.program(
    id="abb_pick",  # Unique identifier of the program. If not provided, the function name will be used.
    name="Pick Blocks",  # Readable name of the program
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
    block1_pocket: int = Field(
        default=3, ge=1, le=6, description="Pocket (1-6) for block 1"
    ),
    block2_pocket: int = Field(
        default=4, ge=1, le=6, description="Pocket (1-6) for block 2"
    ),
    grip_depth: float = Field(
        default=20,
        gt=0,
        lt=BLOCK_HEIGHT,
        description="How far the fingertips reach down from the block top in mm",
    ),
    hover_height: float = Field(
        default=80,
        gt=0,
        description="Clearance of the carried block above the rail top in mm",
    ),
    tool_length: float = Field(
        default=104, gt=0, description="Distance from flange to fingertips in mm"
    ),
    gripper_io: str = Field(
        default="digital_out[0]",
        description="Digital output that closes the gripper (on) and opens it (off)",
    ),
    say_hello: bool = Field(
        default=False, description="Wave hello and wiggle happily at the end"
    ),
):
    """Pick both blocks from the rail and place them into the pocket plate."""
    if block1_pocket == block2_pocket:
        raise ValueError("Both blocks cannot go into the same pocket")

    cell = ctx.cell
    controller = await cell.controller("abby")
    cycle = ctx.cycle(extra={"app": "visual-studio-code"})

    travel = MotionSettings(tcp_velocity_limit=250)
    approach = MotionSettings(tcp_velocity_limit=60)
    gentle = MotionSettings(tcp_velocity_limit=25)
    playful = MotionSettings(tcp_velocity_limit=150)

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
        # Tool z points down, jaws (flange y) close along the rail (world y)
        return Pose((x, y, z, math.pi, 0, 0))

    def tilted(pose: Pose, about_x_deg: float = 0, about_y_deg: float = 0) -> Pose:
        return pose @ Pose(
            (0, 0, 0, math.radians(about_x_deg), math.radians(about_y_deg), 0)
        )

    def nod(pose: Pose, angle_deg: float = 12) -> list:
        return [
            linear(tilted(pose, about_y_deg=angle_deg), settings=playful),
            linear(pose, settings=playful),
        ]

    # The fingertips grip `grip_depth` below the block top, so a carried block
    # hangs `BLOCK_HEIGHT - grip_depth` below the fingertips
    hang = BLOCK_HEIGHT - grip_depth
    carry_z = RAIL_TOP_Z + hover_height + hang

    actions = [joint_ptp(home_joints, settings=travel), io_write(gripper_io, False)]

    if say_hello:
        hello = gripper_at(380, -60, 380)
        actions.append(linear(hello, settings=travel))
        for _ in range(2):  # Wave by tilting the gripper sideways
            actions += [
                linear(tilted(hello, about_x_deg=25), settings=playful),
                linear(tilted(hello, about_x_deg=-25), settings=playful),
            ]
        actions.append(linear(hello, settings=playful))

    targets = [block1_pocket, block2_pocket]
    for i, ((bx, by, bz), pocket) in enumerate(zip(BLOCKS, targets), start=1):
        px, py = POCKETS[pocket - 1]
        grip = gripper_at(bx, by, bz + BLOCK_HEIGHT - grip_depth)
        above_block = gripper_at(bx, by, carry_z)
        above_pocket = gripper_at(px, py, carry_z)
        place = gripper_at(px, py, POCKET_FLOOR_Z + 1 + hang)  # 1 mm above the floor

        actions += [
            linear(above_block, settings=travel),
            *nod(above_block),  # Peek at the block
            linear(grip, settings=approach),
            io_write(gripper_io, True),  # Close the gripper
            wait(0.5),
            linear(above_block, settings=approach),  # Lift straight out of the rail
            circular(  # Hop over to the pocket
                above_pocket,
                intermediate=gripper_at((bx + px) / 2, (by + py) / 2, carry_z + 60),
                settings=travel,
            ),
            linear(place @ Pose((0, 0, -15, 0, 0, 0)), settings=approach),
            linear(place, settings=gentle),  # Set it down gently
            io_write(gripper_io, False),  # Open the gripper
            wait(0.5),
            linear(above_pocket, settings=approach),
            *nod(above_pocket, angle_deg=-12),  # Proud nod
        ]

    if say_hello:
        happy = gripper_at(380, 260, carry_z + 60)
        actions.append(linear(happy, settings=travel))
        for _ in range(3):  # Happy wiggle about the gripper axis
            actions += [
                linear(
                    happy @ Pose((0, 0, 0, 0, 0, math.radians(30))), settings=playful
                ),
                linear(
                    happy @ Pose((0, 0, 0, 0, 0, math.radians(-30))), settings=playful
                ),
            ]
        actions.append(linear(happy, settings=playful))

    actions.append(joint_ptp(home_joints, settings=travel))

    # Start the cycle
    await cycle.start()

    # Plan the movements (shows in 3D viewer or creates an rrd file)
    joint_trajectory = await motion_group.plan(actions, tcp)

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

    print(
        f"Picking block 1 into pocket {block1_pocket} and block 2 into pocket {block2_pocket}..."
    )
    try:
        await motion_group.execute(joint_trajectory, tcp, actions=actions)
    finally:
        if bridge is not None:
            await bridge.stop_streaming()

    # Finish the cycle
    await cycle.finish()
    print("All blocks placed!")


if __name__ == "__main__":
    run_program(start)
