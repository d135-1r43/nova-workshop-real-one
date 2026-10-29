"""
Sanding an even (flat) bar with a virtual KUKA KR16 R2010 robot.

The motion imitates hand sanding:
- Approach the bar perpendicular to its surface and touch down gently
- Stroke back and forth along the bar in straight lines
- Lift off and return home

The start (home) position is the taught "home" pose of the flange in world
coordinates; the TCP "1" is the Sander, as configured on the robot controller.

Geometry (world coordinates, robot base on the table at the origin):
- The center of the sanding area is at (`surface_x`, `surface_y`, `surface_z`)
- The bar axis is turned by `bar_yaw_deg` about the world z-axis
  (0 = along the world x-axis, away from the robot; 90 = along the world y-axis)
- The face is tilted by `tilt_deg` about the bar axis (0 = horizontal, facing up)
- The sanding disc is the robot's TCP "1" (the Sander) and is tilted by
  `pad_angle_deg` against the surface across the bar
  (like a hand-held disc sander; this also keeps the robot wrist out of
  singularities)
"""

import asyncio
import math
import os
import sys

import nova
import rerun as rr
import rerun.blueprint as rrb
from nova import run_program
from nova.actions import collision_free, linear
from nova.types import MotionSettings, Pose
from pydantic import Field

# Taught "home" pose of the flange (TCP 2) in world: x, y, z in mm, rotation vector in rad
HOME_FLANGE_POSE = (477.7, 12.8, 1360.4, 2.8166, -0.0555, 1.2229)


# Let the viewer find the rerun executable of this venv, even if the venv is not activated
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

# Keep a handle on the viewer so the program can stream the live robot state into it
rerun_viewer = nova.viewers.Rerun()


# Configure the robot program
@nova.program(
    id="kuka_sander",  # Unique identifier of the program. If not provided, the function name will be used.
    name="Sand Bar",  # Readable name of the program
    viewer=rerun_viewer,
    # preconditions=nova.ProgramPreconditions(
    #     controllers=[
    #         virtual_controller(
    #             name="hansi2",
    #             manufacturer=api.models.Manufacturer.KUKA,
    #             type="kuka-kr16_r2010_2",
    #         )
    #     ],
    #     cleanup_controllers=False,
    # ),
)
async def start(
    ctx: nova.ProgramContext,
    count: int = Field(
        default=1, ge=1, le=10, description="Number of sanding passes over the bar"
    ),
    bar_length: float = Field(default=456, gt=0, description="Length of the bar in mm"),
    surface_x: float = Field(
        default=874,
        description="Distance of the bar center in front of the robot base in mm",
    ),
    surface_y: float = Field(
        default=186.7, description="Sideways offset of the bar center in mm"
    ),
    surface_z: float = Field(
        default=137.9,
        description="Height of the sanding surface above the robot base in mm",
    ),
    bar_yaw_deg: float = Field(
        default=0,
        description="Direction of the bar axis about the world z-axis in degrees",
    ),
    tilt_deg: float = Field(
        default=0, ge=-90, le=90, description="Tilt of the bar surface in degrees"
    ),
    pad_angle_deg: float = Field(
        default=10,
        ge=-30,
        le=30,
        description="Tilt of the sanding disc against the surface in degrees",
    ),
    strokes: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of strokes (forth, back, ...) along the bar",
    ),
):
    """Sand the surface of a tilted flat bar."""
    cell = ctx.cell
    controller = await cell.controller("robot")
    cycle = ctx.cycle(extra={"app": "visual-studio-code"})

    settings = MotionSettings(tcp_velocity_limit=50)

    motion_group = controller[0]
    tcp = "1"  # Sander, as configured on the robot controller
    # The home pose is taught for the flange, express it for the Sander TCP "1"
    home = Pose(HOME_FLANGE_POSE) @ await motion_group.tcp_offset(tcp)

    # Surface frame: tool z points into the surface, x runs along the bar, y across it
    surface = (
        Pose((surface_x, surface_y, surface_z, 0, 0, math.radians(bar_yaw_deg)))
        @ Pose((0, 0, 0, math.pi, 0, 0))
        @ Pose((0, 0, 0, math.radians(tilt_deg), 0, 0))
    )
    # Tilt the disc about its contact point
    disc_tilt = Pose((0, 0, 0, math.radians(pad_angle_deg), 0, 0))

    def disc_at(x: float) -> Pose:
        return surface @ Pose((x, 0, 0, 0, 0, 0)) @ disc_tilt

    lift = Pose((0, 0, -50, 0, 0, 0))  # 50 mm off the surface (against tool z)

    bar_start, bar_end = -bar_length / 2, bar_length / 2

    first_contact = disc_at(bar_start)
    actions = [
        collision_free(home, settings=settings),  # From wherever the robot is
        # Hover over bar start; collision_free keeps the wrist close to home
        # (cartesian_ptp flips A4/A6 by ~170 deg and winds up the cables)
        collision_free(first_contact @ lift, settings=settings),
        linear(first_contact, settings=settings),  # Touch down gently
    ]

    x = bar_start
    for _ in range(strokes):
        x = bar_end if x == bar_start else bar_start
        actions.append(linear(disc_at(x), settings=settings))

    actions += [
        linear(disc_at(x) @ lift, settings=settings),  # Lift off
        collision_free(home, settings=settings),
    ]

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

    print(f"Sanding a {bar_length} mm bar...")
    try:
        for i in range(count):
            print(f"Sanding pass {i + 1} of {count}")
            await motion_group.execute(joint_trajectory, tcp, actions=actions)
            print(f"Pass {i + 1} completed")
            await asyncio.sleep(1)
    finally:
        if bridge is not None:
            await bridge.stop_streaming()

    # Finish the cycle
    await cycle.finish()
    print("Sanding completed!")


if __name__ == "__main__":
    run_program(start)
