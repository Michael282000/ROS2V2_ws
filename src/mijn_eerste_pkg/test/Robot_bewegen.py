#!/usr/bin/env python3
"""Read Doosan joint positions and move safely to joint coordinates.

For a virtual test, launch the Doosan virtual/simulation bringup first and set
ROBOT_MODE = "virtual". After verifying the motion there, switch to "real"
and launch the real robot bringup. This script connects to whichever controller
is currently running; ROBOT_MODE is a safety/status check, not a simulator switch.
"""

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, MoveJoint

# =========================== USER SETTINGS ============================
# Start with "virtual" and a virtual Doosan bringup. Change to "real" only
# when the target and path have been checked in the virtual environment.
ROBOT_MODE = "real"  # "virtual" or "real"
ROBOT_ID = "dsr01"

# Absolute joint target: [J1, J2, J3, J4, J5, J6], in degrees.
TARGET_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MOVE_VELOCITY_DEG_S = 5.0
MOVE_ACCELERATION_DEG_S2 = 5.0

# False: read and display current joints and target, but do not move.
# True: execute MoveJoint on the connected virtual or real controller.
ENABLE_MOTION = True
# =======================================================================

SERVICE_TIMEOUT = 10.0
RESPONSE_TIMEOUT = 30.0

CONTROLLER_NAMESPACE = f"/{ROBOT_ID}/dsr_controller2"
POSITION_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/get_current_posj"
MOVE_JOINT_SERVICE = (
    f"/{ROBOT_ID}/dsr_controller2/motion/move_joint"
)


def wait_for_service(client, service_name):
    """Wait for one ROS service or raise a clear error."""
    print(f"Waiting for service: {service_name}")
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
        raise RuntimeError(
            f"Service not found: {service_name}. Start the Doosan {ROBOT_MODE} "
            f"bringup and verify ROBOT_ID ({ROBOT_ID})."
        )


def call_service(node, client, request, description):
    """Call a ROS service and wait for its response."""
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=RESPONSE_TIMEOUT)

    if not future.done():
        raise RuntimeError(f"Timed out while waiting for {description}.")

    response = future.result()
    if response is None or not response.success:
        raise RuntimeError(f"The controller rejected or failed {description}.")
    return response


def get_joint_positions(node, position_client):
    """Read and return the current six joint angles in degrees."""
    response = call_service(
        node,
        position_client,
        GetCurrentPosj.Request(),
        "the current joint-position response",
    )
    return [float(angle) for angle in response.pos]


def print_joint_positions(title, positions):
    """Print an ordered six-joint position in degrees."""
    print(f"{title}:")
    for index, angle in enumerate(positions, start=1):
        print(f"  J{index}: {angle:.3f}°")


def move_to_joint_coordinates(
    node,
    move_client,
    joints_deg,
    velocity_deg_s=MOVE_VELOCITY_DEG_S,
    acceleration_deg_s2=MOVE_ACCELERATION_DEG_S2,
):
    """Move absolutely and synchronously to [J1, J2, J3, J4, J5, J6] in degrees."""
    if len(joints_deg) != 6:
        raise ValueError("Joint coordinates must contain exactly six angles.")
    if velocity_deg_s <= 0 or acceleration_deg_s2 <= 0:
        raise ValueError("Velocity and acceleration must be greater than zero.")

    request = MoveJoint.Request()
    request.pos = [float(angle) for angle in joints_deg]
    request.vel = float(velocity_deg_s)
    request.acc = float(acceleration_deg_s2)
    request.time = 0.0
    request.radius = 0.0
    request.mode = 0       # Absolute joint target.
    request.blend_type = 0
    request.sync_type = 0  # Return only after the movement completes.

    print_joint_positions("Target joint coordinates", request.pos)
    call_service(node, move_client, request, "the MoveJoint command")
    print("MoveJoint completed.")


def main():
    if ROBOT_MODE not in {"virtual", "real"}:
        raise ValueError('ROBOT_MODE must be "virtual" or "real".')

    rclpy.init()
    node = rclpy.create_node("read_and_move_joint")

    try:
        position_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)
        move_client = node.create_client(MoveJoint, MOVE_JOINT_SERVICE)

        print(f"Connected mode: {ROBOT_MODE}")
        wait_for_service(position_client, POSITION_SERVICE)

        # This is always performed before any possible movement.
        current_joints = get_joint_positions(node, position_client)
        print_joint_positions("Current joint position before motion", current_joints)
        print_joint_positions("Configured target joint position", TARGET_JOINTS_DEG)

        if not ENABLE_MOTION:
            print("Motion is disabled: no MoveJoint command was sent.")
            return 0

        wait_for_service(move_client, MOVE_JOINT_SERVICE)
        move_to_joint_coordinates(
            node,
            move_client,
            TARGET_JOINTS_DEG,
            MOVE_VELOCITY_DEG_S,
            MOVE_ACCELERATION_DEG_S2,
        )

        final_joints = get_joint_positions(node, position_client)
        print_joint_positions("Current joint position after motion", final_joints)
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
