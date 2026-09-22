#!/usr/bin/env python3
"""Read Doosan joint positions and optionally move with MoveJoint.

Set ENABLE_MOTION to True only after verifying TARGET_JOINTS_DEG is safe.
"""

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, MoveJoint

# =========================== USER SETTINGS ============================
ROBOT_MODE = "real"  # "virtual" or "real"; used in status messages.
ROBOT_ID = "dsr01"

# Change these six target angles (degrees) only after checking the path.
TARGET_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MOVE_VELOCITY_DEG_S = 5.0
MOVE_ACCELERATION_DEG_S2 = 5.0

# Safety switch: the robot moves only when this is explicitly True.
ENABLE_MOTION = True
# =======================================================================

SERVICE_TIMEOUT = 10.0
RESPONSE_TIMEOUT = 30.0

CONTROLLER_NAMESPACE = f"/{ROBOT_ID}/dsr_controller2"
POSITION_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/get_current_posj"
MOVE_JOINT_SERVICE = f"/{ROBOT_ID}/motion/move_joint"


def wait_for_service(client, service_name):
    """Wait for one ROS service and raise a readable error if unavailable."""
    print(f"Waiting for service: {service_name}")
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
        raise RuntimeError(
            f"Service not found: {service_name}. Start Doosan bringup in "
            f"{ROBOT_MODE} mode and verify ROBOT_ID ({ROBOT_ID})."
        )


def call_service(node, client, request, description):
    """Call a ROS service and wait for its response with a timeout."""
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=RESPONSE_TIMEOUT)

    if not future.done():
        raise RuntimeError(f"Timed out while waiting for {description}.")

    response = future.result()
    if response is None or not response.success:
        raise RuntimeError(f"The controller rejected or failed {description}.")
    return response


def get_joint_positions(node, client):
    """Return the current six joint angles in degrees."""
    response = call_service(
        node, client, GetCurrentPosj.Request(), "the joint-position response"
    )
    return [round(angle, 3) for angle in response.pos]


def print_joint_positions(title, positions):
    """Print six joint angles in a consistent format."""
    print(f"{title}:")
    for index, angle in enumerate(positions, start=1):
        print(f"  J{index}: {angle:.3f}°")


def move_joint(node, client):
    """Send one absolute, synchronous MoveJoint command."""
    if len(TARGET_JOINTS_DEG) != 6:
        raise ValueError("TARGET_JOINTS_DEG must contain exactly six angles.")

    request = MoveJoint.Request()
    request.pos = [float(angle) for angle in TARGET_JOINTS_DEG]
    request.vel = float(MOVE_VELOCITY_DEG_S)
    request.acc = float(MOVE_ACCELERATION_DEG_S2)
    request.time = 0.0
    request.radius = 0.0
    request.mode = 0       # Absolute joint target.
    request.blend_type = 0
    request.sync_type = 0  # Wait until this motion has completed.

    print_joint_positions("Target joint position", request.pos)
    call_service(node, client, request, "the MoveJoint command")
    print("MoveJoint completed.")


def main():
    if ROBOT_MODE not in {"virtual", "real"}:
        raise ValueError('ROBOT_MODE must be "virtual" or "real".')

    rclpy.init()
    node = rclpy.create_node("read_and_move_joint")

    try:
        position_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)
        move_client = node.create_client(MoveJoint, MOVE_JOINT_SERVICE)

        print(f"Doosan mode: {ROBOT_MODE}")
        wait_for_service(position_client, POSITION_SERVICE)
        print_joint_positions("Current joint position", get_joint_positions(node, position_client))

        if not ENABLE_MOTION:
            print("Motion disabled: set ENABLE_MOTION = True after checking the target.")
            return 0

        wait_for_service(move_client, MOVE_JOINT_SERVICE)
        move_joint(node, move_client)
        print_joint_positions("Final joint position", get_joint_positions(node, position_client))
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
