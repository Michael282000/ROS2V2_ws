#!/usr/bin/env python3
"""Read the current joint angles from a Doosan robot/controller.

Set ROBOT_MODE below to ``"virtual"`` for the Doosan virtual controller or
to ``"real"`` for the physical robot. Start the matching Doosan bringup
launch before running this script.
"""

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, MoveJoint


# =========================== USER SETTINGS ============================
# Change only this value when switching between the simulator and robot.
ROBOT_MODE = "virtual"  # "virtual" or "real"

# Must match the robot ID/namespace used in the Doosan bringup launch.
ROBOT_ID = "dsr01"

# =======================================================================

CONTROLLER_NAMESPACE = f"/{ROBOT_ID}/dsr_controller2"
POSITION_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/get_current_posj"
SERVICE_TIMEOUT_SECONDS = 10.0


def wait_for_service(client, name):
    """Wait once for a service and show an error if it is unavailable."""
    print(f"Waiting for {name} ...")
    if client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SECONDS):
        return True
    print(
        f"Could not find {name}. Start the Doosan bringup in {ROBOT_MODE} "
        "mode and check ROBOT_ID.",
    )
    return False


def read_joint_position(node, client, label):
    """Request and print the current six joint angles in degrees."""
    future = client.call_async(GetCurrentPosj.Request())
    rclpy.spin_until_future_complete(node, future)
    response = future.result()

    if response is None or not response.success:
        raise RuntimeError(
            "The controller could not return the joint position."
        )

    position = [round(angle, 3) for angle in response.pos]
    print(f"{label} joint position (degrees):")
    for number, angle in enumerate(position, start=1):
        print(f"  J{number}: {angle}")
    return position





def main():
    if ROBOT_MODE not in ("virtual", "real"):
        raise ValueError('ROBOT_MODE must be "virtual" or "real".')
    rclpy.init()
    node = rclpy.create_node("robot_bewegen")
    position_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)

    try:
        print(f"Doosan mode selected in this file: {ROBOT_MODE}")
        if not wait_for_service(position_client, POSITION_SERVICE):
            return 1
        read_joint_position(node, position_client, "Current")
        print("Done: no robot movement was commanded.")
        return 0
    except RuntimeError as error:
        print(f"Error: {error}")
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
