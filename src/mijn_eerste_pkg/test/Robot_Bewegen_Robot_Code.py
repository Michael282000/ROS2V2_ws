#!/usr/bin/env python3
"""Execute Doosan robot instructions from Human_instructions.json.

Supported JSON instructions:
[
  {"task": "set", "param": "max-velocity", "value": 0.5},
  {"task": "set", "param": "max-acceleration", "value": 0.5},
  {"task": "move_joint", "value": [0, -30, 60, 0, 60, 0]},
  {"task": "wait", "value": 2.0}
]

Joint angles, velocity and acceleration use degrees, degrees/s and degrees/s².
Set ENABLE_MOTION = False to validate/read instructions without moving.
"""

import json
import time
from pathlib import Path

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, MoveJoint

# =========================== USER SETTINGS ============================
ROBOT_MODE = "virtual"  # "virtual" or "real"; informational safety check.
ROBOT_ID = "dsr01"
INSTRUCTION_FILE = Path("/home/bbrug1/ROS2V2_ws/src/mijn_eerste_pkg/test/Human.json")
ENABLE_MOTION = True

DEFAULT_VELOCITY_DEG_S = 5.0
DEFAULT_ACCELERATION_DEG_S2 = 5.0
SERVICE_TIMEOUT = 10.0
RESPONSE_TIMEOUT = 60.0
# ======================================================================

CONTROLLER_NAMESPACE = f"/{ROBOT_ID}/dsr_controller2"
POSITION_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/get_current_posj"
MOVE_JOINT_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/move_joint"


def wait_for_service(client, service_name):
    print(f"Waiting for service: {service_name}")
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
        raise RuntimeError(
            f"Service not found: {service_name}. Start the Doosan {ROBOT_MODE} "
            f"bringup and verify ROBOT_ID ({ROBOT_ID})."
        )


def call_service(node, client, request, description):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=RESPONSE_TIMEOUT)

    if not future.done():
        raise RuntimeError(f"Timed out while waiting for {description}.")

    response = future.result()
    if response is None or not response.success:
        raise RuntimeError(f"The controller rejected or failed {description}.")
    return response


def get_joint_positions(node, position_client):
    response = call_service(
        node,
        position_client,
        GetCurrentPosj.Request(),
        "the current joint-position response",
    )
    return [float(angle) for angle in response.pos]


def print_joint_positions(title, positions):
    print(f"{title}:")
    for index, angle in enumerate(positions, start=1):
        print(f"  J{index}: {angle:.3f}°")


def move_joint(node, move_client, joints_deg, velocity, acceleration):
    if not isinstance(joints_deg, list) or len(joints_deg) != 6:
        raise ValueError("move_joint.value must be a list with exactly six angles.")
    if velocity <= 0 or acceleration <= 0:
        raise ValueError("Velocity and acceleration must be greater than zero.")

    request = MoveJoint.Request()
    request.pos = [float(angle) for angle in joints_deg]
    request.vel = float(velocity)
    request.acc = float(acceleration)
    request.time = 0.0
    request.radius = 0.0
    request.mode = 0       # Absolute joint target.
    request.blend_type = 0
    request.sync_type = 0  # Wait until the movement has completed.

    print_joint_positions("Moving to", request.pos)
    print(f"  Velocity: {request.vel:.3f}°/s, acceleration: {request.acc:.3f}°/s²")
    call_service(node, move_client, request, "the MoveJoint command")
    print("MoveJoint completed.")


def load_instructions(filename):
    try:
        with filename.open("r", encoding="utf-8") as file:
            instructions = json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Instruction file not found: {filename.resolve()}")
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {filename}: {error}")

    if not isinstance(instructions, list):
        raise ValueError("The JSON root must be a list of instruction objects.")
    return instructions


def execute_instructions(node, position_client, move_client, instructions):
    velocity = DEFAULT_VELOCITY_DEG_S
    acceleration = DEFAULT_ACCELERATION_DEG_S2

    for line_number, instruction in enumerate(instructions, start=1):
        if not isinstance(instruction, dict):
            raise ValueError(f"Instruction {line_number} must be a JSON object.")

        task = instruction.get("task")
        print(f"\nInstruction {line_number}: {instruction}")

        if task == "set":
            param = instruction.get("param")
            try:
                value = float(instruction["value"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Instruction {line_number}: set requires a numeric value.")
            if value <= 0:
                raise ValueError(f"Instruction {line_number}: set value must be greater than zero.")

            if param == "max-velocity":
                velocity = value
                print(f"Maximum velocity set to {velocity:.3f}°/s")
            elif param == "max-acceleration":
                acceleration = value
                print(f"Maximum acceleration set to {acceleration:.3f}°/s²")
            else:
                raise ValueError(
                    f"Instruction {line_number}: unsupported set parameter '{param}'."
                )

        elif task == "move_joint":
            joints = instruction.get("value")
            if not ENABLE_MOTION:
                print_joint_positions("Motion disabled; would move to", joints)
                continue
            move_joint(node, move_client, joints, velocity, acceleration)
            print_joint_positions(
                "Current joint position", get_joint_positions(node, position_client)
            )

        elif task == "wait":
            try:
                seconds = float(instruction["value"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Instruction {line_number}: wait requires seconds as a number.")
            if seconds < 0:
                raise ValueError(f"Instruction {line_number}: wait time cannot be negative.")
            print(f"Waiting {seconds:.3f} seconds...")
            time.sleep(seconds)

        else:
            raise ValueError(f"Instruction {line_number}: unsupported task '{task}'.")


def main():
    if ROBOT_MODE not in {"virtual", "real"}:
        raise ValueError('ROBOT_MODE must be "virtual" or "real".')

    instructions = load_instructions(INSTRUCTION_FILE)
    rclpy.init()
    node = rclpy.create_node("execute_human_instructions")

    try:
        position_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)
        move_client = node.create_client(MoveJoint, MOVE_JOINT_SERVICE)

        print(f"Connected mode: {ROBOT_MODE}")
        print(f"Instruction file: {INSTRUCTION_FILE.resolve()}")
        wait_for_service(position_client, POSITION_SERVICE)
        if ENABLE_MOTION:
            wait_for_service(move_client, MOVE_JOINT_SERVICE)

        print_joint_positions(
            "Current joint position before instructions",
            get_joint_positions(node, position_client),
        )
        execute_instructions(node, position_client, move_client, instructions)
        print("\nAll instructions completed.")
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
