#!/usr/bin/env python3
"""Execute Human_instructions.json on a Doosan ROS 2 controller.

Supported JSON tasks:
  {"task":"set", "param":"velocity", "value":0.5}
  {"task":"ptp", "value":[0,-30,60,0,60,0], "vel":0.5}
  {"task":"move-ee", "mode":"ptp",
   "pose":{"pos":[0.40,-0.10,0.25], "quat":[0,0.7071,0.7071,0]},
   "config":{"shoulder":"SR","elbow":"EU","wrist":"WN"}, "vel":0.25}
  {"task":"wait", "value":2}

JSON poses use metres and quaternions [x, y, z, w]. Doosan MoveJointx
uses millimetres and ZYZ Euler degrees. The complete script is preflighted
before the first motion command.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, Ikin, MoveJoint, MoveJointx

# =========================== USER SETTINGS ============================
ROBOT_ID = "dsr01"
INSTRUCTION_FILE = Path("/home/bbrug1/ROS2V2_ws/src/mijn_eerste_pkg/test/Human.json")
ENABLE_MOTION = True

# Initial modal values. PTP values are fractions of the configured PTP maxima.
BACKEND_INITIAL_VELOCITY = 0.01
BACKEND_INITIAL_ACCELERATION = 0.01
BACKEND_INITIAL_MAX_VELOCITY = 1.0
BACKEND_INITIAL_MAX_ACCELERATION = 1.0

# Verify these controller-specific values before using a real robot.
DOOSAN_PTP_MAX_VELOCITY_DEG_S = 180.0
DOOSAN_PTP_MAX_ACCELERATION_DEG_S2 = 180.0

# Fill only after experimentally validating the native Doosan solution-space
# number (0..7) for this model. An omitted config selects the closest solution.
CANONICAL_CONFIG_TO_DOOSAN_SOL = {
    CANONICAL_CONFIG_TO_DOOSAN_SOL = {
    ("SL","ED","WN"):0, ("SL","ED","WF"):1, ("SL","EU","WN"):2, ("SL","EU","WF"):3,
    ("SR","ED","WN"):4, ("SR","ED","WF"):5, ("SR","EU","WN"):6, ("SR","EU","WF"):7,
}
}

SERVICE_TIMEOUT = 10.0
POSITION_RESPONSE_TIMEOUT = 10.0
IK_RESPONSE_TIMEOUT = 10.0
MOTION_TIMEOUT_MARGIN = 30.0
MOTION_TIMEOUT_MINIMUM = 30.0
# =======================================================================

CONTROLLER_NAMESPACE = f"/{ROBOT_ID}/dsr_controller2"
POSITION_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/get_current_posj"
IKIN_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/ikin"
MOVE_JOINT_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/move_joint"
MOVE_JOINTX_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/move_jointx"


@dataclass
class ModalValues:
    velocity: float
    acceleration: float
    max_velocity: float
    max_acceleration: float


class InterpreterError(Exception):
    def __init__(self, line, code, message):
        super().__init__(message)
        self.line, self.code, self.message = line, code, message


def emit(line, status, code=None, payload=None):
    print(json.dumps({"line": line, "status": status, "code": code, "payload": payload}))


def finite_positive(value, line, field):
    if isinstance(value, bool):
        raise InterpreterError(line, "E_PARSE", f"{field} must be a number.")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InterpreterError(line, "E_PARSE", f"{field} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise InterpreterError(line, "E_PARSE", f"{field} must be finite and greater than zero.")
    return value


def initial_modal_values():
    values = ModalValues(
        finite_positive(BACKEND_INITIAL_VELOCITY, 0, "BACKEND_INITIAL_VELOCITY"),
        finite_positive(BACKEND_INITIAL_ACCELERATION, 0, "BACKEND_INITIAL_ACCELERATION"),
        finite_positive(BACKEND_INITIAL_MAX_VELOCITY, 0, "BACKEND_INITIAL_MAX_VELOCITY"),
        finite_positive(BACKEND_INITIAL_MAX_ACCELERATION, 0, "BACKEND_INITIAL_MAX_ACCELERATION"),
    )
    if values.velocity > values.max_velocity or values.acceleration > values.max_acceleration:
        raise ValueError("Initial velocity/acceleration cannot exceed their configured caps.")
    return values


def report_initial_values(values):
    print(json.dumps({
        "event": "initial_modal_values",
        "velocity": values.velocity,
        "acceleration": values.acceleration,
        "max_velocity": values.max_velocity,
        "max_acceleration": values.max_acceleration,
        "ptp_backend_max_velocity_deg_s": DOOSAN_PTP_MAX_VELOCITY_DEG_S,
        "ptp_backend_max_acceleration_deg_s2": DOOSAN_PTP_MAX_ACCELERATION_DEG_S2,
    }))


def motion_value(instruction, line, short_name, long_name, default):
    present = [name for name in (short_name, long_name) if name in instruction]
    if len(present) > 1:
        raise InterpreterError(line, "E_PARSE", f"Use only one of {short_name} and {long_name}.")
    return finite_positive(instruction[present[0]], line, present[0]) if present else default


def validate_ptp(line, velocity, acceleration, modal):
    if velocity > 1.0 or velocity > modal.max_velocity:
        raise InterpreterError(line, "E_LIMIT", "PTP velocity exceeds its cap or the portable range 0–1.")
    if acceleration > 1.0 or acceleration > modal.max_acceleration:
        raise InterpreterError(line, "E_LIMIT", "PTP acceleration exceeds its cap or the portable range 0–1.")


def parse_joints(value, line):
    if not isinstance(value, list) or len(value) != 6:
        raise InterpreterError(line, "E_PARSE", "PTP value must be a six-element joint-angle list.")
    try:
        joints = [float(angle) for angle in value]
    except (TypeError, ValueError):
        raise InterpreterError(line, "E_PARSE", "Every joint angle must be numeric.")
    if not all(math.isfinite(angle) for angle in joints):
        raise InterpreterError(line, "E_PARSE", "Every joint angle must be finite.")
    return joints


def parse_config(value, line):
    if not isinstance(value, dict) or set(value) != {"shoulder", "elbow", "wrist"}:
        raise InterpreterError(line, "E_PARSE", "config needs shoulder, elbow, and wrist fields.")
    config = (value["shoulder"], value["elbow"], value["wrist"])
    if config[0] not in {"SL", "SR"} or config[1] not in {"EU", "ED"} or config[2] not in {"WN", "WF"}:
        raise InterpreterError(line, "E_PARSE", "Invalid canonical IK configuration.")
    if config not in CANONICAL_CONFIG_TO_DOOSAN_SOL:
        raise InterpreterError(line, "E_UNSUPPORTED", "No experimentally validated Doosan sol mapping for this config.")
    sol = CANONICAL_CONFIG_TO_DOOSAN_SOL[config]
    if not isinstance(sol, int) or not 0 <= sol <= 7:
        raise InterpreterError(line, "E_UNSUPPORTED", "Configured Doosan solution-space mapping must be an integer 0–7.")
    return config, sol


def quaternion_to_zyz_degrees(quat, line):
    if not isinstance(quat, list) or len(quat) != 4:
        raise InterpreterError(line, "E_PARSE", "pose.quat must be [x, y, z, w].")
    try:
        x, y, z, w = (float(value) for value in quat)
    except (TypeError, ValueError):
        raise InterpreterError(line, "E_PARSE", "Quaternion values must be numeric.")
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise InterpreterError(line, "E_PARSE", "Quaternion values must be finite.")
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if abs(norm - 1.0) > 1e-3:
        raise InterpreterError(line, "E_PARSE", "Quaternion norm deviates from 1 by more than 1e-3.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x*x + y*y)
    beta = math.acos(max(-1.0, min(1.0, r22)))
    if abs(math.sin(beta)) > 1e-9:
        alpha = math.atan2(r12, r02)
        gamma = math.atan2(r21, -r20)
    elif r22 > 0:
        alpha = math.atan2(r10, r00)
        gamma = 0.0
    else:
        alpha = math.atan2(-r01, -r00)
        gamma = 0.0
    return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]


def parse_pose(value, line):
    if not isinstance(value, dict) or set(value) != {"pos", "quat"}:
        raise InterpreterError(line, "E_PARSE", "pose needs pos and quat fields.")
    pos = value["pos"]
    if not isinstance(pos, list) or len(pos) != 3:
        raise InterpreterError(line, "E_PARSE", "pose.pos must be [x, y, z] in metres.")
    try:
        xyz = [float(component) for component in pos]
    except (TypeError, ValueError):
        raise InterpreterError(line, "E_PARSE", "pose.pos values must be numeric.")
    if not all(math.isfinite(component) for component in xyz):
        raise InterpreterError(line, "E_PARSE", "pose.pos values must be finite.")
    return [component * 1000.0 for component in xyz] + quaternion_to_zyz_degrees(value["quat"], line)


def parse_program(instructions, initial):
    if not isinstance(instructions, list):
        raise InterpreterError(0, "E_PARSE", "The JSON root must be a list of instructions.")
    modal = ModalValues(**vars(initial))
    plan = []
    for line, instruction in enumerate(instructions, start=1):
        if not isinstance(instruction, dict) or not isinstance(instruction.get("task"), str):
            raise InterpreterError(line, "E_PARSE", "Each instruction must be an object with a string task.")
        task = instruction["task"]
        if task == "set":
            if set(instruction) != {"task", "param", "value"} or instruction["param"] not in {
                "velocity", "acceleration", "max-velocity", "max-acceleration"
            }:
                raise InterpreterError(line, "E_PARSE", "Invalid set instruction.")
            param = instruction["param"]
            value = finite_positive(instruction["value"], line, "value")
            if param == "velocity":
                if value > modal.max_velocity:
                    raise InterpreterError(line, "E_LIMIT", "velocity exceeds max-velocity.")
                modal.velocity = value
            elif param == "acceleration":
                if value > modal.max_acceleration:
                    raise InterpreterError(line, "E_LIMIT", "acceleration exceeds max-acceleration.")
                modal.acceleration = value
            elif param == "max-velocity":
                if value < modal.velocity:
                    raise InterpreterError(line, "E_LIMIT", "max-velocity cannot be below the active velocity.")
                modal.max_velocity = value
            else:
                if value < modal.acceleration:
                    raise InterpreterError(line, "E_LIMIT", "max-acceleration cannot be below the active acceleration.")
                modal.max_acceleration = value
            plan.append((line, "set", {"param": param, "value": value}))
        elif task in {"ptp", "move_joint"}:
            allowed = {"task", "value", "vel", "velocity", "acc", "acceleration"}
            if not set(instruction) <= allowed or "value" not in instruction:
                raise InterpreterError(line, "E_PARSE", "Invalid PTP instruction.")
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            validate_ptp(line, velocity, acceleration, modal)
            plan.append((line, "ptp", (parse_joints(instruction["value"], line), velocity, acceleration)))
        elif task == "move-ee":
            allowed = {"task", "mode", "pose", "config", "vel", "velocity", "acc", "acceleration"}
            if not set(instruction) <= allowed or instruction.get("mode") not in {"ptp", "lin"} or "pose" not in instruction:
                raise InterpreterError(line, "E_PARSE", "Invalid move-ee instruction.")
            if instruction["mode"] == "lin":
                raise InterpreterError(line, "E_UNSUPPORTED", "lin cannot faithfully validate a fixed IK branch on this backend.")
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            validate_ptp(line, velocity, acceleration, modal)
            sol = parse_config(instruction["config"], line)[1] if "config" in instruction else None
            plan.append((line, "move-ee-ptp", (parse_pose(instruction["pose"], line), sol, velocity, acceleration)))
        elif task == "wait":
            if set(instruction) != {"task", "value"}:
                raise InterpreterError(line, "E_PARSE", "Invalid wait instruction.")
            plan.append((line, "wait", finite_positive(instruction["value"], line, "value")))
        else:
            raise InterpreterError(line, "E_UNSUPPORTED", f"Unsupported task: {task}.")
    return plan


def wait_for_service(client, service_name):
    print(f"Waiting for service: {service_name}")
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
        raise RuntimeError(f"Service not found: {service_name}.")


def call_service(node, client, request, description, timeout):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done():
        raise RuntimeError(f"Timed out while waiting for {description}.")
    response = future.result()
    if response is None or not response.success:
        raise RuntimeError(f"The controller rejected or failed {description}.")
    return response


def current_joints(node, position_client):
    response = call_service(node, position_client, GetCurrentPosj.Request(), "the current joint-position response", POSITION_RESPONSE_TIMEOUT)
    return [float(angle) for angle in response.pos]


def ik_solution(node, ik_client, pose, sol):
    request = Ikin.Request()
    request.pos, request.sol_space, request.ref = pose, sol, 0
    future = ik_client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=IK_RESPONSE_TIMEOUT)
    response = future.result() if future.done() else None
    if response is None or not response.success:
        return None
    # Ikin.srv returns the six joint values in conv_posj, not pos.
    return [float(angle) for angle in response.conv_posj]


def angular_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def resolve_move_ee_plan(node, position_client, ik_client, plan):
    current = current_joints(node, position_client)
    resolved = []
    for line, task, payload in plan:
        if task == "ptp":
            current = payload[0]
        elif task == "move-ee-ptp":
            pose, requested_sol, velocity, acceleration = payload
            print(json.dumps({
                "event": "move_ee_preflight",
                "line": line,
                "doosan_pose_mm_zyz_deg": pose,
                "requested_sol": requested_sol,
            })) # kan uiteindelijk verwijderd worden
            candidates = [(sol, ik_solution(node, ik_client, pose, sol)) for sol in range(8)]
            valid = [(sol, joints) for sol, joints in candidates if joints is not None]
            print(json.dumps({
                "event": "ik_candidates",
                "line": line,
                "valid_solutions": [sol for sol, _ in valid],
            }))
            if not valid:
                raise InterpreterError(line, "E_UNREACHABLE", "Target pose has no IK solution.")
            if requested_sol is None:
                sol, joints = min(valid, key=lambda item: angular_distance(current, item[1]))
            else:
                joints = next((candidate for sol, candidate in valid if sol == requested_sol), None)
                if joints is None:
                    raise InterpreterError(line, "E_UNREACHABLE_CONFIG", "Requested IK branch has no solution at target.")
                sol = requested_sol
            resolved.append((line, task, (pose, sol, velocity, acceleration)))
            current = joints
            continue
        resolved.append((line, task, payload))
    return resolved


def ptp_timeout(start, target, velocity):
    max_delta = max(abs(a - b) for a, b in zip(start, target))
    speed = max(velocity * DOOSAN_PTP_MAX_VELOCITY_DEG_S, 1e-6)
    return max(MOTION_TIMEOUT_MINIMUM, max_delta / speed + MOTION_TIMEOUT_MARGIN)


def execute_plan(node, position_client, ik_client, move_client, move_jointx_client, plan):
    current = current_joints(node, position_client)
    for line, task, payload in plan:
        try:
            if task == "set":
                emit(line, "ok", payload=payload)
            elif task == "wait":
                time.sleep(payload)
                emit(line, "ok")
            elif task == "ptp":
                target, velocity, acceleration = payload
                if ENABLE_MOTION:
                    request = MoveJoint.Request()
                    request.pos = target
                    request.vel = velocity * DOOSAN_PTP_MAX_VELOCITY_DEG_S
                    request.acc = acceleration * DOOSAN_PTP_MAX_ACCELERATION_DEG_S2
                    request.time, request.radius, request.mode = 0.0, 0.0, 0
                    request.blend_type, request.sync_type = 0, 0
                    call_service(node, move_client, request, "the MoveJoint command", ptp_timeout(current, target, velocity))
                current = target
                emit(line, "ok")
            elif task == "move-ee-ptp":
                pose, sol, velocity, acceleration = payload
                if ENABLE_MOTION:
                    request = MoveJointx.Request()
                    request.pos = pose
                    request.sol, request.vel, request.acc = sol, velocity * DOOSAN_PTP_MAX_VELOCITY_DEG_S, acceleration * DOOSAN_PTP_MAX_ACCELERATION_DEG_S2
                    request.time, request.radius, request.ref, request.mode = 0.0, 0.0, 0, 0
                    request.blend_type, request.sync_type = 0, 0
                    call_service(node, move_jointx_client, request, "the MoveJointx command", MOTION_TIMEOUT_MINIMUM + 60.0)
                current = ik_solution(node, ik_client, pose, sol) or current
                emit(line, "ok")
        except RuntimeError as error:
            emit(line, "error", "E_DEVICE", str(error))
            return 1
    return 0


def main():
    rclpy.init()
    node = rclpy.create_node("human_instruction_executor")
    try:
        initial = initial_modal_values()
        report_initial_values(initial)
        with INSTRUCTION_FILE.open(encoding="utf-8") as file:
            plan = parse_program(json.load(file), initial)
        position_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)
        ik_client = node.create_client(Ikin, IKIN_SERVICE)
        move_client = node.create_client(MoveJoint, MOVE_JOINT_SERVICE)
        move_jointx_client = node.create_client(MoveJointx, MOVE_JOINTX_SERVICE)
        wait_for_service(position_client, POSITION_SERVICE)
        if any(task == "move-ee-ptp" for _, task, _ in plan):
            wait_for_service(ik_client, IKIN_SERVICE)
            plan = resolve_move_ee_plan(node, position_client, ik_client, plan)
            if ENABLE_MOTION:
                wait_for_service(move_jointx_client, MOVE_JOINTX_SERVICE)
        if ENABLE_MOTION and any(task == "ptp" for _, task, _ in plan):
            wait_for_service(move_client, MOVE_JOINT_SERVICE)
        return execute_plan(node,position_client,ik_client,move_client, move_jointx_client, plan,)
    except InterpreterError as error:
        emit(error.line, "error", error.code, error.message)
        return 1
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        emit(0, "error", "E_DEVICE", str(error))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
