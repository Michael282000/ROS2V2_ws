#!/usr/bin/env python3
"""Execute Human_instructions.json on a Doosan ROS 2 controller.

Supported JSON tasks:
  {"task":"set", "param":"velocity", "value":0.5}
  {"task":"ptp", "value":[0,-30,60,0,60,0], "vel":0.5}
  {"task":"move-ee", "mode":"ptp",
   "pose":{"pos":[0.40,-0.10,0.25], "quat":[0,0.7071,0.7071,0]},
   "config":{"shoulder":"SR","elbow":"EU","wrist":"WN"}, "vel":0.25}
  {"task":"wait", "value":2}

JSON poses use metres and quaternions [x, y, z, w].  Doosan MoveJointx
uses a pose in millimetres plus ZYZ Euler degrees, so this backend converts it.
The complete script is preflighted before the first motion command.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, Ikin, MoveJoint, MoveJointx

# =========================== USER SETTINGS ============================
ROBOT_MODE = "virtual"  # Test virtual first; change to "real" only when safe.
ROBOT_ID = "dsr01"
INSTRUCTION_FILE = Path("Human_instructions.json")
ENABLE_MOTION = True

# Initial modal values. PTP values are fractions of the configured PTP maxima.
BACKEND_INITIAL_VELOCITY = 0.5
BACKEND_INITIAL_ACCELERATION = 0.5
BACKEND_INITIAL_MAX_VELOCITY = 1.0
BACKEND_INITIAL_MAX_ACCELERATION = 1.0

# Verify these controller-specific values before using a real robot.
DOOSAN_PTP_MAX_VELOCITY_DEG_S = 180.0
DOOSAN_PTP_MAX_ACCELERATION_DEG_S2 = 180.0

# IMPORTANT: Fill this table only after experimentally validating the native
# Doosan solution-space number (0..7) for your robot model. Leave it empty to
# safely reject an asserted canonical config with E_UNSUPPORTED. A move-ee PTP
# without config still works: the closest available IK solution is selected.
CANONICAL_CONFIG_TO_DOOSAN_SOL = {
    # ("SL", "EU", "WN"): 0,
    # ("SL", "EU", "WF"): 1,
    # ("SL", "ED", "WN"): 2,
    # ("SL", "ED", "WF"): 3,
    # ("SR", "EU", "WN"): 4,
    # ("SR", "EU", "WF"): 5,
    # ("SR", "ED", "WN"): 6,
    # ("SR", "ED", "WF"): 7,
}

SERVICE_TIMEOUT = 10.0
POSITION_RESPONSE_TIMEOUT = 10.0
IK_RESPONSE_TIMEOUT = 10.0
MOTION_TIMEOUT_MARGIN = 30.0
MOTION_TIMEOUT_MINIMUM = 30.0
# =======================================================================

CONTROLLER_NAMESPACE = f"/{ROBOT_ID}/dsr_controller2"
POSITION_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/get_current_posj"
IKIN_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/ikin"
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

    # Rotation matrix, then the ZYZ convention used by the standard Doosan posx API.
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
    elif r22 > 0:  # beta = 0: use alpha + gamma as one Z rotation.
        alpha, gamma = math.atan2(r10, r00), 0.0
    else:          # beta = pi: choose gamma = 0 deterministically.
        alpha, gamma = math.atan2(-r01, -r00), 0.0
    return [math.degrees(angle) for angle in (alpha, beta, gamma)]


def parse_pose(value, line):
    if not isinstance(value, dict) or set(value) != {"pos", "quat"}:
        raise InterpreterError(line, "E_PARSE", "pose needs pos and quat fields only.")
    if not isinstance(value["pos"], list) or len(value["pos"]) != 3:
        raise InterpreterError(line, "E_PARSE", "pose.pos must be [x, y, z] in metres.")
    try:
        position_m = [float(number) for number in value["pos"]]
    except (TypeError, ValueError):
        raise InterpreterError(line, "E_PARSE", "pose.pos values must be numeric.")
    if not all(math.isfinite(number) for number in position_m):
        raise InterpreterError(line, "E_PARSE", "pose.pos values must be finite.")
    return [1000.0 * number for number in position_m] + quaternion_to_zyz_degrees(value["quat"], line)


def preflight(instructions, modal):
    plan = []
    for line, instruction in enumerate(instructions, start=1):
        if not isinstance(instruction, dict):
            raise InterpreterError(line, "E_PARSE", "Line must be a JSON object.")
        task = instruction.get("task")
        if task == "set":
            param = instruction.get("param")
            if param not in {"velocity", "acceleration", "max-velocity", "max-acceleration"} or "value" not in instruction:
                raise InterpreterError(line, "E_PARSE", "Invalid set instruction.")
            value = finite_positive(instruction["value"], line, "value")
            candidate = ModalValues(**vars(modal))
            setattr(candidate, param.replace("-", "_"), value)
            if candidate.velocity > candidate.max_velocity or candidate.acceleration > candidate.max_acceleration:
                raise InterpreterError(line, "E_LIMIT", "Modal velocity or acceleration exceeds its cap.")
            modal = candidate
            plan.append((line, "set", {"param": param, "value": value}))
        elif task in {"ptp", "move_joint"}:
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            validate_ptp(line, velocity, acceleration, modal)
            plan.append((line, "ptp", {"joints": parse_joints(instruction.get("value"), line), "velocity": velocity, "acceleration": acceleration}))
        elif task == "move-ee":
            if instruction.get("mode") != "ptp":
                raise InterpreterError(line, "E_UNSUPPORTED", "LIN is rejected: this backend cannot faithfully assert IK branch continuity.")
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            validate_ptp(line, velocity, acceleration, modal)
            config, sol = parse_config(instruction["config"], line) if "config" in instruction else (None, None)
            plan.append((line, "move-ee-ptp", {"pose": parse_pose(instruction.get("pose"), line), "config": config, "sol": sol, "velocity": velocity, "acceleration": acceleration}))
        elif task == "wait":
            plan.append((line, "wait", {"seconds": finite_positive(instruction.get("value"), line, "value")}))
        else:
            raise InterpreterError(line, "E_PARSE", "Unsupported or missing task.")
    return plan


def wait_for_service(client, name):
    print(f"Waiting for service: {name}")
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
        raise RuntimeError(f"Service not found: {name}.")


def call_service(node, client, request, description, timeout):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done():
        raise RuntimeError(f"Timed out after {timeout:.1f} s while waiting for {description}.")
    response = future.result()
    if response is None or not response.success:
        raise RuntimeError(f"The controller rejected or failed {description}.")
    return response


def get_joint_positions(node, client):
    return [float(value) for value in call_service(node, client, GetCurrentPosj.Request(), "joint positions", POSITION_RESPONSE_TIMEOUT).pos]


def ik_solution(node, ik_client, pose, sol):
    request = Ikin.Request()
    request.pos, request.sol_space, request.ref = pose, sol, 0  # DR_BASE
    future = ik_client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=IK_RESPONSE_TIMEOUT)
    response = future.result() if future.done() else None
    return None if response is None or not response.success else [float(value) for value in response.conv_posj]


def resolve_move_ee_plan(node, position_client, ik_client, plan):
    """Resolve/check every move-ee target before the first move command."""
    current = get_joint_positions(node, position_client)
    for line, task, item in plan:
        if task == "ptp":
            current = item["joints"]
            continue
        if task != "move-ee-ptp":
            continue
        if item["sol"] is not None:
            joints = ik_solution(node, ik_client, item["pose"], item["sol"])
            if joints is None:
                raise InterpreterError(line, "E_UNREACHABLE_CONFIG", "Requested IK branch has no solution at the target.")
        else:
            candidates = [(sol, ik_solution(node, ik_client, item["pose"], sol)) for sol in range(8)]
            candidates = [(sol, joints) for sol, joints in candidates if joints is not None]
            if not candidates:
                raise InterpreterError(line, "E_UNREACHABLE", "Target pose has no IK solution.")
            item["sol"], joints = min(candidates, key=lambda entry: sum((a-b)**2 for a, b in zip(current, entry[1])))
        item["predicted_joints"] = joints
        current = joints


def ptp_timeout(current, target, fraction):
    speed = fraction * DOOSAN_PTP_MAX_VELOCITY_DEG_S
    distance = max(abs(a-b) for a, b in zip(current, target))
    return max(MOTION_TIMEOUT_MINIMUM, distance / speed + MOTION_TIMEOUT_MARGIN)


def execute_ptp(node, position_client, move_client, item):
    current = get_joint_positions(node, position_client)
    vel = item["velocity"] * DOOSAN_PTP_MAX_VELOCITY_DEG_S
    acc = item["acceleration"] * DOOSAN_PTP_MAX_ACCELERATION_DEG_S2
    request = MoveJoint.Request()
    request.pos, request.vel, request.acc = item["joints"], vel, acc
    request.time, request.radius, request.mode, request.blend_type, request.sync_type = 0.0, 0.0, 0, 0, 0
    call_service(node, move_client, request, "the MoveJoint command", ptp_timeout(current, item["joints"], item["velocity"]))
    return {"joints": get_joint_positions(node, position_client), "velocity_fraction": item["velocity"], "acceleration_fraction": item["acceleration"]}


def execute_move_ee_ptp(node, position_client, move_jointx_client, item):
    current = get_joint_positions(node, position_client)
    vel = item["velocity"] * DOOSAN_PTP_MAX_VELOCITY_DEG_S
    acc = item["acceleration"] * DOOSAN_PTP_MAX_ACCELERATION_DEG_S2
    request = MoveJointx.Request()
    request.pos, request.vel, request.acc = item["pose"], vel, acc
    request.time, request.radius, request.ref, request.mode = 0.0, 0.0, 0, 0
    request.blend_type, request.sol, request.sync_type = 0, item["sol"], 0
    call_service(node, move_jointx_client, request, "the MoveJointx command", ptp_timeout(current, item["predicted_joints"], item["velocity"]))
    return {"pose_mm_zyz_deg": item["pose"], "sol": item["sol"], "joints": get_joint_positions(node, position_client), "velocity_fraction": item["velocity"], "acceleration_fraction": item["acceleration"]}


def execute_plan(node, position_client, move_client, move_jointx_client, plan):
    for line, task, item in plan:
        try:
            if task == "set":
                payload = item
            elif task == "wait":
                time.sleep(item["seconds"])
                payload = None
            elif not ENABLE_MOTION:
                payload = {"motion_disabled": True, **item}
            elif task == "ptp":
                payload = execute_ptp(node, position_client, move_client, item)
            else:
                payload = execute_move_ee_ptp(node, position_client, move_jointx_client, item)
            emit(line, "ok", None, payload)
        except RuntimeError as error:
            emit(line, "error", "E_DEVICE", str(error))
            return 1
    return 0


def load_instructions(filename):
    try:
        with filename.open("r", encoding="utf-8") as file:
            result = json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Instruction file not found: {filename.resolve()}")
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON: {error}")
    if not isinstance(result, list):
        raise RuntimeError("JSON root must be a list.")
    return result


def main():
    modal = initial_modal_values()
    report_initial_values(modal)
    try:
        plan = preflight(load_instructions(INSTRUCTION_FILE), modal)
    except InterpreterError as error:
        emit(error.line, "error", error.code, error.message)
        return 1
    except RuntimeError as error:
        emit(0, "error", "E_PARSE", str(error))
        return 1
    rclpy.init()
    node = rclpy.create_node("execute_human_instructions")
    try:
        position_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)
        ik_client = node.create_client(Ikin, IKIN_SERVICE)
        move_client = node.create_client(MoveJoint, MOVE_JOINT_SERVICE)
        move_jointx_client = node.create_client(MoveJointx, MOVE_JOINTX_SERVICE)
        wait_for_service(position_client, POSITION_SERVICE)
        if any(task == "move-ee-ptp" for _, task, _ in plan):
            wait_for_service(ik_client, IKIN_SERVICE)
            resolve_move_ee_plan(node, position_client, ik_client, plan)
            if ENABLE_MOTION:
                wait_for_service(move_jointx_client, MOVE_JOINTX_SERVICE)
        if ENABLE_MOTION and any(task == "ptp" for _, task, _ in plan):
            wait_for_service(move_client, MOVE_JOINT_SERVICE)
        return execute_plan(node, position_client, move_client, move_jointx_client, plan)
    except InterpreterError as error:
        emit(error.line, "error", error.code, error.message)
        return 1
    except RuntimeError as error:
        emit(0, "error", "E_DEVICE", str(error))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
