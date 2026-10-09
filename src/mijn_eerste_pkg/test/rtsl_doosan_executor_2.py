#!/usr/bin/env python3
"""RTSL v0.1 interpreter for a Doosan arm on ROS 2 (dsr_controller2).

Safety/semantics in this revision:
- RTSL Cartesian poses are evaluated in DR_WORLD (reference 2).
- move-ee mode=ptp solves and validates IK, then executes that exact joint
  solution with MoveJoint rather than asking the controller to solve IK again.
- move-ee mode=lin performs adaptive SLERP/linear-position branch preflight
  immediately before motion, from the robot's measured current joint state.
- Explicit LIN config is asserted at the measured start state and verified at
  the measured end state. Joint comparisons are 360-degree robust.
- LIN is refused unless the installed Doosan MoveLine implementation has been
  explicitly verified to use shortest-path SO(3)/SLERP-equivalent orientation
  interpolation. Set DOOSAN_MOVELINE_ORIENTATION_IS_SLERP only after verifying
  this against the controller documentation and the physical robot.

Supported JSON tasks:
  {"task":"set", "param":"velocity", "value":0.5}
  {"task":"ptp"|"move_joint", "value":[0,-30,60,0,60,0], "vel":0.5}
  {"task":"move-ee", "mode":"ptp",
   "pose":{"pos":[0.40,-0.10,0.25], "quat":[0.0,0.7071,0.7071,0.0]},
   "config":{"shoulder":"SR","elbow":"EU","wrist":"WN"}, "vel":0.25}
  {"task":"move-ee", "mode":"lin",
   "pose":{"pos":[0.40,-0.10,0.25], "quat":[0.0,0.7071,0.7071,0.0]},
   "config":{"shoulder":"SR","elbow":"EU","wrist":"WN"}, "vel":0.05}
  {"task":"wait", "value":2}

RTSL: metres, radians/seconds/SI in source language; quaternion order w,x,y,z.
Doosan posx: mm, mm, mm, ZYZ Euler degrees.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, Ikin, MoveJoint, MoveLine

try:
    from dsr_msgs2.srv import Fkin
    HAVE_FKIN = True
except ImportError:
    Fkin = None
    HAVE_FKIN = False

# =========================== USER SETTINGS ============================
ROBOT_ID = "dsr01"
INSTRUCTION_FILE = Path("/home/bbrug1/ROS2V2_ws/src/mijn_eerste_pkg/test/Human.json")

ENABLE_MOTION = True

ROBOT_MODEL = "a0509"
CONTROLLER_VERSION = "DRCF GF02100100 / DRFL GL013304"

# RTSL specifies all Cartesian poses in the fixed world frame.  Doosan's
# standard reference constants are DR_BASE=0, DR_TOOL=1, DR_WORLD=2.
# Do not change this to DR_BASE unless RTSL's world frame is deliberately
# configured to coincide exactly with the robot base frame.
RTSL_REFERENCE = 2  # DR_WORLD

# Do not enable this until MoveLine's orientation interpolation has been
# confirmed for this exact controller/firmware and tested on the physical cell.
# If it is false, LIN is rejected rather than falsely claiming RTSL SLERP.
DOOSAN_MOVELINE_ORIENTATION_IS_SLERP = False

BACKEND_INITIAL_VELOCITY = 0.01
BACKEND_INITIAL_ACCELERATION = 0.01
BACKEND_INITIAL_MAX_VELOCITY = 1.0
BACKEND_INITIAL_MAX_ACCELERATION = 1.0

# PTP backend maxima: fraction 1.0 maps to these values. Verify on controller.
DOOSAN_PTP_MAX_VELOCITY_DEG_S = 150.0
DOOSAN_PTP_MAX_ACCELERATION_DEG_S2 = 150.0

# LIN caps in RTSL SI units. Verify against the cell safety envelope.
BACKEND_LIN_MAX_VELOCITY_M_S = 0.25
BACKEND_LIN_MAX_ACCELERATION_M_S2 = 1.0

# Rotational companions for Doosan MoveLine [mm/s, deg/s] and
# [mm/s^2, deg/s^2]. RTSL v0.1 has no separate rotational fields.
LIN_DEFAULT_ROT_VEL_DEG_S = 30.0
LIN_DEFAULT_ROT_ACC_DEG_S2 = 60.0

# Adaptive LIN preflight: every segment is no larger than either limit, with
# at least LIN_MIN_SEGMENTS. More samples cost more IK/FK service calls.
LIN_MIN_SEGMENTS = 20
LIN_MAX_SAMPLE_TRANSLATION_MM = 5.0
LIN_MAX_SAMPLE_ROTATION_DEG = 2.0
LIN_MAX_JOINT_JUMP_DEG = 15.0
LIN_BRANCH_MATCH_TOL_DEG = 2.0
LIN_MIN_POS_FOR_TRANSLATION_MM = 0.1

# Execution-state verification tolerances.
PTP_END_MATCH_TOL_DEG = 2.0

JOINT_LIMITS_DEG = [
    (-360.0, 360.0),
    (-360.0, 360.0),
    (-160.0, 160.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
]

CANONICAL_CONFIG_TO_DOOSAN_SOL = {
    ("SL", "ED", "WN"): 0, ("SL", "ED", "WF"): 1,
    ("SL", "EU", "WN"): 2, ("SL", "EU", "WF"): 3,
    ("SR", "ED", "WN"): 4, ("SR", "ED", "WF"): 5,
    ("SR", "EU", "WN"): 6, ("SR", "EU", "WF"): 7,
}

FKIN_POSITION_TOL_MM = 0.5
FKIN_ORIENTATION_TOL_DEG = 0.5
LARGE_MOTION_WARN_DEG = 180.0

SERVICE_TIMEOUT = 10.0
POSITION_RESPONSE_TIMEOUT = 10.0
IK_RESPONSE_TIMEOUT = 10.0
MOTION_TIMEOUT_MARGIN = 30.0
MOTION_TIMEOUT_MINIMUM = 30.0
# =======================================================================

CONTROLLER_NAMESPACE = f"/{ROBOT_ID}/dsr_controller2"
POSITION_SERVICE = f"{CONTROLLER_NAMESPACE}/aux_control/get_current_posj"
IKIN_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/ikin"
FKIN_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/fkin"
MOVE_JOINT_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/move_joint"
MOVE_LINE_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/move_line"


@dataclass
class ModalValues:
    velocity: float
    acceleration: float
    max_velocity: float
    max_acceleration: float


class InterpreterError(Exception):
    def __init__(self, line, code, message):
        super().__init__(message)
        self.line = line
        self.code = code
        self.message = message


def emit(line, status, code=None, payload=None):
    print(json.dumps({"line": line, "status": status,
                      "code": code, "payload": payload}), flush=True)


def note(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


# --------------------------------------------------------------------- #
# Scalar parsing and modal values
# --------------------------------------------------------------------- #
def _as_float(value, line, field):
    if isinstance(value, bool):
        raise InterpreterError(line, "E_PARSE", f"{field} must be a number.")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InterpreterError(line, "E_PARSE", f"{field} must be a number.")
    if not math.isfinite(value):
        raise InterpreterError(line, "E_PARSE", f"{field} must be finite.")
    return value


def finite_positive(value, line, field):
    value = _as_float(value, line, field)
    if value <= 0:
        raise InterpreterError(line, "E_PARSE", f"{field} must be greater than zero.")
    return value


def finite_nonnegative(value, line, field):
    value = _as_float(value, line, field)
    if value < 0:
        raise InterpreterError(line, "E_PARSE", f"{field} must not be negative.")
    return value


def initial_modal_values():
    values = ModalValues(
        finite_positive(BACKEND_INITIAL_VELOCITY, 0, "BACKEND_INITIAL_VELOCITY"),
        finite_positive(BACKEND_INITIAL_ACCELERATION, 0, "BACKEND_INITIAL_ACCELERATION"),
        finite_positive(BACKEND_INITIAL_MAX_VELOCITY, 0, "BACKEND_INITIAL_MAX_VELOCITY"),
        finite_positive(BACKEND_INITIAL_MAX_ACCELERATION, 0, "BACKEND_INITIAL_MAX_ACCELERATION"),
    )
    for name, value in (
        ("DOOSAN_PTP_MAX_VELOCITY_DEG_S", DOOSAN_PTP_MAX_VELOCITY_DEG_S),
        ("DOOSAN_PTP_MAX_ACCELERATION_DEG_S2", DOOSAN_PTP_MAX_ACCELERATION_DEG_S2),
        ("BACKEND_LIN_MAX_VELOCITY_M_S", BACKEND_LIN_MAX_VELOCITY_M_S),
        ("BACKEND_LIN_MAX_ACCELERATION_M_S2", BACKEND_LIN_MAX_ACCELERATION_M_S2),
        ("LIN_DEFAULT_ROT_VEL_DEG_S", LIN_DEFAULT_ROT_VEL_DEG_S),
        ("LIN_DEFAULT_ROT_ACC_DEG_S2", LIN_DEFAULT_ROT_ACC_DEG_S2),
    ):
        finite_positive(value, 0, name)
    if values.velocity > values.max_velocity or values.acceleration > values.max_acceleration:
        raise ValueError("Initial velocity/acceleration cannot exceed configured caps.")
    return values


def report_initial_values(values):
    note("initial_modal_values",
         robot_model=ROBOT_MODEL,
         controller_version=CONTROLLER_VERSION,
         rtsl_reference=RTSL_REFERENCE,
         velocity=values.velocity,
         acceleration=values.acceleration,
         max_velocity=values.max_velocity,
         max_acceleration=values.max_acceleration,
         ptp_backend_max_velocity_deg_s=DOOSAN_PTP_MAX_VELOCITY_DEG_S,
         ptp_backend_max_acceleration_deg_s2=DOOSAN_PTP_MAX_ACCELERATION_DEG_S2,
         lin_backend_max_velocity_m_s=BACKEND_LIN_MAX_VELOCITY_M_S,
         lin_backend_max_acceleration_m_s2=BACKEND_LIN_MAX_ACCELERATION_M_S2,
         lin_default_rot_vel_deg_s=LIN_DEFAULT_ROT_VEL_DEG_S,
         lin_default_rot_acc_deg_s2=LIN_DEFAULT_ROT_ACC_DEG_S2,
         moveline_orientation_slerp_verified=DOOSAN_MOVELINE_ORIENTATION_IS_SLERP,
         quaternion_order="wxyz",
         fkin_validation=HAVE_FKIN,
         motion_enabled=ENABLE_MOTION)
    if not HAVE_FKIN:
        note("warning", detail="Fkin unavailable: IK solutions are checked against "
                               "joint limits only; LIN cannot execute.")
    if not DOOSAN_MOVELINE_ORIENTATION_IS_SLERP:
        note("warning", detail="LIN is disabled until Doosan MoveLine orientation "
                               "interpolation is verified as SLERP-equivalent.")


def motion_value(instruction, line, short_name, long_name, default):
    present = [name for name in (short_name, long_name) if name in instruction]
    if len(present) > 1:
        raise InterpreterError(line, "E_PARSE",
                               f"Use only one of {short_name} and {long_name}.")
    return finite_positive(instruction[present[0]], line, present[0]) if present else default


def validate_ptp(line, velocity, acceleration, modal):
    if velocity > 1.0 or velocity > modal.max_velocity:
        raise InterpreterError(line, "E_LIMIT",
                               "PTP velocity exceeds its cap or portable range 0-1.")
    if acceleration > 1.0 or acceleration > modal.max_acceleration:
        raise InterpreterError(line, "E_LIMIT",
                               "PTP acceleration exceeds its cap or portable range 0-1.")


def validate_lin(line, velocity, acceleration, modal):
    if velocity > modal.max_velocity:
        raise InterpreterError(line, "E_LIMIT", "LIN velocity exceeds modal max-velocity.")
    if acceleration > modal.max_acceleration:
        raise InterpreterError(line, "E_LIMIT", "LIN acceleration exceeds modal max-acceleration.")
    if velocity > BACKEND_LIN_MAX_VELOCITY_M_S:
        raise InterpreterError(line, "E_LIMIT",
                               f"LIN velocity {velocity} m/s exceeds back-end cap "
                               f"{BACKEND_LIN_MAX_VELOCITY_M_S} m/s.")
    if acceleration > BACKEND_LIN_MAX_ACCELERATION_M_S2:
        raise InterpreterError(line, "E_LIMIT",
                               f"LIN acceleration {acceleration} m/s^2 exceeds back-end cap "
                               f"{BACKEND_LIN_MAX_ACCELERATION_M_S2} m/s^2.")


def warn_modal_ambiguity(line, param, value):
    if param not in {"velocity", "acceleration", "max-velocity", "max-acceleration"}:
        return
    kind = "velocity" if "velocity" in param else "acceleration"
    lin_unit = "m/s" if kind == "velocity" else "m/s^2"
    note("warning_modal_ambiguity", line=line, param=param, value=value,
         detail=f"RTSL v0.1 has one modal '{kind}' shared by ptp (fraction 0..1) "
                f"and lin ({lin_unit}). Prefer explicit per-line vel/acc overrides.")


# --------------------------------------------------------------------- #
# Joint and configuration helpers
# --------------------------------------------------------------------- #
def joint_delta_deg(a, b):
    """Shortest equivalent angular difference a-b, in [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def angular_distance(a, b):
    return sum(joint_delta_deg(x, y) ** 2 for x, y in zip(a, b))


def max_joint_jump_deg(a, b):
    return max(abs(joint_delta_deg(x, y)) for x, y in zip(a, b))


def joint_limit_violations(joints):
    out = []
    for index, (angle, (low, high)) in enumerate(zip(joints, JOINT_LIMITS_DEG), start=1):
        if not low <= angle <= high:
            out.append({"joint": index, "value": round(angle, 4), "limit": [low, high]})
    return out


def parse_joints(value, line):
    if not isinstance(value, list) or len(value) != 6:
        raise InterpreterError(line, "E_PARSE",
                               "PTP value must be a six-element joint-angle list.")
    joints = [_as_float(angle, line, f"joint {index + 1}")
              for index, angle in enumerate(value)]
    violations = joint_limit_violations(joints)
    if violations:
        raise InterpreterError(line, "E_LIMIT",
                               f"Joint angles outside configured limits: {violations}")
    return joints


def parse_config(value, line):
    if not isinstance(value, dict) or set(value) != {"shoulder", "elbow", "wrist"}:
        raise InterpreterError(line, "E_PARSE",
                               "config needs shoulder, elbow, and wrist fields.")
    config = (value["shoulder"], value["elbow"], value["wrist"])
    if (config[0] not in {"SL", "SR"} or config[1] not in {"EU", "ED"}
            or config[2] not in {"WN", "WF"}):
        raise InterpreterError(line, "E_PARSE", "Invalid canonical IK configuration.")
    sol = CANONICAL_CONFIG_TO_DOOSAN_SOL.get(config)
    if sol is None:
        raise InterpreterError(line, "E_UNSUPPORTED",
                               "No validated Doosan sol mapping for this config.")
    return config, sol


# --------------------------------------------------------------------- #
# Orientation and pose conversion
# --------------------------------------------------------------------- #
def quaternion_to_matrix(w, x, y, z):
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)),
        (2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)),
    )


def matrix_to_zyz_degrees(r):
    beta = math.acos(max(-1.0, min(1.0, r[2][2])))
    if abs(math.sin(beta)) > 1e-9:
        alpha = math.atan2(r[1][2], r[0][2])
        gamma = math.atan2(r[2][1], -r[2][0])
    elif r[2][2] > 0:
        alpha, gamma = math.atan2(r[1][0], r[0][0]), 0.0
    else:
        alpha, gamma = math.atan2(-r[0][1], -r[0][0]), 0.0
    return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]


def zyz_degrees_to_matrix(a_deg, b_deg, c_deg):
    a, b, c = (math.radians(v) for v in (a_deg, b_deg, c_deg))
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b),
                              math.sin(b), math.cos(c), math.sin(c))
    return (
        (ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb),
        (sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb),
        (-sb * cc,                sb * sc,                cb),
    )


def rotation_angle_deg(ra, rb):
    trace = sum(ra[i][0] * rb[i][0] + ra[i][1] * rb[i][1] + ra[i][2] * rb[i][2]
                for i in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def quaternion_normalise(quat, line):
    if not isinstance(quat, list) or len(quat) != 4:
        raise InterpreterError(line, "E_PARSE", "quat must be [w, x, y, z].")
    w, x, y, z = (_as_float(v, line, "quat") for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-3:
        raise InterpreterError(line, "E_PARSE",
                               "Quaternion norm deviates from 1 by more than 1e-3.")
    return [w / norm, x / norm, y / norm, z / norm]


def matrix_to_quaternion(r):
    """Return normalised (w, x, y, z), using Shepperd's stable method."""
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x, y, z = (0.25 * s, (r[2][1] - r[1][2]) / s,
                      (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w, x, y, z = ((r[2][1] - r[1][2]) / s, 0.25 * s,
                      (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s)
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w, x, y, z = ((r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s,
                      0.25 * s, (r[1][2] + r[2][1]) / s)
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w, x, y, z = ((r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s,
                      (r[1][2] + r[2][1]) / s, 0.25 * s)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / norm, x / norm, y / norm, z / norm]


def quaternion_slerp(q0, q1, t):
    """Shortest-arc SLERP between unit wxyz quaternions."""
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = [-v for v in q1]
        dot = -dot
    if dot > 0.9995:
        out = [a + t * (b - a) for a, b in zip(q0, q1)]
        norm = math.sqrt(sum(v * v for v in out))
        return [v / norm for v in out]
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / math.sin(theta_0)
    s1 = math.sin(theta) / math.sin(theta_0)
    return [s0 * a + s1 * b for a, b in zip(q0, q1)]


def parse_pose(value, line):
    """RTSL pose -> (position_mm, unit quaternion wxyz)."""
    if not isinstance(value, dict) or set(value) != {"pos", "quat"}:
        raise InterpreterError(line, "E_PARSE", "pose needs pos and quat fields.")
    if not isinstance(value["pos"], list) or len(value["pos"]) != 3:
        raise InterpreterError(line, "E_PARSE", "pose.pos must be [x, y, z] in metres.")
    position_mm = [_as_float(c, line, "pose.pos") * 1000.0 for c in value["pos"]]
    return position_mm, quaternion_normalise(value["quat"], line)


def posx_from_pose(position_mm, quat_wxyz):
    return list(position_mm) + matrix_to_zyz_degrees(quaternion_to_matrix(*quat_wxyz))


def self_test():
    r = quaternion_to_matrix(0.0, 1.0, 0.0, 0.0)
    assert abs(r[0][0] - 1.0) < 1e-12 and abs(r[2][2] + 1.0) < 1e-12, r
    q = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]
    zyz = matrix_to_zyz_degrees(quaternion_to_matrix(*q))
    assert rotation_angle_deg(quaternion_to_matrix(*q), zyz_degrees_to_matrix(*zyz)) < 1e-6
    qh = quaternion_slerp([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 0.5)
    expected = [math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0, 0.0]
    assert max(abs(a - b) for a, b in zip(qh, expected)) < 1e-9, qh
    assert abs(joint_delta_deg(-179.0, 181.0)) < 1e-9
    assert sorted(CANONICAL_CONFIG_TO_DOOSAN_SOL.values()) == list(range(8))


# --------------------------------------------------------------------- #
# Program parsing: no hardware calls and no IK resolution occur here.
# --------------------------------------------------------------------- #
def parse_program(instructions, initial):
    if not isinstance(instructions, list):
        raise InterpreterError(0, "E_PARSE", "JSON root must be a list of instructions.")
    modal = ModalValues(**vars(initial))
    plan = []
    for line, instruction in enumerate(instructions, start=1):
        if not isinstance(instruction, dict) or not isinstance(instruction.get("task"), str):
            raise InterpreterError(line, "E_PARSE",
                                   "Each instruction needs an object with string task.")
        task = instruction["task"]
        if task == "set":
            if set(instruction) != {"task", "param", "value"} or instruction["param"] not in {
                "velocity", "acceleration", "max-velocity", "max-acceleration"
            }:
                raise InterpreterError(line, "E_PARSE", "Invalid set instruction.")
            param = instruction["param"]
            value = finite_positive(instruction["value"], line, "value")
            warn_modal_ambiguity(line, param, value)
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
                    raise InterpreterError(line, "E_LIMIT",
                                           "max-velocity cannot be below active velocity.")
                modal.max_velocity = value
            else:
                if value < modal.acceleration:
                    raise InterpreterError(line, "E_LIMIT",
                                           "max-acceleration cannot be below active acceleration.")
                modal.max_acceleration = value
            plan.append((line, "set", {"param": param, "value": value}))
        elif task in {"ptp", "move_joint"}:
            allowed = {"task", "value", "vel", "velocity", "acc", "acceleration"}
            if not set(instruction) <= allowed or "value" not in instruction:
                raise InterpreterError(line, "E_PARSE", "Invalid PTP instruction.")
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            validate_ptp(line, velocity, acceleration, modal)
            plan.append((line, "ptp", (parse_joints(instruction["value"], line),
                                       velocity, acceleration)))
        elif task == "move-ee":
            allowed = {"task", "mode", "pose", "config", "vel", "velocity", "acc", "acceleration"}
            if (not set(instruction) <= allowed or instruction.get("mode") not in {"ptp", "lin"}
                    or "pose" not in instruction):
                raise InterpreterError(line, "E_PARSE", "Invalid move-ee instruction.")
            mode = instruction["mode"]
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            requested_sol = parse_config(instruction["config"], line)[1] if "config" in instruction else None
            position_mm, quat_wxyz = parse_pose(instruction["pose"], line)
            if mode == "ptp":
                validate_ptp(line, velocity, acceleration, modal)
                plan.append((line, "move-ee-ptp",
                             (position_mm, quat_wxyz, requested_sol, velocity, acceleration)))
            else:
                validate_lin(line, velocity, acceleration, modal)
                plan.append((line, "move-ee-lin",
                             (position_mm, quat_wxyz, requested_sol, velocity, acceleration)))
        elif task == "wait":
            if set(instruction) != {"task", "value"}:
                raise InterpreterError(line, "E_PARSE", "Invalid wait instruction.")
            plan.append((line, "wait", finite_nonnegative(instruction["value"], line, "value")))
        elif task in {"take-image", "gripper"}:
            raise InterpreterError(line, "E_UNSUPPORTED",
                                   f"This bare robot-controller back-end has no {task} support.")
        else:
            raise InterpreterError(line, "E_UNSUPPORTED", f"Unsupported task: {task}.")
    return plan


# --------------------------------------------------------------------- #
# ROS service plumbing
# --------------------------------------------------------------------- #
def wait_for_service(client, service_name, required=True):
    print(f"Waiting for service: {service_name}", flush=True)
    if client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
        return True
    if required:
        raise RuntimeError(f"Service not found: {service_name}.")
    note("warning", detail=f"optional service unavailable: {service_name}")
    return False


def call_service(node, client, request, description, timeout):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done():
        future.cancel()
        raise RuntimeError(f"Timed out while waiting for {description}.")
    response = future.result()
    if response is None or not getattr(response, "success", False):
        raise RuntimeError(f"Controller rejected or failed {description}.")
    return response


def try_call(node, client, request, timeout):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done():
        future.cancel()
        return None
    response = future.result()
    if response is None or not getattr(response, "success", False):
        return None
    return response


def current_joints(node, position_client):
    response = call_service(node, position_client, GetCurrentPosj.Request(),
                            "the current joint-position response", POSITION_RESPONSE_TIMEOUT)
    joints = [float(angle) for angle in response.pos]
    if len(joints) != 6:
        raise RuntimeError(f"Controller returned {len(joints)} joints; expected six.")
    return joints


def raw_ik(node, ik_client, pose, sol):
    request = Ikin.Request()
    request.pos = [float(v) for v in pose]
    request.sol_space = int(sol)
    request.ref = RTSL_REFERENCE
    response = try_call(node, ik_client, request, IK_RESPONSE_TIMEOUT)
    if response is None:
        return None
    joints = [float(angle) for angle in response.conv_posj]
    return joints if len(joints) == 6 else None


def raw_fk(node, fk_client, joints):
    request = Fkin.Request()
    request.pos = [float(angle) for angle in joints]
    request.ref = RTSL_REFERENCE
    response = try_call(node, fk_client, request, IK_RESPONSE_TIMEOUT)
    if response is None:
        return None
    pose = [float(v) for v in response.conv_posx]
    return pose if len(pose) == 6 else None


def ik_solution(node, ik_client, fk_client, pose, sol):
    joints = raw_ik(node, ik_client, pose, sol)
    if joints is None:
        return None, "ikin reported failure"
    violations = joint_limit_violations(joints)
    if violations:
        return None, f"joint limits violated: {violations}"
    posx = raw_fk(node, fk_client, joints)
    if posx is None:
        return None, "fkin round-trip failed"
    dp = math.sqrt(sum((a - b) ** 2 for a, b in zip(posx[:3], pose[:3])))
    drot = rotation_angle_deg(zyz_degrees_to_matrix(*posx[3:]),
                              zyz_degrees_to_matrix(*pose[3:]))
    if dp > FKIN_POSITION_TOL_MM or drot > FKIN_ORIENTATION_TOL_DEG:
        return None, f"fkin round-trip off by {dp:.3f} mm / {drot:.3f} deg"
    return joints, None


# --------------------------------------------------------------------- #
# IK resolution and LIN branch preflight
# --------------------------------------------------------------------- #
def select_ptp_solution(node, ik_client, fk_client, current, target_posx, requested_sol, line):
    candidates, rejected = {}, {}
    for sol in range(8):
        joints, reason = ik_solution(node, ik_client, fk_client, target_posx, sol)
        if joints is None:
            rejected[sol] = reason
        else:
            candidates[sol] = joints

    note("ik_candidates", line=line,
         doosan_pose_mm_zyz_deg=[round(v, 4) for v in target_posx],
         requested_sol=requested_sol,
         valid_solutions=sorted(candidates), rejected=rejected)
    if not candidates:
        raise InterpreterError(line, "E_UNREACHABLE", "Target pose has no valid IK solution.")
    if requested_sol is None:
        sol = min(candidates, key=lambda candidate: angular_distance(current, candidates[candidate]))
    elif requested_sol in candidates:
        sol = requested_sol
    else:
        raise InterpreterError(line, "E_UNREACHABLE_CONFIG",
                               f"Requested IK branch sol={requested_sol} is invalid at target "
                               f"({rejected.get(requested_sol, 'unknown reason')}).")
    return sol, candidates[sol]


def interpolate_posx(start_pos_mm, start_quat, end_pos_mm, end_quat, t):
    pos = [a + t * (b - a) for a, b in zip(start_pos_mm, end_pos_mm)]
    quat = quaternion_slerp(start_quat, end_quat, t)
    return posx_from_pose(pos, quat)


def determine_start_sol(node, ik_client, fk_client, start_joints, start_posx):
    """Find the validated native sol that matches the measured current joints."""
    best_sol, best_joints, best_error = None, None, float("inf")
    for sol in range(8):
        joints, reason = ik_solution(node, ik_client, fk_client, start_posx, sol)
        if joints is None:
            continue
        error = angular_distance(joints, start_joints)
        if error < best_error:
            best_sol, best_joints, best_error = sol, joints, error
    if best_sol is None:
        return None, "no valid sol reproduces current pose"
    mismatch = max_joint_jump_deg(best_joints, start_joints)
    if mismatch > LIN_BRANCH_MATCH_TOL_DEG:
        return None, (f"closest sol={best_sol} differs from measured joints by "
                      f"{mismatch:.2f} deg")
    return best_sol, None


def assert_start_branch(node, ik_client, fk_client, start_joints, start_posx, sol, line):
    """Ensure an explicitly requested LIN branch is the measured start branch."""
    branch_joints, reason = ik_solution(node, ik_client, fk_client, start_posx, sol)
    if branch_joints is None:
        raise InterpreterError(line, "E_CONFIG_CHANGE",
                               f"LIN start pose is not valid on asserted sol={sol}: {reason}.")
    mismatch = max_joint_jump_deg(branch_joints, start_joints)
    if mismatch > LIN_BRANCH_MATCH_TOL_DEG:
        raise InterpreterError(line, "E_CONFIG_CHANGE",
                               f"LIN starts on a different branch than asserted sol={sol}; "
                               f"joint difference is {mismatch:.2f} deg.")


def lin_segment_count(start_pos_mm, start_quat, end_pos_mm, end_quat):
    distance_mm = math.sqrt(sum((a - b) ** 2 for a, b in zip(start_pos_mm, end_pos_mm)))
    rotation_deg = rotation_angle_deg(quaternion_to_matrix(*start_quat),
                                      quaternion_to_matrix(*end_quat))
    by_translation = math.ceil(distance_mm / LIN_MAX_SAMPLE_TRANSLATION_MM)
    by_rotation = math.ceil(rotation_deg / LIN_MAX_SAMPLE_ROTATION_DEG)
    return max(LIN_MIN_SEGMENTS, by_translation, by_rotation), distance_mm, rotation_deg


def preflight_lin(node, ik_client, fk_client, start_joints, start_posx, start_quat,
                  end_pos_mm, end_quat, sol, line):
    """Validate every adaptive SLERP/straight-line sample on one fixed branch."""
    segments, distance_mm, rotation_deg = lin_segment_count(
        start_posx[:3], start_quat, end_pos_mm, end_quat)
    previous_joints = start_joints
    maximum_jump = 0.0

    for index in range(1, segments + 1):
        t = index / segments
        sample_pose = interpolate_posx(start_posx[:3], start_quat, end_pos_mm, end_quat, t)
        joints, reason = ik_solution(node, ik_client, fk_client, sample_pose, sol)
        if joints is None:
            raise InterpreterError(
                line, "E_CONFIG_CHANGE",
                f"LIN preflight sample {index}/{segments} (t={t:.4f}) rejected on "
                f"sol={sol}: {reason}. The asserted branch does not cover the path.")
        jump = max_joint_jump_deg(joints, previous_joints)
        if jump > LIN_MAX_JOINT_JUMP_DEG:
            raise InterpreterError(
                line, "E_CONFIG_CHANGE",
                f"LIN preflight joint jump {jump:.2f} deg between samples {index - 1} "
                f"and {index} exceeds {LIN_MAX_JOINT_JUMP_DEG} deg on sol={sol}.")
        maximum_jump = max(maximum_jump, jump)
        previous_joints = joints

    note("lin_preflight_ok", line=line, sol=sol,
         distance_mm=round(distance_mm, 3), rotation_deg=round(rotation_deg, 3),
         segments=segments, max_jump_deg=round(maximum_jump, 3))
    return previous_joints, distance_mm, rotation_deg, segments


# --------------------------------------------------------------------- #
# Motion execution
# --------------------------------------------------------------------- #
def ptp_timeout(start, target, velocity):
    max_delta = max_joint_jump_deg(start, target)
    speed = max(velocity * DOOSAN_PTP_MAX_VELOCITY_DEG_S, 1e-6)
    return max(MOTION_TIMEOUT_MINIMUM, max_delta / speed + MOTION_TIMEOUT_MARGIN)


def lin_timeout(distance_mm, rotation_deg, velocity_m_s):
    translation_time = distance_mm / max(velocity_m_s * 1000.0, 1e-6)
    rotation_time = rotation_deg / max(LIN_DEFAULT_ROT_VEL_DEG_S, 1e-6)
    return max(MOTION_TIMEOUT_MINIMUM,
               max(translation_time, rotation_time) + MOTION_TIMEOUT_MARGIN)


def issue_move_joint(node, move_client, target, velocity, acceleration, timeout):
    request = MoveJoint.Request()
    request.pos = [float(v) for v in target]
    request.vel = float(velocity * DOOSAN_PTP_MAX_VELOCITY_DEG_S)
    request.acc = float(acceleration * DOOSAN_PTP_MAX_ACCELERATION_DEG_S2)
    request.time = 0.0
    request.radius = 0.0
    request.mode = 0
    request.blend_type = 0
    request.sync_type = 0
    call_service(node, move_client, request, "the MoveJoint command", timeout)


def issue_move_line(node, move_line_client, end_posx, velocity, acceleration,
                    distance_mm, rotation_deg):
    request = MoveLine.Request()
    request.pos = [float(v) for v in end_posx]
    request.vel = [float(velocity * 1000.0), float(LIN_DEFAULT_ROT_VEL_DEG_S)]
    request.acc = [float(acceleration * 1000.0), float(LIN_DEFAULT_ROT_ACC_DEG_S2)]
    request.time = 0.0
    request.radius = 0.0
    request.ref = RTSL_REFERENCE
    request.mode = 0
    request.blend_type = 0
    request.sync_type = 0
    call_service(node, move_line_client, request, "the MoveLine command",
                 lin_timeout(distance_mm, rotation_deg, velocity))


def execute_plan(node, position_client, ik_client, fk_client, move_client,
                 move_line_client, plan):
    for line, task, payload in plan:
        try:
            if task == "set":
                emit(line, "ok", payload=payload)
                continue

            if task == "wait":
                time.sleep(payload)
                emit(line, "ok")
                continue

            if task == "ptp":
                target, velocity, acceleration = payload
                current = current_joints(node, position_client)
                if ENABLE_MOTION:
                    issue_move_joint(node, move_client, target, velocity, acceleration,
                                     ptp_timeout(current, target, velocity))
                    actual = current_joints(node, position_client)
                    mismatch = max_joint_jump_deg(actual, target)
                    if mismatch > PTP_END_MATCH_TOL_DEG:
                        raise InterpreterError(
                            line, "E_DEVICE",
                            f"PTP ended {mismatch:.2f} deg from commanded joint target.")
                emit(line, "ok", payload={"posj": [round(a, 4) for a in target],
                                          "motion": ENABLE_MOTION})
                continue

            if task == "move-ee-ptp":
                position_mm, quat_wxyz, requested_sol, velocity, acceleration = payload
                current = current_joints(node, position_client)
                target_posx = posx_from_pose(position_mm, quat_wxyz)
                sol, joints = select_ptp_solution(node, ik_client, fk_client, current,
                                                  target_posx, requested_sol, line)
                travel = max_joint_jump_deg(current, joints)
                if travel > LARGE_MOTION_WARN_DEG:
                    note("warning", line=line, max_joint_travel_deg=round(travel, 2),
                         detail="large joint travel to selected IK solution")
                if ENABLE_MOTION:
                    # Execute the exact solution validated above. MoveJointx is not used:
                    # it would make the controller solve IK again and could pick differently.
                    issue_move_joint(node, move_client, joints, velocity, acceleration,
                                     ptp_timeout(current, joints, velocity))
                    actual = current_joints(node, position_client)
                    mismatch = max_joint_jump_deg(actual, joints)
                    if mismatch > PTP_END_MATCH_TOL_DEG:
                        raise InterpreterError(
                            line, "E_DEVICE",
                            f"move-ee ptp ended {mismatch:.2f} deg from validated sol={sol}.")
                emit(line, "ok", payload={"sol": sol, "mode": "ptp",
                                          "posx_mm_zyz_deg": [round(v, 4) for v in target_posx],
                                          "posj": [round(a, 4) for a in joints],
                                          "motion": ENABLE_MOTION})
                continue

            if task == "move-ee-lin":
                if not DOOSAN_MOVELINE_ORIENTATION_IS_SLERP:
                    raise InterpreterError(
                        line, "E_UNSUPPORTED",
                        "LIN requires verified SLERP-equivalent Doosan MoveLine orientation "
                        "interpolation; set DOOSAN_MOVELINE_ORIENTATION_IS_SLERP only after "
                        "controller/documentation and physical-cell verification.")

                end_pos_mm, end_quat, requested_sol, velocity, acceleration = payload
                # Resolve immediately before executing this line from measured state.
                start_joints = current_joints(node, position_client)
                start_posx = raw_fk(node, fk_client, start_joints)
                if start_posx is None:
                    raise InterpreterError(line, "E_DEVICE",
                                           "Fkin failed on current joints before LIN motion.")
                start_quat = matrix_to_quaternion(zyz_degrees_to_matrix(*start_posx[3:]))

                if requested_sol is None:
                    sol, reason = determine_start_sol(node, ik_client, fk_client,
                                                      start_joints, start_posx)
                    if sol is None:
                        raise InterpreterError(line, "E_UNREACHABLE_CONFIG",
                                               f"Cannot infer LIN start branch: {reason}.")
                    note("lin_sol_asserted_from_start", line=line, sol=sol)
                else:
                    sol = requested_sol
                    assert_start_branch(node, ik_client, fk_client, start_joints,
                                        start_posx, sol, line)

                end_joints, distance_mm, rotation_deg, segments = preflight_lin(
                    node, ik_client, fk_client, start_joints, start_posx, start_quat,
                    end_pos_mm, end_quat, sol, line)
                end_posx = posx_from_pose(end_pos_mm, end_quat)

                if distance_mm < LIN_MIN_POS_FOR_TRANSLATION_MM:
                    note("warning", line=line,
                         detail=f"LIN pure rotation ({distance_mm:.3f} mm): translational "
                                "speed is irrelevant; rotational back-end default is used.")

                if ENABLE_MOTION:
                    issue_move_line(node, move_line_client, end_posx, velocity, acceleration,
                                    distance_mm, rotation_deg)
                    # Detect a controller-side branch switch after physical execution.
                    actual = current_joints(node, position_client)
                    mismatch = max_joint_jump_deg(actual, end_joints)
                    if mismatch > LIN_BRANCH_MATCH_TOL_DEG:
                        raise InterpreterError(
                            line, "E_CONFIG_CHANGE",
                            f"LIN controller end state differs from preflight sol={sol} by "
                            f"{mismatch:.2f} deg; possible controller-side branch switch.")

                emit(line, "ok", payload={
                    "sol": sol,
                    "mode": "lin",
                    "posx_mm_zyz_deg": [round(v, 4) for v in end_posx],
                    "posj": [round(a, 4) for a in end_joints],
                    "distance_mm": round(distance_mm, 3),
                    "rotation_deg": round(rotation_deg, 3),
                    "preflight_segments": segments,
                    "motion": ENABLE_MOTION,
                })
                continue

            raise RuntimeError(f"Internal error: unknown planned task {task!r}.")

        except RuntimeError as error:
            emit(line, "error", "E_DEVICE", str(error))
            return 1
        except InterpreterError as error:
            emit(error.line, "error", error.code, error.message)
            return 1
    return 0


def main():
    self_test()
    rclpy.init()
    node = rclpy.create_node("rtsl_doosan_executor")
    try:
        initial = initial_modal_values()
        report_initial_values(initial)
        with INSTRUCTION_FILE.open(encoding="utf-8") as file:
            plan = parse_program(json.load(file), initial)

        position_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)
        ik_client = node.create_client(Ikin, IKIN_SERVICE)
        move_client = node.create_client(MoveJoint, MOVE_JOINT_SERVICE)
        move_line_client = node.create_client(MoveLine, MOVE_LINE_SERVICE)
        fk_client = node.create_client(Fkin, FKIN_SERVICE) if HAVE_FKIN else None

        wait_for_service(position_client, POSITION_SERVICE)
        needs_cartesian = any(task in {"move-ee-ptp", "move-ee-lin"}
                              for _, task, _ in plan)
        needs_lin = any(task == "move-ee-lin" for _, task, _ in plan)
        needs_joint_motion = any(task in {"ptp", "move-ee-ptp"}
                                 for _, task, _ in plan)

        if needs_cartesian:
            wait_for_service(ik_client, IKIN_SERVICE)
            if fk_client is None or not wait_for_service(fk_client, FKIN_SERVICE, required=False):
                raise RuntimeError("move-ee requires Fkin for IK round-trip validation.")
        if ENABLE_MOTION and needs_joint_motion:
            wait_for_service(move_client, MOVE_JOINT_SERVICE)
        if ENABLE_MOTION and needs_lin:
            wait_for_service(move_line_client, MOVE_LINE_SERVICE)

        return execute_plan(node, position_client, ik_client, fk_client, move_client,
                            move_line_client, plan)
    except InterpreterError as error:
        emit(error.line, "error", error.code, error.message)
        return 1
    except AssertionError as error:
        emit(0, "error", "E_DEVICE", f"self-test failed: {error}")
        return 1
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        emit(0, "error", "E_DEVICE", str(error))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
