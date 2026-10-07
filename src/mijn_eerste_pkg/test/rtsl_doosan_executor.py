#!/usr/bin/env python3
"""RTSL v0.1 interpreter for a Doosan arm on ROS 2 (dsr_controller2).

v2 adds `move-ee mode=lin`:
  - straight Cartesian line, SLERP on orientation
  - branch constancy enforced by preflight sampling (Ikin+Fkin per sample
    with a fixed sol, joint-jump check between consecutive samples)
  - vel/acc are TCP translational m/s and m/s^2 (RTSL spec); rotational
    components use configurable defaults because RTSL v0.1 does not expose
    them separately.

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

UNITS AND CONVENTIONS
---------------------
RTSL v0.1 (General rules): poses are in metres and the orientation is a unit
quaternion in (w, x, y, z) order, Hamilton convention, world -> TCP. Doosan
posx is [mm, mm, mm, deg, deg, deg] with ZYZ Euler angles.

PTP vel/acc are fractions 0..1 of the back-end's PTP maxima.
LIN vel/acc are TCP translational m/s and m/s^2. See the KNOWN FOOTGUN note
in the modal-value warning: the same `set velocity 0.5` is interpreted
differently per mode. This is RTSL v0.1 as written; v0.2 should split them.

VALIDATION OF IK RESULTS (unchanged from v1)
--------------------------------------------
Ikin returns only `success`. Guards: Fkin round-trip and joint-limit check.

SOLUTION SPACE (unchanged, validated 40/40 on a0509)
----------------------------------------------------
    sol = 4*S + 2*E + W      S: SL=0 SR=1 | E: ED=0 EU=1 | W: WN=0 WF=1
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, Ikin, MoveJoint, MoveJointx, MoveLine

try:
    from dsr_msgs2.srv import Fkin
    HAVE_FKIN = True
except ImportError:
    Fkin = None
    HAVE_FKIN = False

# =========================== USER SETTINGS ============================
ROBOT_ID = "dsr01"
INSTRUCTION_FILE = Path("/home/bbrug1/ROS2V2_ws/src/mijn_eerste_pkg/test/Human.json")

ENABLE_MOTION = False

ROBOT_MODEL = "a0509"
CONTROLLER_VERSION = "DRCF GF02100100 / DRFL GL013304"

BACKEND_INITIAL_VELOCITY = 0.01
BACKEND_INITIAL_ACCELERATION = 0.01
BACKEND_INITIAL_MAX_VELOCITY = 1.0
BACKEND_INITIAL_MAX_ACCELERATION = 1.0

# PTP backend maxima (fraction 1.0 -> this value). VERIFY on your controller.
DOOSAN_PTP_MAX_VELOCITY_DEG_S = 150.0
DOOSAN_PTP_MAX_ACCELERATION_DEG_S2 = 150.0

# LIN caps in RTSL SI units. VERIFY against your cell safety envelope.
# a0509 datasheet TCP speed ~1 m/s; this is a conservative cell default.
BACKEND_LIN_MAX_VELOCITY_M_S = 0.25
BACKEND_LIN_MAX_ACCELERATION_M_S2 = 1.0

# Rotational companions for MoveLine. RTSL v0.1 has no separate field, so
# these are a back-end policy. Keep them modest; raise only after visual
# verification on the real cell.
LIN_DEFAULT_ROT_VEL_DEG_S = 30.0
LIN_DEFAULT_ROT_ACC_DEG_S2 = 60.0

# LIN preflight (branch-constancy sampling).
LIN_SAMPLES = 20                 # inclusive of start and end
LIN_MAX_JOINT_JUMP_DEG = 15.0    # per-joint drempel between adjacent samples
LIN_MIN_POS_FOR_TRANSLATION_MM = 0.1   # below this we treat as pure rotation

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
MOVE_JOINTX_SERVICE = f"{CONTROLLER_NAMESPACE}/motion/move_jointx"
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
        self.line, self.code, self.message = line, code, message


def emit(line, status, code=None, payload=None):
    print(json.dumps({"line": line, "status": status,
                      "code": code, "payload": payload}), flush=True)


def note(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


# --------------------------------------------------------------------- #
# scalar parsing (unchanged)
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
    if values.velocity > values.max_velocity or values.acceleration > values.max_acceleration:
        raise ValueError("Initial velocity/acceleration cannot exceed their configured caps.")
    return values


def report_initial_values(values):
    note("initial_modal_values",
         robot_model=ROBOT_MODEL,
         controller_version=CONTROLLER_VERSION,
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
         quaternion_order="wxyz",
         fkin_validation=HAVE_FKIN,
         motion_enabled=ENABLE_MOTION)
    if not HAVE_FKIN:
        note("warning", detail="Fkin unavailable: IK solutions are checked against "
                               "joint limits only, not against the requested pose. "
                               "LIN preflight cannot run reliably.")


def motion_value(instruction, line, short_name, long_name, default):
    present = [name for name in (short_name, long_name) if name in instruction]
    if len(present) > 1:
        raise InterpreterError(line, "E_PARSE",
                               f"Use only one of {short_name} and {long_name}.")
    return finite_positive(instruction[present[0]], line, present[0]) if present else default


def validate_ptp(line, velocity, acceleration, modal):
    if velocity > 1.0 or velocity > modal.max_velocity:
        raise InterpreterError(line, "E_LIMIT",
                               "PTP velocity exceeds its cap or the portable range 0-1.")
    if acceleration > 1.0 or acceleration > modal.max_acceleration:
        raise InterpreterError(line, "E_LIMIT",
                               "PTP acceleration exceeds its cap or the portable range 0-1.")


def validate_lin(line, velocity, acceleration, modal):
    """RTSL v0.1: LIN vel/acc are TCP translational m/s and m/s^2."""
    if velocity > modal.max_velocity:
        raise InterpreterError(line, "E_LIMIT",
                               "LIN velocity exceeds the modal max-velocity.")
    if acceleration > modal.max_acceleration:
        raise InterpreterError(line, "E_LIMIT",
                               "LIN acceleration exceeds the modal max-acceleration.")
    if velocity > BACKEND_LIN_MAX_VELOCITY_M_S:
        raise InterpreterError(line, "E_LIMIT",
                               f"LIN velocity {velocity} m/s exceeds back-end cap "
                               f"{BACKEND_LIN_MAX_VELOCITY_M_S} m/s.")
    if acceleration > BACKEND_LIN_MAX_ACCELERATION_M_S2:
        raise InterpreterError(line, "E_LIMIT",
                               f"LIN acceleration {acceleration} m/s^2 exceeds back-end cap "
                               f"{BACKEND_LIN_MAX_ACCELERATION_M_S2} m/s^2.")


def warn_modal_ambiguity(line, param, value):
    """RTSL v0.1 does not separate ptp (fraction) and lin (m/s) modals.
    Shout about it on every change so the author cannot miss it."""
    if param not in {"velocity", "acceleration", "max-velocity", "max-acceleration"}:
        return
    kind = "velocity" if "velocity" in param else "acceleration"
    ptp_unit = "fraction 0..1"
    lin_unit_v = "m/s"
    lin_unit_a = "m/s^2"
    note("warning_modal_ambiguity",
         line=line, param=param, value=value,
         detail=f"RTSL v0.1 has one modal '{kind}' shared by ptp ({ptp_unit}) "
                f"and lin ({lin_unit_v if kind == 'velocity' else lin_unit_a}). "
                f"The same number means different things in each mode. "
                f"Prefer explicit per-line vel()/acc() overrides.")


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
                               f"Joint angles outside the configured limits: {violations}")
    return joints


def parse_config(value, line):
    if not isinstance(value, dict) or set(value) != {"shoulder", "elbow", "wrist"}:
        raise InterpreterError(line, "E_PARSE",
                               "config needs shoulder, elbow, and wrist fields.")
    config = (value["shoulder"], value["elbow"], value["wrist"])
    if (config[0] not in {"SL", "SR"} or config[1] not in {"EU", "ED"}
            or config[2] not in {"WN", "WF"}):
        raise InterpreterError(line, "E_PARSE", "Invalid canonical IK configuration.")
    if config not in CANONICAL_CONFIG_TO_DOOSAN_SOL:
        raise InterpreterError(line, "E_UNSUPPORTED",
                               "No experimentally validated Doosan sol mapping for this config.")
    sol = CANONICAL_CONFIG_TO_DOOSAN_SOL[config]
    if isinstance(sol, bool) or not isinstance(sol, int) or not 0 <= sol <= 7:
        raise InterpreterError(line, "E_UNSUPPORTED",
                               "Configured Doosan solution-space mapping must be an integer 0-7.")
    return config, sol


# --------------------------------------------------------------------- #
# orientation + quaternion math (SLERP added)
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
        raise InterpreterError(line, "E_PARSE",
                               "quat must be [w, x, y, z] (RTSL v0.1 order).")
    w, x, y, z = (_as_float(v, line, "quat") for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-3:
        raise InterpreterError(line, "E_PARSE",
                               "Quaternion norm deviates from 1 by more than 1e-3.")
    return [w / norm, x / norm, y / norm, z / norm]


def quaternion_to_zyz_degrees(quat, line):
    w, x, y, z = quaternion_normalise(quat, line)
    return matrix_to_zyz_degrees(quaternion_to_matrix(w, x, y, z))


def matrix_to_quaternion(r):
    """Return (w, x, y, z). Shepperd's method, numerically stable."""
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s
    return [w, x, y, z]


def quaternion_slerp(q0, q1, t):
    """SLERP between unit quaternions (w,x,y,z). t in [0,1]."""
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:          # take the shorter arc
        q1 = [-v for v in q1]
        dot = -dot
    if dot > 0.9995:       # nearly parallel: linear + renormalise
        out = [a + t * (b - a) for a, b in zip(q0, q1)]
        n = math.sqrt(sum(v * v for v in out))
        return [v / n for v in out]
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return [s0 * a + s1 * b for a, b in zip(q0, q1)]


def parse_pose(value, line):
    """RTSL pose (metres + wxyz quat) -> Doosan posx [mm, mm, mm, deg, deg, deg]."""
    if not isinstance(value, dict) or set(value) != {"pos", "quat"}:
        raise InterpreterError(line, "E_PARSE", "pose needs pos and quat fields.")
    pos = value["pos"]
    if not isinstance(pos, list) or len(pos) != 3:
        raise InterpreterError(line, "E_PARSE", "pose.pos must be [x, y, z] in metres.")
    xyz = [_as_float(c, line, "pose.pos") for c in pos]
    return [c * 1000.0 for c in xyz] + quaternion_to_zyz_degrees(value["quat"], line)


def parse_pose_wxyz(value, line):
    """Same as parse_pose but keeps the quaternion as (w,x,y,z) for SLERP."""
    if not isinstance(value, dict) or set(value) != {"pos", "quat"}:
        raise InterpreterError(line, "E_PARSE", "pose needs pos and quat fields.")
    pos = value["pos"]
    if not isinstance(pos, list) or len(pos) != 3:
        raise InterpreterError(line, "E_PARSE", "pose.pos must be [x, y, z] in metres.")
    xyz_mm = [_as_float(c, line, "pose.pos") * 1000.0 for c in pos]
    quat = quaternion_normalise(value["quat"], line)
    return xyz_mm, quat


def self_test():
    r = quaternion_to_matrix(0.0, 1.0, 0.0, 0.0)
    assert abs(r[0][0] - 1.0) < 1e-12 and abs(r[2][2] + 1.0) < 1e-12, r
    quat = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]
    zyz = quaternion_to_zyz_degrees(quat, 0)
    assert rotation_angle_deg(quaternion_to_matrix(*quat),
                              zyz_degrees_to_matrix(*zyz)) < 1e-6, zyz
    assert sorted(CANONICAL_CONFIG_TO_DOOSAN_SOL.values()) == list(range(8))
    # SLERP sanity: midpoint of 0 and 180 deg about x is 90 deg about x.
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = [0.0, 1.0, 0.0, 0.0]
    qh = quaternion_slerp(q0, q1, 0.5)
    expected = [math.cos(math.radians(45.0)), math.sin(math.radians(45.0)), 0.0, 0.0]
    assert max(abs(a - b) for a, b in zip(qh, expected)) < 1e-9, qh


# --------------------------------------------------------------------- #
# program parsing
# --------------------------------------------------------------------- #
def parse_program(instructions, initial):
    if not isinstance(instructions, list):
        raise InterpreterError(0, "E_PARSE", "The JSON root must be a list of instructions.")
    modal = ModalValues(**vars(initial))
    plan = []
    for line, instruction in enumerate(instructions, start=1):
        if not isinstance(instruction, dict) or not isinstance(instruction.get("task"), str):
            raise InterpreterError(line, "E_PARSE",
                                   "Each instruction must be an object with a string task.")
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
                    raise InterpreterError(line, "E_LIMIT",
                                           "acceleration exceeds max-acceleration.")
                modal.acceleration = value
            elif param == "max-velocity":
                if value < modal.velocity:
                    raise InterpreterError(line, "E_LIMIT",
                                           "max-velocity cannot be below the active velocity.")
                modal.max_velocity = value
            else:
                if value < modal.acceleration:
                    raise InterpreterError(line, "E_LIMIT",
                                           "max-acceleration cannot be below the active acceleration.")
                modal.max_acceleration = value
            plan.append((line, "set", {"param": param, "value": value}))
        elif task in {"ptp", "move_joint"}:
            allowed = {"task", "value", "vel", "velocity", "acc", "acceleration"}
            if not set(instruction) <= allowed or "value" not in instruction:
                raise InterpreterError(line, "E_PARSE", "Invalid PTP instruction.")
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            validate_ptp(line, velocity, acceleration, modal)
            plan.append((line, "ptp",
                         (parse_joints(instruction["value"], line), velocity, acceleration)))
        elif task == "move-ee":
            allowed = {"task", "mode", "pose", "config", "vel", "velocity", "acc", "acceleration"}
            if (not set(instruction) <= allowed
                    or instruction.get("mode") not in {"ptp", "lin"}
                    or "pose" not in instruction):
                raise InterpreterError(line, "E_PARSE", "Invalid move-ee instruction.")
            mode = instruction["mode"]
            velocity = motion_value(instruction, line, "vel", "velocity", modal.velocity)
            acceleration = motion_value(instruction, line, "acc", "acceleration", modal.acceleration)
            sol = parse_config(instruction["config"], line)[1] if "config" in instruction else None
            if mode == "ptp":
                validate_ptp(line, velocity, acceleration, modal)
                plan.append((line, "move-ee-ptp",
                             (parse_pose(instruction["pose"], line), sol,
                              velocity, acceleration)))
            else:
                validate_lin(line, velocity, acceleration, modal)
                pose_mm, quat_wxyz = parse_pose_wxyz(instruction["pose"], line)
                plan.append((line, "move-ee-lin",
                             (pose_mm, quat_wxyz, sol, velocity, acceleration)))
        elif task == "wait":
            if set(instruction) != {"task", "value"}:
                raise InterpreterError(line, "E_PARSE", "Invalid wait instruction.")
            plan.append((line, "wait", finite_nonnegative(instruction["value"], line, "value")))
        elif task in {"take-image", "gripper"}:
            raise InterpreterError(line, "E_UNSUPPORTED",
                                   f"This back-end targets a bare robot controller: "
                                   f"{task} is not available.")
        else:
            raise InterpreterError(line, "E_UNSUPPORTED", f"Unsupported task: {task}.")
    return plan


# --------------------------------------------------------------------- #
# ROS plumbing (unchanged)
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
        raise RuntimeError(f"The controller rejected or failed {description}.")
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
    return [float(angle) for angle in response.pos]


def raw_ik(node, ik_client, pose, sol):
    request = Ikin.Request()
    request.pos = [float(v) for v in pose]
    request.sol_space = int(sol)
    request.ref = 0
    response = try_call(node, ik_client, request, IK_RESPONSE_TIMEOUT)
    if response is None:
        return None
    return [float(angle) for angle in response.conv_posj]


def raw_fk(node, fk_client, joints):
    if fk_client is None:
        return None
    request = Fkin.Request()
    request.pos = [float(angle) for angle in joints]
    request.ref = 0
    response = try_call(node, fk_client, request, IK_RESPONSE_TIMEOUT)
    if response is None:
        return None
    return [float(v) for v in response.conv_posx]


def ik_solution(node, ik_client, fk_client, pose, sol):
    joints = raw_ik(node, ik_client, pose, sol)
    if joints is None:
        return None, "ikin reported failure"
    violations = joint_limit_violations(joints)
    if violations:
        return None, f"joint limits violated: {violations}"
    if fk_client is not None:
        posx = raw_fk(node, fk_client, joints)
        if posx is None:
            return None, "fkin round-trip failed"
        dp = math.sqrt(sum((a - b) ** 2 for a, b in zip(posx[:3], pose[:3])))
        drot = rotation_angle_deg(zyz_degrees_to_matrix(*posx[3:]),
                                  zyz_degrees_to_matrix(*pose[3:]))
        if dp > FKIN_POSITION_TOL_MM or drot > FKIN_ORIENTATION_TOL_DEG:
            return None, f"fkin round-trip off by {dp:.3f} mm / {drot:.3f} deg"
    return joints, None


def angular_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


# --------------------------------------------------------------------- #
# LIN preflight
# --------------------------------------------------------------------- #
def interpolate_posx(start_pos_mm, start_quat, end_pos_mm, end_quat, t):
    """Linear on position (mm), SLERP on orientation. Returns Doosan posx."""
    pos = [a + t * (b - a) for a, b in zip(start_pos_mm, end_pos_mm)]
    quat = quaternion_slerp(start_quat, end_quat, t)
    zyz = matrix_to_zyz_degrees(quaternion_to_matrix(*quat))
    return pos + zyz


def determine_start_sol(node, ik_client, fk_client, start_joints):
    """Infer which sol reproduces the current joint configuration."""
    start_posx = raw_fk(node, fk_client, start_joints)
    if start_posx is None:
        return None, start_posx, "fkin on current joints failed"
    best_sol, best_err = None, float("inf")
    for sol in range(8):
        joints, reason = ik_solution(node, ik_client, fk_client, start_posx, sol)
        if joints is None:
            continue
        err = angular_distance(joints, start_joints)
        if err < best_err:
            best_err, best_sol = err, sol
    if best_sol is None:
        return None, start_posx, "no sol reproduces the current pose"
    return best_sol, start_posx, None


def preflight_lin(node, ik_client, fk_client,
                  start_joints, start_posx, start_quat,
                  end_pos_mm, end_quat, sol, line):
    """Sample the straight-line path and verify branch constancy.
    Returns (joints_at_end, samples_meta) or raises InterpreterError."""
    if fk_client is None:
        raise InterpreterError(line, "E_UNSUPPORTED",
                               "LIN preflight requires Fkin; this build lacks it.")

    start_pos_mm = start_posx[:3]
    distance_mm = math.sqrt(sum((a - b) ** 2 for a, b in zip(start_pos_mm, end_pos_mm)))
    rot_deg = rotation_angle_deg(quaternion_to_matrix(*start_quat),
                                 quaternion_to_matrix(*end_quat))

    previous_joints = start_joints
    samples_meta = []
    for i in range(1, LIN_SAMPLES + 1):      # skip i=0 (start) which is current
        t = i / LIN_SAMPLES
        sample_pose = interpolate_posx(start_pos_mm, start_quat,
                                       end_pos_mm, end_quat, t)
        joints, reason = ik_solution(node, ik_client, fk_client, sample_pose, sol)
        if joints is None:
            raise InterpreterError(
                line, "E_CONFIG_CHANGE",
                f"LIN preflight: sample {i}/{LIN_SAMPLES} (t={t:.3f}) rejected "
                f"on sol={sol}: {reason}. The requested branch does not cover "
                f"the full straight-line path.")
        jump = max(abs(a - b) for a, b in zip(joints, previous_joints))
        if jump > LIN_MAX_JOINT_JUMP_DEG:
            raise InterpreterError(
                line, "E_CONFIG_CHANGE",
                f"LIN preflight: joint-jump {jump:.2f} deg between samples "
                f"{i-1} and {i} exceeds drempel {LIN_MAX_JOINT_JUMP_DEG} deg. "
                f"Likely a branch switch or near-singularity on sol={sol}.")
        samples_meta.append({"t": round(t, 3), "jump_deg": round(jump, 3)})
        previous_joints = joints

    note("lin_preflight_ok", line=line, sol=sol,
         distance_mm=round(distance_mm, 3),
         rotation_deg=round(rot_deg, 3),
         samples=LIN_SAMPLES,
         max_jump_deg=round(max(s["jump_deg"] for s in samples_meta), 3))
    return previous_joints


# --------------------------------------------------------------------- #
# plan resolution
# --------------------------------------------------------------------- #
def resolve_move_ee_plan(node, position_client, ik_client, fk_client, plan):
    current = current_joints(node, position_client)
    note("preflight_start", current_posj=[round(a, 4) for a in current])
    resolved = []
    for line, task, payload in plan:
        if task == "ptp":
            current = payload[0]
            resolved.append((line, task, payload))
            continue
        if task == "move-ee-ptp":
            pose, requested_sol, velocity, acceleration = payload
            candidates, rejected = {}, {}
            for sol in range(8):
                joints, reason = ik_solution(node, ik_client, fk_client, pose, sol)
                if joints is None:
                    rejected[sol] = reason
                else:
                    candidates[sol] = joints

            note("ik_candidates", line=line,
                 doosan_pose_mm_zyz_deg=[round(v, 4) for v in pose],
                 requested_sol=requested_sol,
                 valid_solutions=sorted(candidates),
                 rejected=rejected)

            if not candidates:
                raise InterpreterError(line, "E_UNREACHABLE",
                                       "Target pose has no valid IK solution.")

            if requested_sol is None:
                sol = min(candidates, key=lambda s: angular_distance(current, candidates[s]))
            elif requested_sol in candidates:
                sol = requested_sol
            else:
                raise InterpreterError(
                    line, "E_UNREACHABLE_CONFIG",
                    f"Requested IK branch sol={requested_sol} is not valid at the target "
                    f"({rejected.get(requested_sol, 'unknown reason')}).")

            joints = candidates[sol]
            travel = max(abs(a - b) for a, b in zip(current, joints))
            if travel > LARGE_MOTION_WARN_DEG:
                note("warning", line=line, max_joint_travel_deg=round(travel, 2),
                     detail="large joint travel; SL/SR branches coincide in Cartesian "
                            "space on arms without shoulder offset.")

            resolved.append((line, task, (pose, sol, velocity, acceleration, joints)))
            current = joints
            continue

        if task == "move-ee-lin":
            end_pos_mm, end_quat, requested_sol, velocity, acceleration = payload
            # Determine start pose + branch
            start_posx = raw_fk(node, fk_client, current) if fk_client else None
            if start_posx is None:
                raise InterpreterError(line, "E_UNSUPPORTED",
                                       "LIN requires Fkin to establish the start pose.")
            start_quat = matrix_to_quaternion(zyz_degrees_to_matrix(*start_posx[3:]))

            if requested_sol is None:
                sol, _, reason = determine_start_sol(node, ik_client, fk_client, current)
                if sol is None:
                    raise InterpreterError(line, "E_UNREACHABLE_CONFIG",
                                           f"Cannot infer start sol: {reason}.")
                note("lin_sol_asserted_from_start", line=line, sol=sol)
            else:
                sol = requested_sol

            # Reject pure-rotation runs with no translational motion only to
            # warn; keep executing (SLERP + rotational default handle it).
            distance_mm = math.sqrt(sum((a - b) ** 2
                                        for a, b in zip(start_posx[:3], end_pos_mm)))
            if distance_mm < LIN_MIN_POS_FOR_TRANSLATION_MM:
                note("warning", line=line,
                     detail=f"LIN pure rotation (distance {distance_mm:.3f} mm). "
                            f"Translational vel ignored; rotational default used.")

            end_joints = preflight_lin(node, ik_client, fk_client,
                                       current, start_posx, start_quat,
                                       end_pos_mm, end_quat, sol, line)

            # End pose as Doosan posx (what MoveLine expects)
            end_posx = (list(end_pos_mm)
                        + matrix_to_zyz_degrees(quaternion_to_matrix(*end_quat)))
            resolved.append((line, task, (end_posx, sol, velocity, acceleration,
                                          end_joints, distance_mm)))
            current = end_joints
            continue

        resolved.append((line, task, payload))
    return resolved


# --------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------- #
def ptp_timeout(start, target, velocity):
    max_delta = max(abs(a - b) for a, b in zip(start, target))
    speed = max(velocity * DOOSAN_PTP_MAX_VELOCITY_DEG_S, 1e-6)
    return max(MOTION_TIMEOUT_MINIMUM, max_delta / speed + MOTION_TIMEOUT_MARGIN)


def lin_timeout(distance_mm, velocity_m_s):
    speed_mm_s = max(velocity_m_s * 1000.0, 1e-6)
    return max(MOTION_TIMEOUT_MINIMUM, distance_mm / speed_mm_s + MOTION_TIMEOUT_MARGIN)


def execute_plan(node, position_client, move_client, move_jointx_client,
                 move_line_client, plan):
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
                    call_service(node, move_client, request, "the MoveJoint command",
                                 ptp_timeout(current, target, velocity))
                current = target
                emit(line, "ok", payload={"posj": [round(a, 4) for a in target],
                                          "motion": ENABLE_MOTION})
            elif task == "move-ee-ptp":
                pose, sol, velocity, acceleration, joints = payload
                if ENABLE_MOTION:
                    request = MoveJointx.Request()
                    request.pos = [float(v) for v in pose]
                    request.sol = int(sol)
                    request.vel = velocity * DOOSAN_PTP_MAX_VELOCITY_DEG_S
                    request.acc = acceleration * DOOSAN_PTP_MAX_ACCELERATION_DEG_S2
                    request.time, request.radius = 0.0, 0.0
                    request.ref, request.mode = 0, 0
                    request.blend_type, request.sync_type = 0, 0
                    call_service(node, move_jointx_client, request,
                                 "the MoveJointx command",
                                 ptp_timeout(current, joints, velocity))
                current = joints
                emit(line, "ok", payload={"sol": sol, "mode": "ptp",
                                          "posj": [round(a, 4) for a in joints],
                                          "motion": ENABLE_MOTION})
            elif task == "move-ee-lin":
                end_posx, sol, velocity, acceleration, end_joints, distance_mm = payload
                if ENABLE_MOTION:
                    request = MoveLine.Request()
                    request.pos = [float(v) for v in end_posx]
                    # vel = [mm/s, deg/s], acc = [mm/s^2, deg/s^2]
                    request.vel = [float(velocity * 1000.0),
                                   float(LIN_DEFAULT_ROT_VEL_DEG_S)]
                    request.acc = [float(acceleration * 1000.0),
                                   float(LIN_DEFAULT_ROT_ACC_DEG_S2)]
                    request.time, request.radius = 0.0, 0.0
                    request.ref = 0        # DR_BASE; set to 2 for DR_WORLD if applicable
                    request.mode = 0       # absolute
                    request.blend_type, request.sync_type = 0, 0
                    call_service(node, move_line_client, request,
                                 "the MoveLine command",
                                 lin_timeout(distance_mm, velocity))
                current = end_joints
                emit(line, "ok", payload={"sol": sol, "mode": "lin",
                                          "posx_mm_zyz_deg": [round(v, 4) for v in end_posx],
                                          "posj": [round(a, 4) for a in end_joints],
                                          "distance_mm": round(distance_mm, 3),
                                          "motion": ENABLE_MOTION})
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
        move_jointx_client = node.create_client(MoveJointx, MOVE_JOINTX_SERVICE)
        move_line_client = node.create_client(MoveLine, MOVE_LINE_SERVICE)
        fk_client = node.create_client(Fkin, FKIN_SERVICE) if HAVE_FKIN else None

        wait_for_service(position_client, POSITION_SERVICE)

        needs_ik = any(task in ("move-ee-ptp", "move-ee-lin") for _, task, _ in plan)
        needs_lin = any(task == "move-ee-lin" for _, task, _ in plan)

        if needs_ik:
            wait_for_service(ik_client, IKIN_SERVICE)
            if fk_client is not None and not wait_for_service(fk_client, FKIN_SERVICE,
                                                              required=False):
                fk_client = None
            if needs_lin and fk_client is None:
                raise RuntimeError("LIN motion requires Fkin; service unavailable.")
            plan = resolve_move_ee_plan(node, position_client, ik_client, fk_client, plan)
            if ENABLE_MOTION:
                if any(task == "move-ee-ptp" for _, task, _ in plan):
                    wait_for_service(move_jointx_client, MOVE_JOINTX_SERVICE)
                if needs_lin:
                    wait_for_service(move_line_client, MOVE_LINE_SERVICE)

        if ENABLE_MOTION and any(task == "ptp" for _, task, _ in plan):
            wait_for_service(move_client, MOVE_JOINT_SERVICE)

        return execute_plan(node, position_client, move_client, move_jointx_client,
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
