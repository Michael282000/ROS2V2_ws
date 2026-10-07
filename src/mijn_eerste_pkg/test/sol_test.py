#!/usr/bin/env python3
"""Calibrate CANONICAL_CONFIG_TO_DOOSAN_SOL for a Doosan arm.

THIS SCRIPT DOES NOT MOVE THE ROBOT.
Only Ikin / Fkin / GetCurrentPosj clients are created. Ikin and Fkin are pure
computation inside the controller; GetCurrentPosj is an encoder read-out.
No MoveJoint / MoveJointx client is ever created, so no motion command can be
emitted by this file.

WHY THE NAIVE sign(q3) TEST FAILS
---------------------------------
The obvious hypothesis "bit1 of sol == sign(q3)" is WRONG. Measured on an
a0509, the sign of q3 depends on bit1 AND bit2:

    sign(q3) > 0  <=>  bit1 XOR bit2

Reason: bit2 rotates joint 1 by 180 deg and negates q2/q3. The arm's *points*
(shoulder, elbow, wrist centre) are unchanged, but the joint signs flip. A
mapping built on sign(q3) alone is silently wrong for sol 4..7.

This script therefore reconstructs the elbow POINT in 3D and applies the RTSL
definition directly: "EU when the elbow point lies above the line from shoulder
point to wrist centre". That test is frame-invariant and needs no human
judgement, so the elbow bit is resolved fully numerically.

HOW THE GEOMETRY IS OBTAINED
----------------------------
The planar arm model (d1, L2, L3, d6) is fitted by Gauss-Newton from the Ikin
solutions themselves, by minimising the kinematic closure residual

    || E(q1, q2) - WC(pose) || - L3

over all (pose, sol) pairs. On the reference data set this converges to a
max residual of 2.3e-4 mm, i.e. the model is exact. No DH table or datasheet
is needed, and the fit is self-validating: a large residual means the model
does not apply to your robot and the script refuses to emit a mapping.

ON THE SHOULDER BIT
-------------------
RTSL v0.1 defines SL/SR as "wrist centre left/right of the vertical plane
through the base axis". For a FIXED TCP pose the wrist centre is fixed, so the
test is degenerate. Worse, on an arm without a shoulder offset the sol s and
sol s+4 branches place shoulder, elbow and wrist centre at IDENTICAL points.
The script detects this degeneracy and reports it. The only operational
criterion left is whether joint 1 points toward the wrist-centre azimuth or
~180 deg away from it. The association SL <-> Doosan "Lefty" is then a LABEL
CONVENTION, not a measurement. Flag for RTSL v0.2.

Doosan reference enumeration (movejx / ikin manual):
    sol : shoulder(MSB) elbow wrist(LSB)
    0 000 Lefty  Below  NoFlip     4 100 Righty Below  NoFlip
    1 001 Lefty  Below  Flip       5 101 Righty Below  Flip
    2 010 Lefty  Above  NoFlip     6 110 Righty Above  NoFlip
    3 011 Lefty  Above  Flip       7 111 Righty Above  Flip
"""

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from dsr_msgs2.srv import GetCurrentPosj, Ikin

try:
    from dsr_msgs2.srv import Fkin
    HAVE_FKIN = True
except ImportError:  # older dsr_msgs2 builds
    Fkin = None
    HAVE_FKIN = False

# =========================== USER SETTINGS ============================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "a0509"           # the mapping is model-specific
CONTROLLER_VERSION = "UNKNOWN"  # fill in: the mapping is firmware-specific

# Doosan posx: [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg] (ZYZ Euler).
# Pick well-conditioned poses: reachable, away from the base column, spread in
# azimuth, and ideally at least one with |q5| near 90 deg (far from both wrist
# singularities at q5 = 0 and q5 = +-180). The script reports the margin.
TEST_POSES = [
    [400.0, -100.0, 250.0, 0.0, 180.0, 0.0],
    [500.0, 150.0, 450.0, 30.0, 150.0, -20.0],
    [350.0, 300.0, 600.0, -45.0, 120.0, 60.0],
    [450.0, -250.0, 400.0, 60.0, 95.0, -30.0],
    [300.0, 350.0, 300.0, -120.0, 85.0, 40.0],
]

# Which Doosan shoulder label is RTSL "SL". Doosan calls bit2 == 0 "Lefty".
# This is a naming convention, not a measurement (see module docstring).
SL_IS_BIT2_CLEAR = True

# --- tolerances -------------------------------------------------------
KINEMATIC_FIT_TOL_MM = 0.05      # max closure residual to accept the model
FKIN_POSITION_TOL_MM = 0.5       # Ikin -> Fkin round-trip position tolerance
FKIN_ORIENTATION_TOL_DEG = 0.5   # Ikin -> Fkin round-trip rotation tolerance
WRIST_SINGULARITY_MARGIN_DEG = 15.0   # warn when |q5| or 180-|q5| is below
ELBOW_PERP_MIN_MM = 5.0          # ignore near-straight arms (no elbow side)
AZIMUTH_TOLERANCE_DEG = 60.0     # band for "q1 points toward the target"
SHOULDER_DEGENERACY_TOL_MM = 1.0 # below this, sol s and s+4 coincide in space

IK_RESPONSE_TIMEOUT = 10.0
SERVICE_TIMEOUT = 10.0
REPORT_FILE = Path("/tmp/doosan_sol_calibration.json")
# =======================================================================

NS = f"/{ROBOT_ID}/dsr_controller2"
IKIN_SERVICE = f"{NS}/motion/ikin"
FKIN_SERVICE = f"{NS}/motion/fkin"
POSITION_SERVICE = f"{NS}/aux_control/get_current_posj"

J1, J2, J3, J5 = 0, 1, 2, 4
Z_AXIS = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------- #
def zyz_matrix(a_deg, b_deg, c_deg):
    """Doosan posx orientation (ZYZ Euler, degrees) -> rotation matrix."""
    a, b, c = (math.radians(v) for v in (a_deg, b_deg, c_deg))
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b),
                              math.sin(b), math.cos(c), math.sin(c))
    return np.array([
        [ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb],
        [sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb],
        [-sb * cc,                sb * sc,                cb],
    ])


def tcp_z_axis(pose):
    return zyz_matrix(pose[3], pose[4], pose[5])[:, 2]


def rotation_angle_deg(r_a, r_b):
    """Geodesic angle between two rotation matrices, in degrees."""
    trace = float(np.trace(r_a.T @ r_b))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def wrap180(angle):
    return (angle + 180.0) % 360.0 - 180.0


def wrist_centre(pose, d6):
    return np.array(pose[:3], dtype=float) - d6 * tcp_z_axis(pose)


def elbow_point(q1_deg, q2_deg, d1, l2):
    """Elbow position in the base frame from joints 1 and 2."""
    q1, q2 = math.radians(q1_deg), math.radians(q2_deg)
    u = np.array([math.cos(q1), math.sin(q1), 0.0])
    return np.array([0.0, 0.0, d1]) + l2 * (math.sin(q2) * u +
                                            math.cos(q2) * Z_AXIS)


# --------------------------------------------------------------------- #
# ROS plumbing
# --------------------------------------------------------------------- #
def wait(client, name, required=True):
    if client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
        return True
    if required:
        raise RuntimeError(f"Service not found: {name}")
    print(f"  [warn] optional service unavailable: {name}")
    return False


def call(node, client, request, timeout):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done():
        return None
    response = future.result()
    if response is None or not getattr(response, "success", False):
        return None
    return response


def ikin(node, client, pose, sol):
    request = Ikin.Request()
    request.pos, request.sol_space, request.ref = pose, sol, 0
    response = call(node, client, request, IK_RESPONSE_TIMEOUT)
    if response is None:
        return None
    return [round(float(a), 4) for a in response.conv_posj]


def fkin(node, client, joints):
    """Return posx for the given posj, or None."""
    if client is None:
        return None
    request = Fkin.Request()
    request.pos, request.ref = [float(a) for a in joints], 0
    response = call(node, client, request, IK_RESPONSE_TIMEOUT)
    if response is None:
        return None
    return [float(v) for v in response.conv_posx]


# --------------------------------------------------------------------- #
# step 1 - probe
# --------------------------------------------------------------------- #
def probe(node, ik_client, fk_client):
    """Return [{pose, joints, sol, ...}] for every valid Ikin solution."""
    records, warnings = [], []
    for index, pose in enumerate(TEST_POSES):
        print(f"\n--- pose {index}: {pose}")
        for sol in range(8):
            joints = ikin(node, ik_client, pose, sol)
            if joints is None:
                print(f"  sol {sol} ({sol:03b}) : no solution")
                continue

            # Ikin.srv has no status field, so verify the solution by pushing
            # it back through Fkin. Without this, out-of-workspace or singular
            # results are silently accepted as valid branches.
            round_trip = fkin(node, fk_client, joints)
            if round_trip is not None:
                dp = float(np.linalg.norm(np.array(round_trip[:3]) -
                                          np.array(pose[:3])))
                dr = rotation_angle_deg(zyz_matrix(*round_trip[3:]),
                                        zyz_matrix(*pose[3:]))
                if dp > FKIN_POSITION_TOL_MM or dr > FKIN_ORIENTATION_TOL_DEG:
                    print(f"  sol {sol} ({sol:03b}) : REJECTED by Fkin "
                          f"round-trip (dp={dp:.3f} mm, dr={dr:.3f} deg)")
                    warnings.append(f"pose {index} sol {sol}: Fkin mismatch")
                    continue
            else:
                dp = dr = None

            margin = min(abs(joints[J5]), 180.0 - abs(joints[J5]))
            if margin < WRIST_SINGULARITY_MARGIN_DEG:
                warnings.append(f"pose {index} sol {sol}: |q5|={joints[J5]:.1f}"
                                f" is {margin:.1f} deg from a wrist singularity")

            records.append({"pose_index": index, "sol": sol, "joints": joints,
                            "fkin_dp_mm": dp, "fkin_drot_deg": dr,
                            "wrist_margin_deg": round(margin, 2)})
            print(f"  sol {sol} ({sol:03b}) : q1={joints[J1]:+8.2f} "
                  f"q2={joints[J2]:+8.2f} q3={joints[J3]:+8.2f} "
                  f"q5={joints[J5]:+8.2f}   (wrist margin {margin:5.1f} deg)")
    return records, warnings


# --------------------------------------------------------------------- #
# step 2 - fit the planar arm model
# --------------------------------------------------------------------- #
def closure_residuals(params, records):
    """|| elbow - wrist_centre || - L3 for every solution, in mm."""
    d1, l2, l3, d6 = params
    out = []
    for rec in records:
        wc = wrist_centre(TEST_POSES[rec["pose_index"]], d6)
        elbow = elbow_point(rec["joints"][J1], rec["joints"][J2], d1, l2)
        out.append(float(np.linalg.norm(elbow - wc)) - l3)
    return np.array(out)


def fit_kinematics(records, start=(150.0, 400.0, 360.0, 120.0), iterations=150):
    params = np.array(start, dtype=float)
    for _ in range(iterations):
        residual = closure_residuals(params, records)
        jac = np.zeros((len(residual), 4))
        for k in range(4):
            bumped = params.copy()
            bumped[k] += 1e-6
            jac[:, k] = (closure_residuals(bumped, records) - residual) / 1e-6
        step, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        params = params + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return params, float(np.max(np.abs(closure_residuals(params, records))))


# --------------------------------------------------------------------- #
# step 3 - the three bit tests
# --------------------------------------------------------------------- #
def test_wrist_bit(records):
    """bit0 == 1  <=>  q5 < 0  (RTSL: WF = flip = negative wrist bend)."""
    evidence, agree = [], True
    for rec in records:
        q5 = rec["joints"][J5]
        if abs(q5) < 1e-6:
            continue
        expect = (rec["sol"] & 1) == 1
        got = q5 < 0
        agree &= (expect == got)
        evidence.append({"pose": rec["pose_index"], "sol": rec["sol"],
                         "q5": q5, "bit0": rec["sol"] & 1, "q5_negative": got,
                         "match": expect == got})
    return agree, evidence


def test_elbow_bit(records, params):
    """bit1 == 1  <=>  elbow point lies ABOVE the shoulder->wrist-centre line.

    Frame-invariant 3D test, exactly the RTSL v0.1 geometric definition.
    """
    d1, l2, _, d6 = params
    evidence, agree = [], True
    for rec in records:
        pose = TEST_POSES[rec["pose_index"]]
        shoulder = np.array([0.0, 0.0, d1])
        wc = wrist_centre(pose, d6)
        elbow = elbow_point(rec["joints"][J1], rec["joints"][J2], d1, l2)
        axis = wc - shoulder
        axis /= np.linalg.norm(axis)
        vec = elbow - shoulder
        perp = vec - float(np.dot(vec, axis)) * axis
        if float(np.linalg.norm(perp)) < ELBOW_PERP_MIN_MM:
            continue  # arm essentially straight: no meaningful elbow side
        above = bool(perp[2] > 0.0)
        expect = ((rec["sol"] >> 1) & 1) == 1
        agree &= (expect == above)
        evidence.append({"pose": rec["pose_index"], "sol": rec["sol"],
                         "bit1": (rec["sol"] >> 1) & 1,
                         "elbow_above": above,
                         "perp_z_mm": round(float(perp[2]), 3),
                         "match": expect == above})
    return agree, evidence


def test_shoulder_bit(records, params):
    """bit2 == 1  <=>  q1 points ~180 deg AWAY from the wrist-centre azimuth."""
    _, _, _, d6 = params
    evidence, agree, ambiguous = [], True, 0
    for rec in records:
        wc = wrist_centre(TEST_POSES[rec["pose_index"]], d6)
        azimuth = math.degrees(math.atan2(wc[1], wc[0]))
        delta = wrap180(rec["joints"][J1] - azimuth)
        if abs(delta) <= AZIMUTH_TOLERANCE_DEG:
            away = False
        elif abs(delta) >= 180.0 - AZIMUTH_TOLERANCE_DEG:
            away = True
        else:
            ambiguous += 1
            evidence.append({"pose": rec["pose_index"], "sol": rec["sol"],
                             "dpsi": round(delta, 2), "class": "AMBIGUOUS"})
            continue
        expect = ((rec["sol"] >> 2) & 1) == 1
        agree &= (expect == away)
        evidence.append({"pose": rec["pose_index"], "sol": rec["sol"],
                         "bit2": (rec["sol"] >> 2) & 1,
                         "dpsi": round(delta, 2),
                         "q1_away": away, "match": expect == away})
    return (agree and ambiguous == 0), evidence


def test_shoulder_degeneracy(records, params):
    """Check whether sol s and sol s+4 place the arm at the same points.

    If they do, the arm has no shoulder offset and the RTSL SL/SR definition
    (wrist centre left/right of the base plane) is geometrically meaningless.
    """
    d1, l2, _, _ = params
    by_key = {(r["pose_index"], r["sol"]): r for r in records}
    distances, degenerate = [], True
    for (pose_index, sol), rec in by_key.items():
        if sol >= 4:
            continue
        partner = by_key.get((pose_index, sol + 4))
        if partner is None:
            continue
        a = elbow_point(rec["joints"][J1], rec["joints"][J2], d1, l2)
        b = elbow_point(partner["joints"][J1], partner["joints"][J2], d1, l2)
        dist = float(np.linalg.norm(a - b))
        distances.append({"pose": pose_index, "pair": f"{sol}/{sol + 4}",
                          "elbow_distance_mm": round(dist, 6)})
        degenerate &= dist <= SHOULDER_DEGENERACY_TOL_MM
    return degenerate, distances


# --------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------- #
def emit_mapping():
    """sol = 4*shoulder + 2*elbow + 1*wrist, with the validated bit meanings."""
    mapping = {}
    for shoulder_bit in (0, 1):
        for elbow_bit in (0, 1):
            for wrist_bit in (0, 1):
                sol = 4 * shoulder_bit + 2 * elbow_bit + wrist_bit
                if SL_IS_BIT2_CLEAR:
                    shoulder = "SR" if shoulder_bit else "SL"
                else:
                    shoulder = "SL" if shoulder_bit else "SR"
                mapping[(shoulder, "EU" if elbow_bit else "ED",
                         "WF" if wrist_bit else "WN")] = sol
    return mapping


def print_mapping(mapping):
    print("\n" + "=" * 70)
    print("Paste into the executor:\n")
    print(f"# {ROBOT_MODEL}, controller {CONTROLLER_VERSION}, "
          f"validated by doosan_sol_calibration.py")
    print("# sol = 4*S + 2*E + W   with S: SL=0 SR=1 | E: ED=0 EU=1 | "
          "W: WN=0 WF=1")
    print("CANONICAL_CONFIG_TO_DOOSAN_SOL = {")
    for key in sorted(mapping, key=mapping.get):
        print(f'    ("{key[0]}", "{key[1]}", "{key[2]}"): {mapping[key]},')
    print("}")
    print("=" * 70)


def main():
    rclpy.init()
    node = rclpy.create_node("doosan_sol_calibration")
    report = {"robot_model": ROBOT_MODEL,
              "controller_version": CONTROLLER_VERSION,
              "test_poses": TEST_POSES}
    try:
        ik_client = node.create_client(Ikin, IKIN_SERVICE)
        pos_client = node.create_client(GetCurrentPosj, POSITION_SERVICE)
        wait(ik_client, IKIN_SERVICE)
        wait(pos_client, POSITION_SERVICE, required=False)

        fk_client = None
        if HAVE_FKIN:
            candidate = node.create_client(Fkin, FKIN_SERVICE)
            if wait(candidate, FKIN_SERVICE, required=False):
                fk_client = candidate
        if fk_client is None:
            print("  [warn] Fkin unavailable: Ikin results cannot be "
                  "cross-checked (Ikin.srv has no status field).")

        records, warnings = probe(node, ik_client, fk_client)
        report["solutions"] = records
        report["warnings"] = warnings
        if len(records) < 8:
            raise RuntimeError("Too few valid solutions to calibrate.")

        params, residual = fit_kinematics(records)
        d1, l2, l3, d6 = (round(float(v), 4) for v in params)
        report["kinematics"] = {"d1_mm": d1, "L2_mm": l2, "L3_mm": l3,
                                "d6_mm": d6, "max_residual_mm": residual}
        print("\n=== fitted planar arm model ===")
        print(f"  d1={d1}  L2={l2}  L3={l3}  d6={d6}   "
              f"max closure residual = {residual:.6f} mm")
        model_ok = residual <= KINEMATIC_FIT_TOL_MM
        print(f"  model {'ACCEPTED' if model_ok else 'REJECTED'} "
              f"(tolerance {KINEMATIC_FIT_TOL_MM} mm)")

        wrist_ok, wrist_ev = test_wrist_bit(records)
        elbow_ok, elbow_ev = test_elbow_bit(records, params)
        shoulder_ok, shoulder_ev = test_shoulder_bit(records, params)
        degenerate, degeneracy_ev = test_shoulder_degeneracy(records, params)
        report["tests"] = {
            "wrist": {"pass": wrist_ok, "evidence": wrist_ev},
            "elbow": {"pass": elbow_ok, "evidence": elbow_ev},
            "shoulder": {"pass": shoulder_ok, "evidence": shoulder_ev},
            "shoulder_degenerate": {"degenerate": degenerate,
                                    "evidence": degeneracy_ev},
        }

        def verdict(flag, matched):
            total = len(matched)
            hits = sum(1 for e in matched if e.get("match"))
            return f"{'PASS' if flag else 'FAIL'}  ({hits}/{total} agree)"

        print("\n=== bit tests ===")
        print(f"  bit0 == (q5 < 0)                  : "
              f"{verdict(wrist_ok, wrist_ev)}")
        print(f"  bit1 == elbow above S->WC line    : "
              f"{verdict(elbow_ok, elbow_ev)}")
        print(f"  bit2 == q1 away from WC azimuth   : "
              f"{verdict(shoulder_ok, shoulder_ev)}")

        if degenerate:
            print("\n  [note] sol s and sol s+4 place shoulder, elbow and "
                  "wrist centre at identical\n"
                  "         points (max difference "
                  f"{max(e['elbow_distance_mm'] for e in degeneracy_ev):.6f}"
                  " mm). This arm has no shoulder\n"
                  "         offset, so the RTSL v0.1 SL/SR definition is "
                  "degenerate. SL <-> Doosan\n"
                  "         'Lefty' is a LABEL CONVENTION here, not a "
                  "measurement. Record this.")

        if warnings:
            print("\n=== warnings ===")
            for item in warnings:
                print(f"  - {item}")

        if model_ok and wrist_ok and elbow_ok and shoulder_ok:
            mapping = emit_mapping()
            report["mapping"] = {"|".join(k): v for k, v in mapping.items()}
            print_mapping(mapping)
        else:
            print("\nOne or more tests failed: no mapping emitted.")
            print("Check the fit residual first; if it is large the planar "
                  "model does not describe this robot.")

        REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nEvidence written to {REPORT_FILE}")
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
