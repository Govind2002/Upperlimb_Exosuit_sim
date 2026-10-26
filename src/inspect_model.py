#!/usr/bin/env python3
"""
Load an OpenSim model, initialise it and report bodies, joints, coordinates, actuators and forces.

Usage
  python inspect_model.py arm26_elbow_payload_2p5kg.osim
  python inspect_model.py arm26_elbow_payload_2p5kg.osim --expect-payload-kg 2.5 --check-gravity
  python inspect_model.py arm26_elbow_payload_2p5kg.osim --geometry-dir /path/to/Geometry
  python inspect_model.py arm26_elbow_payload_2p5kg.osim --visualize

Written against the OpenSim 4.x Python API (Version="40000" model files load in 4.0 and later).
I could not execute it against OpenSim in the environment where it was written; the calls marked
"API-sensitive" are wrapped in try/except so a version difference degrades to a warning, not a crash.
"""
import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import opensim as osim


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def v3(v):
    return tuple(v.get(i) for i in range(3))


def v6(v):
    return tuple(v.get(i) for i in range(6))


def warn(msg):
    print(f"  [WARN] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")
    return False


def ok(msg):
    print(f"  [ OK ] {msg}")
    return True


# --------------------------------------------------------------------------------------
# geometry dependency check (pure XML, no OpenSim needed)
# --------------------------------------------------------------------------------------
def check_geometry_files(osim_path, extra_dirs):
    root = ET.parse(osim_path).getroot()
    files = sorted({e.text.strip() for e in root.iter("mesh_file") if e.text})
    base = os.path.dirname(os.path.abspath(osim_path))
    search = [base, os.path.join(base, "Geometry")] + [d for d in extra_dirs if d]
    print(f"\n== External geometry referenced by the file: {len(files)} mesh files ==")
    missing = []
    for f in files:
        hit = next((d for d in search if os.path.isfile(os.path.join(d, f))), None)
        if hit is None:
            missing.append(f)
    if missing:
        warn(f"{len(missing)}/{len(files)} .vtp files not found in {search}")
        print("        (bones will not render; kinematics/dynamics are unaffected)")
        print("        e.g. " + ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else ""))
    else:
        ok("all mesh files found")
    return search, missing


# --------------------------------------------------------------------------------------
# independent analytic reference (from the XML only)
# --------------------------------------------------------------------------------------
def analytic_hold_moment(osim_path, theta, payload_name="payload"):
    """Static flexion moment [N m] the elbow must supply to hold the arm at shoulder_elev = 0, elbow = theta.
    Uses only the XML mass properties + the elbow rotation axis; gravity from the file."""
    m = ET.parse(osim_path).getroot().find("Model")
    g = np.array([float(x) for x in m.findtext("gravity").split()])
    bodies = {b.get("name"): b for b in m.find("BodySet/objects")}
    joints = {j.get("name"): j for j in m.find("JointSet/objects")}
    axis = np.array([float(x) for x in
                     joints["r_elbow"].find("SpatialTransform/TransformAxis[@name='rotation1']/axis").text.split()])
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
    items = [(float(bodies["r_ulna_radius_hand"].findtext("mass")),
              np.array([float(x) for x in bodies["r_ulna_radius_hand"].findtext("mass_center").split()]))]
    if payload_name in bodies:
        wj = joints["payload_weld"]
        off = next(pf for pf in wj.find("frames") if pf.get("name") == wj.findtext("socket_parent_frame"))
        p = np.array([float(x) for x in off.findtext("translation").split()])
        c = np.array([float(x) for x in bodies[payload_name].findtext("mass_center").split()])
        items.append((float(bodies[payload_name].findtext("mass")), p + c))
    tau_gravity = sum(axis @ np.cross(R @ p, mass * g) for mass, p in items)
    return -tau_gravity


# --------------------------------------------------------------------------------------
# main report
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--expect-payload-kg", type=float, default=None)
    ap.add_argument("--geometry-dir", default=None)
    ap.add_argument("--check-gravity", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    print(f"OpenSim {osim.GetVersion()}   |   model file: {args.model}")
    search, _ = check_geometry_files(args.model, [args.geometry_dir])
    for d in search:
        if os.path.isdir(d):
            osim.ModelVisualizer.addDirToGeometrySearchPaths(d)

    model = osim.Model(args.model)
    if args.visualize:
        model.setUseVisualizer(True)
    state = model.initSystem()
    try:
        model.equilibrateMuscles(state)          # Thelen2003 muscles need a consistent initial fibre state
    except Exception as e:                        # API-sensitive
        warn(f"equilibrateMuscles failed: {e}")

    print(f"\nModel name: {model.getName()}   gravity: {v3(model.getGravity())}")
    print(f"NQ={state.getNQ()}  NU={state.getNU()}  NZ={state.getNZ()}  controls={model.getNumControls()}")
    print(f"Total mass: {model.getTotalMass(state):.4f} kg")

    print("\n== Bodies ==")
    for i in range(model.getBodySet().getSize()):
        b = model.getBodySet().get(i)
        print(f"  {b.getName():<20} m={b.getMass():8.4f} kg  COM(body frame)={tuple(round(x, 5) for x in v3(b.getMassCenter()))}"
              f"  I[xx yy zz xy xz yz]={tuple(round(x, 6) for x in v6(b.get_inertia()))}")

    print("\n== Joints ==")
    for i in range(model.getJointSet().getSize()):
        j = model.getJointSet().get(i)
        pb = j.getParentFrame().findBaseFrame().getName()
        cb = j.getChildFrame().findBaseFrame().getName()
        print(f"  {j.getName():<14} {j.getConcreteClassName():<12} {pb} -> {cb}   dof={j.numCoordinates()}")

    print("\n== Coordinates ==")
    for i in range(model.getCoordinateSet().getSize()):
        c = model.getCoordinateSet().get(i)
        print(f"  {c.getName():<16} joint={c.getJoint().getName():<12} value={c.getValue(state):+.4f} rad "
              f"range=[{c.getRangeMin():+.4f}, {c.getRangeMax():+.4f}] "
              f"locked={c.getLocked(state)} clamped={c.getClamped(state)}")

    print("\n== Actuators (things that take a control input) ==")
    acts = model.getActuators()
    for i in range(acts.getSize()):
        a = acts.get(i)
        extra = ""
        ca = osim.CoordinateActuator.safeDownCast(a)
        if ca is not None:
            extra = f"coordinate={ca.getCoordinate().getName()} optimal_force={ca.getOptimalForce()} (control = N m)"
        mu = osim.Muscle.safeDownCast(a)
        if mu is not None:
            extra = (f"Fmax={mu.getMaxIsometricForce():.1f} N  lopt={mu.getOptimalFiberLength():.4f} m  "
                     f"lts={mu.getTendonSlackLength():.4f} m")
        print(f"  {a.getName():<18} {a.getConcreteClassName():<18} {extra}")

    print("\n== All forces in ForceSet ==")
    fs = model.getForceSet()
    for i in range(fs.getSize()):
        f = fs.get(i)
        print(f"  {f.getName():<18} {f.getConcreteClassName():<18} appliesForce={f.appliesForce(state)}")

    # ---------------- sanity checks ----------------
    print("\n== Checks ==")
    allgood = True
    names = [model.getBodySet().get(i).getName() for i in range(model.getBodySet().getSize())]
    if args.expect_payload_kg is not None:
        if "payload" not in names:
            allgood &= fail("no body named 'payload'")
        else:
            mp = model.getBodySet().get("payload").getMass()
            allgood &= ok(f"payload mass {mp} kg") if abs(mp - args.expect_payload_kg) < 1e-9 else fail(
                f"payload mass {mp} != {args.expect_payload_kg}")
            expected_total = 1.864572 + 1.534315 + args.expect_payload_kg
            tm = model.getTotalMass(state)
            allgood &= ok(f"total mass {tm:.6f} kg (expected {expected_total:.6f})") if abs(tm - expected_total) < 1e-5 else fail(
                f"total mass {tm:.6f} != {expected_total:.6f}")
    sh = model.getCoordinateSet().get("r_shoulder_elev")
    el = model.getCoordinateSet().get("r_elbow_flex")
    allgood &= ok("shoulder locked") if sh.getLocked(state) else warn("shoulder_elev is NOT locked (paper: only elbow active)") or True
    if abs(el.getRangeMin()) > 1e-9:
        warn("elbow range does not start at 0 rad")
    print("  note: the arm26 elbow range is 0..130 deg (clamped=false); a reference that goes below 0 rad leaves the model's range.")

    if args.check_gravity:
        print("\n== Gravity-only static elbow moment: OpenSim inverse dynamics vs. analytic (from XML) ==")
        try:
            # switch off every force (muscles + exo actuator) so only gravity remains
            for i in range(model.getForceSet().getSize()):
                model.updForceSet().get(i).setAppliesForce(state, False)
            solver = osim.InverseDynamicsSolver(model)
            iel = el.getMobilizerQIndex()
            for deg in (0, 30, 60, 90, 120):
                el.setValue(state, math.radians(deg))
                model.realizeVelocity(state)
                tau = solver.solve(state, osim.Vector(state.getNU(), 0.0))
                a = analytic_hold_moment(args.model, math.radians(deg))
                d = tau.get(iel) - a
                tag = "OK" if abs(d) < 0.05 else "MISMATCH"
                print(f"  elbow={deg:4d} deg   ID={tau.get(iel):8.3f} N m   analytic={a:8.3f} N m   diff={d:+.3f}  {tag}")
        except Exception as e:                    # API-sensitive
            warn(f"gravity check could not run on this OpenSim version: {e}")

    if args.visualize:
        try:                                      # API-sensitive
            import time
            viz = model.updVisualizer().updSimbodyVisualizer()
            viz.setBackgroundType(viz.GroundAndSky)
            for deg in list(range(0, 131, 2)) + list(range(130, -1, -2)):
                el.setValue(state, math.radians(deg))
                viz.drawFrameNow(state)
                time.sleep(0.03)
        except Exception as e:
            warn(f"visualizer unavailable ({e}); open the .osim in the OpenSim GUI instead")

    print("\nRESULT:", "all checks passed" if allgood else "see FAIL lines above")
    return 0 if allgood else 1


if __name__ == "__main__":
    sys.exit(main())
