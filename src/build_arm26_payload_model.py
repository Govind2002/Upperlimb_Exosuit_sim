#!/usr/bin/env python3
"""
Derive an arm26 + payload (+ elbow torque input) model from the ORIGINAL arm26.osim.

Runs WITHOUT OpenSim (pure text insertion + XML sanity checks), so every change is
visible and auditable.  The original file is never modified.

Provenance tags used in the generated XML comments
  [USER]    value given by the user (2.5 kg payload)
  [PAPER]   value/assumption stated in the paper
  [ASSUMED] placeholder that is NOT in the paper or figure -> replace with a measurement
  [VISUAL]  display-only, no inertial meaning

Usage
  python build_arm26_payload_model.py arm26.osim arm26_elbow_payload_2p5kg.osim
"""
import sys
import xml.etree.ElementTree as ET

# ----------------------------------------------------------------------------------
# PARAMETERS  (edit here; everything else is derived)
# ----------------------------------------------------------------------------------
PAYLOAD_MASS_KG = 2.5                       # [USER] additional test condition (paper: 0, 3, 5, 10 kg)

# Payload centre of mass, expressed in the r_ulna_radius_hand body frame (origin = elbow joint centre).
# [PAPER]+[ASSUMED]: the paper's Eq. (1) places the load at distance (l_e + l_w) from the elbow:
#                    l_e = 0.28 m, l_w = 0.04 m (Table II)  ->  0.32 m.
# Mapping that distance to the -y axis of the arm26 forearm frame, with zero x/z offset,
# is MY ASSUMPTION (the long axis of the arm26 forearm frame is -y).  Replace with a measurement.
PAYLOAD_COM_IN_FOREARM = (0.0, -(0.28 + 0.04), 0.0)

# Inertia about the payload's own COM [Ixx Iyy Izz Ixy Ixz Iyz], kg m^2.
# [ASSUMED] point mass (zero), mirroring the lumped M*(l_e+l_w) term of Eq. (1).
# Replace once the real object's shape/dimensions are known (see formulas in the README text).
PAYLOAD_INERTIA = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# [PAPER] footnote 1 of Sec. IV: "Only the elbow degree of freedom is active here."
LOCK_SHOULDER_ELEVATION = True              # locks r_shoulder_elev at its default value (0 rad)

# Idealised elbow torque input (stand-in for tau_a of Eq. (5)/(6)); control value = torque in N m.
ADD_EXO_ELBOW_TORQUE_ACTUATOR = True
EXO_ACTUATOR_NAME = "exo_elbow_torque"

NEW_MODEL_NAME = "arm26_elbow_payload"
VISUAL_SPHERE_RADIUS_M = 0.03               # [VISUAL] display only

NL = "\r\n"   # the original file uses CRLF; keep it so a diff shows only real changes
T = "\t"


def fmt(x):
    return repr(float(x)) if x != 0 else "0"


def vec(v):
    return " ".join(fmt(x) for x in v)


def indent(block, n):
    return NL.join((T * n + ln) if ln else ln for ln in block.split("\n")) + NL


def payload_body_xml():
    return f"""<Body name="payload">
{T}<!--[USER] 2.5 kg additional payload. Paper conditions are 0, 3, 5, 10 kg; 2.5 kg is NOT a paper condition.-->
{T}<!--The geometry used to display the axes of this Frame.-->
{T}<FrameGeometry name="frame_geometry">
{T}{T}<socket_frame>..</socket_frame>
{T}{T}<scale_factors>0.2 0.2 0.2</scale_factors>
{T}</FrameGeometry>
{T}<attached_geometry>
{T}{T}<!--[VISUAL] display-only marker of the payload COM. Radius is arbitrary and has no inertial meaning.-->
{T}{T}<Sphere name="payload_visual">
{T}{T}{T}<socket_frame>..</socket_frame>
{T}{T}{T}<scale_factors>1 1 1</scale_factors>
{T}{T}{T}<Appearance>
{T}{T}{T}{T}<opacity>1</opacity>
{T}{T}{T}{T}<color>0.1 0.1 0.1</color>
{T}{T}{T}</Appearance>
{T}{T}{T}<radius>{fmt(VISUAL_SPHERE_RADIUS_M)}</radius>
{T}{T}</Sphere>
{T}</attached_geometry>
{T}<!--[USER] payload mass (kg)-->
{T}<mass>{fmt(PAYLOAD_MASS_KG)}</mass>
{T}<!--COM is the body origin; the body origin is placed at the payload COM by the weld joint offset.-->
{T}<mass_center>0 0 0</mass_center>
{T}<!--[ASSUMED] point-mass inertia about the COM. Replace with measured/derived values.-->
{T}<inertia>{vec(PAYLOAD_INERTIA)}</inertia>
</Body>"""


def payload_joint_xml():
    pf = lambda name, parent, trans: f"""<PhysicalOffsetFrame name="{name}">
{T}<FrameGeometry name="frame_geometry">
{T}{T}<socket_frame>..</socket_frame>
{T}{T}<scale_factors>0.2 0.2 0.2</scale_factors>
{T}</FrameGeometry>
{T}<socket_parent>{parent}</socket_parent>
{T}<translation>{vec(trans)}</translation>
{T}<orientation>0 0 0</orientation>
</PhysicalOffsetFrame>"""
    frames = (pf("r_ulna_radius_hand_payload_offset", "/bodyset/r_ulna_radius_hand", PAYLOAD_COM_IN_FOREARM)
              + "\n" + pf("payload_offset", "/bodyset/payload", (0, 0, 0)))
    frames = NL.join((T * 2 + ln) if ln else ln for ln in frames.split("\n"))
    return f"""<WeldJoint name="payload_weld">
{T}<!--[ASSUMED] rigid attachment: payload COM fixed in the forearm/hand frame at the translation below (see PAYLOAD_COM_IN_FOREARM).-->
{T}<socket_parent_frame>r_ulna_radius_hand_payload_offset</socket_parent_frame>
{T}<socket_child_frame>payload_offset</socket_child_frame>
{T}<frames>
{frames}
{T}</frames>
</WeldJoint>"""


def exo_actuator_xml():
    return f"""<CoordinateActuator name="{EXO_ACTUATOR_NAME}">
{T}<!--Idealised stand-in for the assistive elbow torque tau_a (Eq. 5-6). Control value = torque in N m about r_elbow_flex (+ = flexion).
{T}    The cable/sheath/gear model (Eq. 2-7) is NOT in this file; it lives outside OpenSim.
{T}    No torque limits are given in the paper, so none are set here.-->
{T}<optimal_force>1</optimal_force>
{T}<coordinate>r_elbow_flex</coordinate>
</CoordinateActuator>"""


def insert_before_close(text, set_close_tag, block, depth):
    """Insert `block` as the last child of the <objects> list that directly precedes `set_close_tag`."""
    end = text.index(set_close_tag)
    obj_close = text.rindex(T * 3 + "</objects>", 0, end)
    return text[:obj_close] + indent(block, depth) + text[obj_close:]


def build(src, dst):
    with open(src, "rb") as f:
        raw = f.read().decode("utf-8")
    assert "\r\n" in raw, "expected CRLF original"
    text = raw

    # 1. model name + change log
    assert '<Model name="arm26">' in text
    log = (f'<Model name="{NEW_MODEL_NAME}">{NL}{T*2}<!--MODIFIED from arm26.osim: (1) body "payload" + WeldJoint "payload_weld" '
           f'(mass [USER], COM/inertia [ASSUMED]); (2) r_shoulder_elev locked at default 0 rad [PAPER footnote]; '
           f'(3) CoordinateActuator "{EXO_ACTUATOR_NAME}" on r_elbow_flex. Nothing else changed.-->')
    text = text.replace('<Model name="arm26">', log, 1)

    # 2. payload body (end of BodySet), weld joint (end of JointSet), actuator (end of ForceSet)
    text = insert_before_close(text, T * 2 + "</BodySet>", payload_body_xml(), 4)
    text = insert_before_close(text, T * 2 + "</JointSet>", payload_joint_xml(), 4)
    if ADD_EXO_ELBOW_TORQUE_ACTUATOR:
        text = insert_before_close(text, T * 2 + "</ForceSet>", exo_actuator_xml(), 4)

    # 3. lock shoulder elevation
    if LOCK_SHOULDER_ELEVATION:
        i = text.index('<Coordinate name="r_shoulder_elev">')
        j = text.index("<locked>false</locked>", i)
        assert j - i < 2500, "unexpected layout"
        text = text[:j] + "<locked>true</locked>" + text[j + len("<locked>false</locked>"):]

    with open(dst, "wb") as f:
        f.write(text.encode("utf-8"))
    return dst


def check(path):
    """Structural checks that do not need OpenSim."""
    root = ET.parse(path).getroot()
    m = root.find("Model")
    bodies = {b.get("name"): b for b in m.find("BodySet/objects")}
    joints = {j.get("name"): j for j in m.find("JointSet/objects")}
    forces = {f.get("name"): f for f in m.find("ForceSet/objects")}
    assert len(bodies) == len(list(m.find("BodySet/objects"))), "duplicate body names"
    # every joint's sockets must resolve to a frame that the joint owns
    child_bodies = []
    for jn, j in joints.items():
        owned = {pf.get("name"): pf.findtext("socket_parent") for pf in j.find("frames")}
        for sock in ("socket_parent_frame", "socket_child_frame"):
            assert j.findtext(sock) in owned, f"{jn}: {sock} -> {j.findtext(sock)} not owned"
        for pf, par in owned.items():
            assert par == "/ground" or par.split("/")[-1] in bodies, f"{jn}/{pf}: parent {par} missing"
        child_bodies.append(owned[j.findtext("socket_child_frame")].split("/")[-1])
    # tree: each non-ground body is the child of exactly one joint
    assert sorted(child_bodies) == sorted(bodies), (sorted(child_bodies), sorted(bodies))
    coords = {c.get("name"): c for j in joints.values() for c in j.iter("Coordinate")}
    for fn, f in forces.items():
        if f.tag == "CoordinateActuator":
            assert f.findtext("coordinate") in coords
    total = sum(float(b.findtext("mass")) for b in bodies.values())
    return dict(bodies=list(bodies), joints=list(joints), forces=list(forces), coords=list(coords), total_mass=total)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "arm26.osim"
    dst = sys.argv[2] if len(sys.argv) > 2 else "arm26_elbow_payload_2p5kg.osim"
    build(src, dst)
    info = check(dst)
    print("wrote", dst)
    for k, v in info.items():
        print(f"  {k}: {v}")
