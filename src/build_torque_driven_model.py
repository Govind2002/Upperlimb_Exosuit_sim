#!/usr/bin/env python3
"""
Derive the TORQUE-DRIVEN forward-dynamics model (Architecture 1) from arm26_elbow_payload_2p5kg.osim.

Changes, all text-level so everything else stays byte-identical:
  1. removes the six Thelen2003Muscle elements (the plant becomes the paper's 1-DOF assumption,
     with OpenSim's real geometry and mass properties; no passive muscle forces, no joint damping)
  2. adds CoordinateActuator "human_elbow_torque"  (the human's net elbow torque, N m, + = flexion)
  3. keeps CoordinateActuator "exo_elbow_torque"   (the exosuit's elbow torque, N m, + = flexion)
  4. renames the model

Needs no OpenSim.   python build_torque_driven_model.py arm26_elbow_payload_2p5kg.osim arm26_torque_driven_2p5kg.osim
"""
import re
import sys
import xml.etree.ElementTree as ET

NL, T = "\r\n", "\t"
NEW_NAME = "arm26_torque_driven"


def human_actuator_xml():
    return (f"{T*4}<CoordinateActuator name=\"human_elbow_torque\">{NL}"
            f"{T*5}<!--The human's net elbow torque tau_h (N m, + = flexion). Control value = torque.{NL}"
            f"{T*5}    Stands in for the six removed elbow muscles. No torque limits are given in the paper, so none are set.-->{NL}"
            f"{T*5}<optimal_force>1</optimal_force>{NL}"
            f"{T*5}<coordinate>r_elbow_flex</coordinate>{NL}"
            f"{T*4}</CoordinateActuator>{NL}")


def build(src, dst):
    with open(src, "rb") as f:
        text = f.read().decode("utf-8")
    assert "\r\n" in text, "expected CRLF input"

    # 1. remove muscles
    pat = re.compile(r"[ \t]*<Thelen2003Muscle name=\"[^\"]+\">.*?</Thelen2003Muscle>\r\n", re.S)
    text, n = pat.subn("", text)
    assert n == 6, f"expected 6 muscles, removed {n}"

    # 2. add human actuator at the end of the ForceSet objects list
    end = text.index(T * 2 + "</ForceSet>")
    obj_close = text.rindex(T * 3 + "</objects>", 0, end)
    text = text[:obj_close] + human_actuator_xml() + text[obj_close:]

    # 3. rename
    m = re.search(r"<Model name=\"[^\"]+\">", text)
    text = text.replace(m.group(0),
                        f"<Model name=\"{NEW_NAME}\">{NL}{T*2}<!--MODIFIED from arm26_elbow_payload_2p5kg.osim: six muscles removed; "
                        f"CoordinateActuator human_elbow_torque added. Torque-driven plant for Architecture 1.-->", 1)
    with open(dst, "wb") as f:
        f.write(text.encode("utf-8"))


def check(path):
    m = ET.parse(path).getroot().find("Model")
    forces = [(f.tag, f.get("name")) for f in m.find("ForceSet/objects")]
    bodies = {b.get("name"): float(b.findtext("mass")) for b in m.find("BodySet/objects")}
    coords = {c.get("name"): c.findtext("locked") for j in m.find("JointSet/objects") for c in j.iter("Coordinate")}
    assert [t for t, _ in forces] == ["CoordinateActuator", "CoordinateActuator"], forces
    assert {n for _, n in forces} == {"exo_elbow_torque", "human_elbow_torque"}
    assert coords["r_shoulder_elev"] == "true"
    return dict(model=m.get("name"), forces=forces, total_mass=sum(bodies.values()), shoulder_locked=coords["r_shoulder_elev"])


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "arm26_elbow_payload_2p5kg.osim"
    dst = sys.argv[2] if len(sys.argv) > 2 else "arm26_torque_driven_2p5kg.osim"
    build(src, dst)
    print("wrote", dst)
    for k, v in check(dst).items():
        print(f"  {k}: {v}")
