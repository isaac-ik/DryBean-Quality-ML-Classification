"""
Builds an animated Blender scene of the bean-sorting conveyor.

Imports conveyor_assembly.glb (one object per printed part, pivots on the
axles / servo shafts / link hinges) and animates:
  - the belt (seamless loop: every link advances one pitch, then the cycle repeats)
  - both drums turning in sync with the belt
  - beans dropping onto the belt, passing the camera, and being diverted by the
    gate paddles down the chutes into bins (or riding off the end = "good")

Usage
  From Blender:   Scripting tab -> Open -> animate_conveyor.py -> Run Script
  Command line:   blender --python animate_conveyor.py
                  blender -b --python animate_conveyor.py -- --save conveyor_animated.blend
                  blender -b --python animate_conveyor.py -- --still frame.png 200

Tweak the SETTINGS block below and re-run; the script rebuilds the scene from scratch.
"""
import json
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Vector

# ---------------------------------------------------------------- SETTINGS
FPS = 30
FRAMES_PER_PITCH = 12            # belt moves one 12 mm link per 12 frames -> 30 mm/s (real belt ~20 mm/s)
SPAWN_EVERY_S = 3.6              # >= 105 mm between beans, like the real machine
FIRST_SPAWN_S = 0.5
LOAD_X = 38.0                    # where beans land on the belt (mm)
SEED = 7
# (label, gate) in the order they are dropped on the belt. gate 0 = rides off the end.
BEANS = [("good", 0), ("black", 1), ("pale", 2), ("foreign", 3), ("good", 0),
         ("spotted", 1), ("good", 0), ("pale", 2), ("black", 1), ("good", 0)]
BEAN_COLOURS = {"good": (0.45, 0.27, 0.13), "black": (0.10, 0.09, 0.08),
                "spotted": (0.55, 0.38, 0.20), "pale": (0.85, 0.74, 0.55), "foreign": (0.62, 0.62, 0.58)}
# ------------------------------------------------------------------------


def find_here():
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    for t in bpy.data.texts:
        if t.filepath:
            cands.append(os.path.dirname(bpy.path.abspath(t.filepath)))
    if bpy.data.filepath:
        cands.append(os.path.dirname(bpy.data.filepath))
    for c in cands:
        if os.path.exists(os.path.join(c, "conveyor_assembly.glb")):
            return c
    raise RuntimeError("Put animate_conveyor.py next to conveyor_assembly.glb and conveyor_params.json")


HERE = find_here()
with open(os.path.join(HERE, "conveyor_params.json"), encoding="utf-8") as f:
    PR = json.load(f)

MM = 0.001
V = PR["pitch"] * FPS / FRAMES_PER_PITCH          # belt speed, mm/s
G = 9810.0                                         # mm/s^2
random.seed(SEED)


def frame_of(t):
    return 1 + t * FPS


def mm(x, y, z):
    return Vector((x * MM, y * MM, z * MM))


# ------------------------------------------------------------ animation helpers
def fcurves_of(obj):
    act = obj.animation_data.action
    try:
        return list(act.fcurves)                   # Blender <= 4.x
    except AttributeError:                         # layered actions (Blender 5)
        slot = obj.animation_data.action_slot
        return list(act.layers[0].strips[0].channelbag(slot).fcurves)


def finish_curves(obj, interp="LINEAR", cyclic=False):
    for fc in fcurves_of(obj):
        for kp in fc.keyframe_points:
            if fc.data_path.startswith("hide"):
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = interp
        if cyclic:
            fc.modifiers.new("CYCLES")


# ------------------------------------------------------------------ scene setup
def reset_scene():
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.actions, bpy.data.worlds):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)
    sc.render.fps = FPS
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.length_unit = "MILLIMETERS"
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    return sc


def import_model():
    bpy.ops.import_scene.gltf(filepath=os.path.join(HERE, "conveyor_assembly.glb"))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in bpy.context.scene.objects:
        o.rotation_mode = "XYZ"
    # crisp edges on the printed parts, smooth curved surfaces
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    except Exception:
        try:
            bpy.ops.object.shade_auto_smooth(angle=math.radians(35))
        except Exception:
            pass
    bpy.ops.object.select_all(action="DESELECT")
    root = bpy.data.objects.new("Conveyor", None)
    bpy.context.scene.collection.objects.link(root)
    for o in bpy.context.scene.objects:
        if o.parent is None and o is not root:
            o.parent = root
    return {o.name: o for o in bpy.context.scene.objects}


def material(name, rgb, rough=0.5, alpha=1.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*(c ** 2.2 for c in rgb), 1)   # sRGB -> linear
    bsdf.inputs["Roughness"].default_value = rough
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        try:
            m.surface_render_method = "BLENDED"
        except AttributeError:
            m.blend_method = "BLEND"
    return m


def new_mesh_object(name, build, mat, coll):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    build(bm)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    coll.objects.link(ob)
    return ob


def make_bin(name, x, y, r, h, coll, mat):
    def build(bm):
        ret = bmesh.ops.create_cone(bm, cap_ends=False, segments=48, radius1=r * MM,
                                    radius2=(r + 3) * MM, depth=h * MM)
        bmesh.ops.translate(bm, verts=ret["verts"], vec=(0, 0, h * MM / 2))
        bottom = bmesh.ops.create_circle(bm, cap_ends=True, segments=48, radius=r * MM)
        bmesh.ops.translate(bm, verts=bottom["verts"], vec=(0, 0, 0.0005))
    ob = new_mesh_object(name, build, mat, coll)
    ob.location = mm(x, y, 0)
    mod = ob.modifiers.new("thickness", "SOLIDIFY")
    mod.thickness = 1.5 * MM
    return ob


def stage(sc):
    coll = bpy.data.collections.new("Stage")
    sc.collection.children.link(coll)
    floor = new_mesh_object("Table", lambda bm: bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.2),
                            material("Table", (0.78, 0.74, 0.68), 0.7), coll)
    floor.location = (0.17, 0.05, -0.0001)
    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.75, 0.78, 0.82, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.25
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass

    def light(name, kind, loc, energy, size=0.3, rot=None):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        if kind == "AREA":
            ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        if rot:
            lo.rotation_euler = rot
        coll.objects.link(lo)
        return lo

    light("Key", "AREA", (0.45, -0.45, 0.6), 30, 0.5, (math.radians(45), 0, math.radians(40)))
    light("Fill", "AREA", (-0.2, 0.5, 0.4), 30, 0.6, (math.radians(-55), 0, math.radians(200)))
    light("Sun", "SUN", (0, 0, 1), 1.2, rot=(math.radians(30), math.radians(10), math.radians(-30)))
    led = light("Camera_LED", "AREA", (PR["camera_x"] * MM, 0, 0.145), 3, 0.03)
    led.data.color = (1.0, 0.97, 0.9)

    target = bpy.data.objects.new("Camera_Target", None)
    target.location = (0.19, 0.03, 0.04)
    coll.objects.link(target)
    cam = bpy.data.objects.new("Render_Camera", bpy.data.cameras.new("Render_Camera"))
    cam.location = (0.50, -0.40, 0.52)
    cam.data.lens = 42
    cam.data.clip_start = 0.005
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    coll.objects.link(cam)
    sc.camera = cam

    # second camera close on the gates (select it and press Ctrl+Numpad0 to look through it)
    t2 = bpy.data.objects.new("Closeup_Target", None)
    t2.location = (0.125, 0.012, 0.058)
    coll.objects.link(t2)
    cam2 = bpy.data.objects.new("Closeup_Camera", bpy.data.cameras.new("Closeup_Camera"))
    cam2.location = (0.085, 0.20, 0.215)
    cam2.data.lens = 40
    cam2.data.clip_start = 0.005
    c2 = cam2.constraints.new("TRACK_TO")
    c2.target = t2
    c2.track_axis, c2.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    coll.objects.link(cam2)

    bin_mat = material("Bin", (0.95, 0.95, 0.93), 0.35)
    bins = {}
    for g, xp in enumerate(PR["gates_x"], start=1):
        bins[g] = make_bin(f"Bin_Gate_{g}", xp + 50, 97, 17, 22, coll, bin_mat)
    bins[0] = make_bin("Bin_Good", PR["length"] + 22, 0, 18, 24, coll, bin_mat)
    return coll


# ------------------------------------------------------------------- belt
def hinge_at(u):
    pts = PR["hinges_xz"]
    n = len(pts)
    i = math.floor(u)
    f = u - i
    a, b = pts[i % n], pts[(i + 1) % n]
    return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f


def animate_belt(objs):
    n = PR["n_links"]
    for i in range(n):
        ob = objs[f"Belt_Link_{i:02d}"]
        prev = None
        for k in range(FRAMES_PER_PITCH + 1):
            u = i + k / FRAMES_PER_PITCH
            ax, az = hinge_at(u)
            bx, bz = hinge_at(u + 1)
            phi = math.atan2(-(bz - az), bx - ax)
            if prev is not None:                       # keep the angle continuous
                while phi - prev > math.pi:
                    phi -= 2 * math.pi
                while phi - prev < -math.pi:
                    phi += 2 * math.pi
            prev = phi
            ob.location = mm(ax, 0, az)
            ob.rotation_euler = (0, phi, 0)
            ob.keyframe_insert("location", frame=1 + k)
            ob.keyframe_insert("rotation_euler", frame=1 + k)
        finish_curves(ob, cyclic=True)
    step = math.radians(360 / PR["n_teeth"])
    for name in ("Drum_Drive", "Drum_Idler"):
        ob = objs[name]
        ob.rotation_euler = (0, 0, 0)
        ob.keyframe_insert("rotation_euler", frame=1)
        ob.rotation_euler = (0, step, 0)
        ob.keyframe_insert("rotation_euler", frame=1 + FRAMES_PER_PITCH)
        finish_curves(ob, cyclic=True)


# ------------------------------------------------------------------- beans
def bean_mesh(bm, label):
    if label == "foreign":                      # a stick / stone chip
        sx, sy, sz = 6.5, 1.8, 1.6
        segs, rings = 10, 6
    else:
        sx, sy, sz = 4.6, 3.3, 2.6
        segs, rings = 24, 12
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co
        if label != "foreign" and z > 0:         # coffee-bean crease on the flat side
            z *= 0.55
            z -= 0.18 * math.exp(-(y / 0.18) ** 2) * (1 - x * x)
        if label == "foreign":
            x *= 1 + 0.15 * math.sin(5 * y + 2 * z)
        v.co = Vector((x * sx * MM, y * sy * MM, z * sz * MM))
    return sz


class Keys:
    """Collects (t, location, rotation) for one bean and writes keyframes."""

    def __init__(self, ob):
        self.ob = ob
        self.t = 0.0

    def at(self, t, x, y, z, rot=(0, 0, 0)):
        self.t = t
        self.ob.location = mm(x, y, z)
        self.ob.rotation_euler = rot
        self.ob.keyframe_insert("location", frame=frame_of(t))
        self.ob.keyframe_insert("rotation_euler", frame=frame_of(t))

    def visible(self, t, show):
        self.ob.hide_viewport = self.ob.hide_render = not show
        self.ob.keyframe_insert("hide_viewport", frame=frame_of(t))
        self.ob.keyframe_insert("hide_render", frame=frame_of(t))


def animate_beans(sc, objs):
    coll = bpy.data.collections.new("Beans")
    sc.collection.children.link(coll)
    gate_windows = {g: [] for g in range(1, len(PR["gates_x"]) + 1)}
    belt_top = PR["belt_top"]
    deploy = math.radians(PR["gate_deploy_deg"])
    py = PR["gate_pivot_y"]
    end_t = 0.0
    for j, (label, gate) in enumerate(BEANS):
        mat = material(f"Bean_{label}", BEAN_COLOURS[label], 0.35 if label != "foreign" else 0.8)
        hz = {}

        def build(bm, label=label):
            hz["v"] = bean_mesh(bm, label)
        ob = new_mesh_object(f"Bean_{j + 1:02d}_{label}_gate{gate}", build, mat, coll)
        rz = belt_top + hz["v"]
        k = Keys(ob)
        ts = FIRST_SPAWN_S + j * SPAWN_EVERY_S
        y0 = random.uniform(-11, 11)
        yaw = random.uniform(-0.5, 0.5)
        k.visible(0, False)
        k.visible(ts - 0.3, True)
        k.at(ts - 0.3, LOAD_X, y0, rz + 30, (0.3, 0.2, yaw))
        k.at(ts, LOAD_X, y0, rz, (0, 0, yaw))

        def on_belt(x):
            return ts + (x - LOAD_X) / V

        if gate:
            xp = PR["gates_x"][gate - 1]
            # bean centre touches the deployed paddle (radius ~4 mm + half paddle thickness)
            clr = 4.0 + PR["fence_t"] / 2
            x_hit = xp + ((y0 - py) * math.cos(deploy) - clr) / math.sin(deploy)
            t_hit = on_belt(x_hit)
            k.at(t_hit, x_hit, y0, rz, (0, 0, yaw))
            # slides along the paddle at the belt speed projected on the paddle
            d = (math.cos(deploy), math.sin(deploy))
            y_edge = PR["wall_in"] - 3.0
            s = (y_edge - y0) / d[1]
            t_edge = t_hit + s / (V * d[0])
            x_edge = x_hit + s * d[0]
            k.at(t_edge, x_edge, y_edge, rz, (0, 0, yaw + 0.6))
            gate_windows[gate].append((on_belt(xp - PR["gate_open_before"]),
                                       max(on_belt(xp + PR["gate_close_after"]), t_edge + 0.5)))
            # pushed over the sill, then down the chute and into the bin
            sill = PR["chute_sill_z"] + hz["v"]
            k.at(t_edge + 0.45, x_edge + 3, PR["wall_out"] + 2, sill, (0.2, 0, yaw + 0.9))
            cx = xp + sum(PR["chute_open_dx"]) / 2
            fy, fz = PR["chute_floor_end"]
            t_chute = t_edge + 0.45 + 0.42
            k.at(t_chute, cx + random.uniform(-6, 6), fy, fz + hz["v"], (1.4, 0.3, yaw + 2.0))
            t_bin = t_chute + 0.16
            k.at(t_bin, cx + random.uniform(-8, 8), 97 + random.uniform(-6, 6), 1.5 + hz["v"],
                 (math.pi, 0.4, yaw + 3.0))
            end_t = max(end_t, t_bin)
        else:
            xd, az = PR["x_drive"], PR["axis_z"]
            rr = rz - az
            t_top = on_belt(xd)
            k.at(t_top, xd, y0, rz, (0, 0, yaw))
            # carried round the drum nose, then drops off
            theta_off = math.radians(75)
            for i in range(1, 6):
                th = theta_off * i / 5
                k.at(t_top + rr * th / V, xd + rr * math.sin(th), y0, az + rr * math.cos(th), (0, th, yaw))
            t0 = t_top + rr * theta_off / V
            x0, z0 = xd + rr * math.sin(theta_off), az + rr * math.cos(theta_off)
            vx, vz = V * math.cos(theta_off), -V * math.sin(theta_off)
            x_bin = PR["length"] + 22 + random.uniform(-5, 5)
            # nudge outward so it clears the frame end and lands in the bin
            t_fall = math.sqrt(2 * (z0 - hz["v"] - 1.5) / G)
            vx = max(vx, (x_bin - x0) / t_fall)
            for i in range(1, 7):
                tt = t_fall * i / 6
                k.at(t0 + tt, x0 + vx * tt, y0 + random.uniform(-1, 1), z0 + vz * tt - 0.5 * G * tt * tt,
                     (0, theta_off + 2.5 * i / 6, yaw))
            end_t = max(end_t, t0 + t_fall)
        finish_curves(ob)

    # gate paddles
    for g, wins in gate_windows.items():
        ob = objs[f"Gate_Paddle_{g}"]
        wins.sort()
        merged = []
        for a, b in wins:
            if merged and a <= merged[-1][1] + 0.3:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        ob.rotation_euler = (0, 0, 0)
        ob.keyframe_insert("rotation_euler", frame=1)
        swing = 4                                   # frames, ~SG90 speed
        for a, b in merged:
            for fr, ang in ((frame_of(a), 0), (frame_of(a) + swing, deploy),
                            (frame_of(b), deploy), (frame_of(b) + swing, 0)):
                ob.rotation_euler = (0, 0, ang)
                ob.keyframe_insert("rotation_euler", frame=fr)
        finish_curves(ob, interp="BEZIER")
    return end_t


# -------------------------------------------------------------------- main
def build():
    sc = reset_scene()
    objs = import_model()
    stage(sc)
    animate_belt(objs)
    end_t = animate_beans(sc, objs)
    sc.frame_start = 1
    sc.frame_end = int(frame_of(end_t + 1.0))
    sc.frame_set(1)
    print(f"Conveyor animation: {sc.frame_end} frames @ {FPS} fps, belt {V:.0f} mm/s")
    return sc


if __name__ == "__main__":
    scene = build()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--save" in argv:
        path = os.path.abspath(argv[argv.index("--save") + 1])
        bpy.ops.wm.save_as_mainfile(filepath=path)
        print("saved", path)
    if "--still" in argv:
        i = argv.index("--still")
        scene.frame_set(int(argv[i + 2]) if len(argv) > i + 2 and argv[i + 2].isdigit() else 1)
        if "--closeup" in argv:
            scene.camera = bpy.data.objects["Closeup_Camera"]
        scene.render.filepath = os.path.abspath(argv[i + 1])
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)
