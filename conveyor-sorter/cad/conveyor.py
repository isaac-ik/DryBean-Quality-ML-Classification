"""
Parametric 3D-printable bean-sorting conveyor (350 mm, Bambu A1 mini friendly).

Generates every printable part as an STL (already oriented for printing) plus
an assembled preview and an interference check.

    python conveyor.py            # write STLs to ../stl
    python conveyor.py --check    # also run the interference check
    python conveyor.py --preview  # also render ../docs/assembly.png

Assembly coordinate system (mm):
    x  along the belt, 0 = load end (idler), 350 = discharge end (drive)
    y  across the belt, 0 = belt centre. -y = "servo side" (wall A), +y = "chute side" (wall B)
    z  up, 0 = table
Beans travel in +x on the top run.
"""
import math
import os
import sys
from functools import reduce

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

FN = 64

# ---------------------------------------------------------------- parameters
L_TOTAL = 350.0          # overall length
SPLIT_X = 175.0          # frame is printed in two halves joined here

# belt (hinged PLA links, 1.75 mm filament as hinge pins)
P = 12.0                 # link pitch
BW = 40.0                # belt width
T = 2.5                  # link plate thickness
KR = 2.5                 # knuckle radius
PIN_R = 1.0              # pin hole radius (2.0 mm hole for 1.75 mm filament)
NSEG = 5                 # knuckle segments across the belt
SEG = BW / NSEG
KG = 0.25                # half of the gap between knuckle segments
CUT_R = KR + 0.5         # clearance around a neighbour's knuckle

# drums: 10-sided polygon, knuckles sit in pockets at the corners
N_TEETH = 10
R_PITCH = P / (2 * math.sin(math.pi / N_TEETH))   # 19.42
DRUM_RI = R_PITCH * math.cos(math.pi / N_TEETH) - 0.1
DRUM_POCKET_R = KR + 0.35
DRUM_LEN = 39.0

AXZ = 36.0               # drum axis height
X_IDLER = 26.0           # idler axis (slot gives +-6 mm of belt tension)
X_DRIVE = 326.0          # drive axis
IDLER_SLOT = (20.0, 32.0)
N_LINKS = int(round((2 * (X_DRIVE - X_IDLER) + N_TEETH * P) / P))   # 60

# frame walls
W_IN = 22.0              # inner face |y|
W_T = 5.0
W_OUT = W_IN + W_T       # 27
W_H = 66.0
FLANGE_W, FLANGE_T = 8.0, 4.0
BED_TOP = AXZ + R_PITCH - KR - 0.3      # knuckles slide on the bed
BED_X = (56.0, 302.0)
BELT_TOP = AXZ + R_PITCH + T            # 57.9

# hardware
M3_R = 1.7
M3_NUT_AF, M3_NUT_D = 6.0, 2.6
M4_R = 2.1
M4_NUT_AF, M4_NUT_D = 7.3, 3.2

# 28BYJ-48 (mounted on wall A, drive end). Shaft is 8 mm above the body centre.
MOTOR_EAR_PITCH = 35.0
MOTOR_SHAFT_OFFSET = 8.0
MOTOR_BOSS_R = 6.0

# sorter gates (SG90). Pivot x positions; fence swings from wall A toward wall B.
GATES = [96.0, 204.0, 272.0]
GATE_PIVOT_Y = -(W_IN + W_T / 2)        # pivot in the plane of wall A
FENCE_Z0 = BELT_TOP + 1.5
FENCE_H = 7.0
FENCE_LEN = 59.0
FENCE_T = 3.0
FENCE_DEPLOY_DEG = 48.0
NOTCH_Z = FENCE_Z0 - 0.4
HUB_R = 6.0
HUB_TOP = FENCE_Z0 + 10.0
PAD_LEN = 18.5           # wide section of the paddle that carries the horn
SERVO_PLATE_TOP = HUB_TOP + 6.7         # nominal; bracket slots give +-4 mm

CAMERA_X = 60.0
TIE_X = [12.0, 92.0, SPLIT_X, 258.0, 338.0]
SPLICE_HOLES = [(SPLIT_X - 15, 12.0), (SPLIT_X - 15, 24.0), (SPLIT_X + 15, 12.0), (SPLIT_X + 15, 24.0)]


# ------------------------------------------------------------------ helpers
def union(parts):
    parts = [p for p in parts if p is not None]
    return reduce(lambda a, b: a + b, parts) if parts else Manifold()


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_z(cx, cy, z0, z1, r, fn=FN):
    return Manifold.cylinder(z1 - z0, r, r, fn).translate([cx, cy, z0])


def cyl_y(cx, cz, y0, y1, r, fn=FN):
    return Manifold.cylinder(y1 - y0, r, r, fn).rotate([-90, 0, 0]).translate([cx, y0, cz])


def hex_y(cx, cz, y0, y1, af):
    return cyl_y(cx, cz, y0, y1, af / math.sqrt(3), 6)


def hex_z(cx, cy, z0, z1, af):
    return cyl_z(cx, cy, z0, z1, af / math.sqrt(3), 6)


def prism_yz(x0, x1, pts):
    """Extrude a polygon given in (y, z) along x from x0 to x1."""
    area = sum(y0 * z1 - y1 * z0 for (y0, z0), (y1, z1) in zip(pts, pts[1:] + pts[:1]))
    if area < 0:
        pts = pts[::-1]
    cs = CrossSection([[(y, z) for y, z in pts]])
    # extrude along +z then map (u,v,w)->(x=w, y=u, z=v)
    m = cs.extrude(x1 - x0)
    mat = np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    return m.transform(mat)


def rot_y_about(m, cx, cz, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    mat = np.array([[c, 0, s, cx - c * cx - s * cz],
                    [0, 1, 0, 0],
                    [-s, 0, c, cz + s * cx - c * cz]], dtype=float)
    return m.transform(mat)


def rot_z_about(m, cx, cy, deg):
    return m.translate([-cx, -cy, 0]).rotate([0, 0, deg]).translate([cx, cy, 0])


def on_bed(m):
    b = m.bounding_box()
    return m.translate([-b[0], -b[1], -b[2]])


def save_stl(m, path):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                         faces=np.asarray(mesh.tri_verts), process=True)
    tm.export(path)
    return tm


def m3_wall_hole(x, z, side):
    """Through hole in a wall with a recessed M3 nut pocket on the inner face."""
    y_out, y_in = side * W_OUT, side * W_IN
    hole = cyl_y(x, z, min(y_out, y_in) - 1, max(y_out, y_in) + 1, M3_R)
    pk = hex_y(x, z, *sorted((y_in, y_in + side * M3_NUT_D)), M3_NUT_AF)
    return hole + pk


# ------------------------------------------------------------- belt link
def belt_link():
    """One link in assembled orientation: hinges at x=0 and x=P on z=0, plate on top."""
    def seg_y(k, grow):
        return -BW / 2 + SEG * k + KG - grow, -BW / 2 + SEG * (k + 1) - KG + grow

    a_segs = [0, 2, 4]   # knuckles at x=0
    b_segs = [1, 3]      # knuckles at x=P
    plate = box(0, P, -BW / 2, BW / 2, 0, T)
    cuts = [cyl_y(0, 0, *seg_y(k, 2 * KG), CUT_R) for k in b_segs]
    cuts += [cyl_y(P, 0, *seg_y(k, 2 * KG), CUT_R) for k in a_segs]
    plate = plate - union(cuts)
    knuckles = [cyl_y(0, 0, *seg_y(k, 0), KR, 40) for k in a_segs]
    knuckles += [cyl_y(P, 0, *seg_y(k, 0), KR, 40) for k in b_segs]
    pins = cyl_y(0, 0, -BW, BW, PIN_R, 24) + cyl_y(P, 0, -BW, BW, PIN_R, 24)
    return (plate + union(knuckles)) - pins


# ------------------------------------------------------------------ drums
def drum_body():
    """Decagon prism along z (0..DRUM_LEN) with knuckle pockets at the corners."""
    rc = DRUM_RI / math.cos(math.pi / N_TEETH)
    pts = [(rc * math.cos(2 * math.pi * i / N_TEETH), rc * math.sin(2 * math.pi * i / N_TEETH))
           for i in range(N_TEETH)]
    body = CrossSection([pts]).extrude(DRUM_LEN)
    pockets = [cyl_z(R_PITCH * math.cos(2 * math.pi * i / N_TEETH),
                     R_PITCH * math.sin(2 * math.pi * i / N_TEETH),
                     -1, DRUM_LEN + 1, DRUM_POCKET_R, 32) for i in range(N_TEETH)]
    # lightening holes
    holes = [cyl_z(10.5 * math.cos(2 * math.pi * i / 5 + 0.3), 10.5 * math.sin(2 * math.pi * i / 5 + 0.3),
                   3, DRUM_LEN - 3, 3.2, 32) for i in range(5)]
    return body - union(pockets) - union(holes)


def drum_drive_print():
    """Printed hub-up. z=0 is the free (+y) end, hub at top goes to the motor (wall A)."""
    hub_len = (W_OUT - 2.0) - DRUM_LEN / 2          # hub reaches y=-25, inside the boss hole
    hub = cyl_z(0, 0, DRUM_LEN, DRUM_LEN + hub_len, 5.25)
    top = DRUM_LEN + hub_len
    d_bore = cyl_z(0, 0, top - 9.5, top + 1, 2.6) ^ box(-1.58, 1.58, -5, 5, top - 9.5, top + 1)
    free_bore = cyl_z(0, 0, -1, 14, M3_R)
    return (drum_body() + hub) - d_bore - free_bore


def drum_idler_print():
    return drum_body() - cyl_z(0, 0, -1, DRUM_LEN + 1, M3_R)


def spacer_print():
    return cyl_z(0, 0, 0, 2.2, 4.5) - cyl_z(0, 0, -1, 3.2, M3_R)


def drum_to_assembly(m, x):
    # print z -> assembly -y ; print (x,y) -> assembly (x, z); phase so a pocket sits on top
    m = m.rotate([0, 0, 90 % (360 / N_TEETH)])
    mat = np.array([[1, 0, 0, x], [0, 0, -1, DRUM_LEN / 2], [0, 1, 0, AXZ]], dtype=float)
    return m.transform(mat)


# ------------------------------------------------------------------ walls
def wall(side, x0, x1):
    """side=-1: wall A (servo + motor side). side=+1: wall B (chutes + camera side)."""
    y_in, y_out = side * W_IN, side * W_OUT
    yb = sorted((y_in, y_out))
    body = box(x0, x1, yb[0], yb[1], 0, W_H)
    fy = sorted((y_in, y_in - side * FLANGE_W))
    flange = box(x0, x1, fy[0], fy[1], 0, FLANGE_T)
    parts = [body, flange]
    lx0, lx1 = max(x0, BED_X[0]), min(x1, BED_X[1])
    if lx1 > lx0:
        ly = sorted((y_in, side * 0.2))
        parts.append(box(lx0, lx1, ly[0], ly[1], BED_TOP - 3, BED_TOP))
    m = union(parts)

    def inside(x, margin=4):
        return x0 + margin <= x <= x1 - margin

    cuts = []
    # tie-bar bolts through the bottom flange, nut pocket underneath
    for xt in TIE_X:
        xs = [xt - 7, xt + 7] if xt == SPLIT_X else [xt]
        for x in xs:
            if inside(x):
                cuts.append(cyl_z(x, side * 18, -1, FLANGE_T + 1, M3_R))
                cuts.append(hex_z(x, side * 18, -1, M3_NUT_D, M3_NUT_AF))
    for x, z in SPLICE_HOLES:
        if inside(x):
            cuts.append(m3_wall_hole(x, z, side))
    # idler tension slot + nut channel
    if inside(IDLER_SLOT[0], 0):
        s0, s1 = IDLER_SLOT
        cuts.append(Manifold.batch_hull([cyl_y(s0, AXZ, yb[0] - 1, yb[1] + 1, M3_R),
                                         cyl_y(s1, AXZ, yb[0] - 1, yb[1] + 1, M3_R)]))
        cy = sorted((y_in - side * 1, y_in + side * M3_NUT_D))
        cuts.append(box(s0 - 3.5, s1 + 3.5, cy[0], cy[1], AXZ - M3_NUT_AF / 2, AXZ + M3_NUT_AF / 2))
    if inside(X_DRIVE):
        if side < 0:   # motor
            cuts.append(cyl_y(X_DRIVE, AXZ, yb[0] - 1, yb[1] + 1, MOTOR_BOSS_R))
            for dx in (-MOTOR_EAR_PITCH / 2, MOTOR_EAR_PITCH / 2):
                xz = (X_DRIVE + dx, AXZ - MOTOR_SHAFT_OFFSET)
                cuts.append(cyl_y(*xz, yb[0] - 1, yb[1] + 1, M4_R))
                cuts.append(hex_y(*xz, *sorted((y_in - side, y_in + side * M4_NUT_D)), M4_NUT_AF))
        else:          # drive drum free-end bolt
            cuts.append(m3_wall_hole(X_DRIVE, AXZ, side))
    for xp in GATES:
        if side < 0:
            cuts.append(box(xp - 8, xp + 62, yb[0] - 1, yb[1] + 1, NOTCH_Z, W_H + 1))
            for x in (xp - 23.5, xp + 23.5):
                if inside(x):
                    cuts.append(m3_wall_hole(x, 40, side))
        else:
            cuts.append(box(xp + 34, xp + 66, yb[0] - 1, yb[1] + 1, 55, W_H + 1))
            for x in (xp + 27, xp + 73):
                if inside(x):
                    cuts.append(m3_wall_hole(x, 46, side))
    if side > 0:
        for z in (18, 34):
            if inside(CAMERA_X):
                cuts.append(m3_wall_hole(CAMERA_X, z, side))
    # trim anything outside this half (gate notch boxes can overhang)
    return (m - union(cuts)) ^ box(x0, x1, -100, 100, -1, 200)


# --------------------------------------------------------------- gate parts
def fence(xp=0.0):
    """Sorting gate paddle, parked position (lies in the plane of wall A)."""
    py = GATE_PIVOT_Y
    arm = box(xp, xp + FENCE_LEN, py - FENCE_T / 2, py + FENCE_T / 2, FENCE_Z0, FENCE_Z0 + FENCE_H)
    tip = cyl_z(xp + FENCE_LEN, py, FENCE_Z0, FENCE_Z0 + FENCE_H, FENCE_T / 2, 24)
    hub = cyl_z(xp, py, FENCE_Z0, HUB_TOP, HUB_R)
    pad = box(xp, xp + PAD_LEN, py - 4, py + 4, FENCE_Z0, HUB_TOP)
    m = arm + tip + hub + pad
    # pocket for the SG90 single-arm horn (horn points along the paddle), centre screw
    pocket = cyl_z(xp, py, HUB_TOP - 2.2, HUB_TOP + 1, 3.9) + \
        box(xp, xp + 17, py - 2.5, py + 2.5, HUB_TOP - 2.2, HUB_TOP + 1)
    screw = cyl_z(xp, py, FENCE_Z0 - 1, HUB_TOP, 1.3, 24) + cyl_z(xp, py, FENCE_Z0 - 1, HUB_TOP - 4.5, 2.9, 32)
    pilots = cyl_z(xp + 9, py, HUB_TOP - 6, HUB_TOP, 0.7, 12) + cyl_z(xp + 13, py, HUB_TOP - 6, HUB_TOP, 0.7, 12)
    return m - pocket - screw - pilots


def servo_bracket(xp=0.0):
    py = GATE_PIVOT_Y
    zt = SERVO_PLATE_TOP
    vy0, vy1 = -W_OUT - 4, -W_OUT
    vert = box(xp - 28, xp + 28, vy0, vy1, 32, zt)
    vert = vert - box(xp - 19, xp + 19, vy0 - 1, vy1 + 1, 57, zt + 1)
    top = box(xp - 28, xp + 28, -35, -14.5, zt - 3, zt)
    gus = [prism_yz(xp + a, xp + b, [(vy1, zt - 9), (vy1, zt - 3), (vy1 + 6, zt - 3)])
           for a, b in ((-27, -21), (21, 27))]
    m = vert + top + union(gus)
    # SG90 body cutout (shaft at the pivot, body extends toward -x) and screw pilots
    cut = box(xp - 16.9, xp + 6.3, py - 6.2, py + 6.2, zt - 4, zt + 1)
    body_cx = xp + (-16.6 + 5.9) / 2
    pil = [cyl_z(body_cx + d, py, zt - 8, zt + 1, 0.9, 16) for d in (-13.9, 13.9)]
    slots = [Manifold.batch_hull([cyl_y(xp + x, 36, vy0 - 1, vy1 + 1, M3_R),
                                  cyl_y(xp + x, 44, vy0 - 1, vy1 + 1, M3_R)]) for x in (-23.5, 23.5)]
    return m - cut - union(pil) - union(slots)


def chute(xp=0.0):
    """Bolts to the outside of wall B under the opening; floor slopes down 30 deg."""
    y0 = W_OUT
    flange = box(xp + 22, xp + 78, y0, y0 + 3, 40, 56)
    floor_pts = [(y0, 55), (y0 + 55, 25), (y0 + 55, 22), (y0, 52)]
    floor = prism_yz(xp + 32, xp + 68, floor_pts)
    side_pts = [(y0, 52), (y0 + 55, 22), (y0 + 55, 40), (y0, 70)]
    sides = prism_yz(xp + 32, xp + 34, side_pts) + prism_yz(xp + 66, xp + 68, side_pts)
    holes = [cyl_y(xp + x, 46, y0 - 1, y0 + 5, M3_R) for x in (27, 73)]
    return (flange + floor + sides) - union(holes)


# ------------------------------------------------------------ camera mount
def camera_mast():
    x0, x1 = CAMERA_X - 6, CAMERA_X + 6
    y0 = W_OUT
    post = box(x0, x1, y0, y0 + 8, 8, 168)
    arm = box(x0, x1, -20, y0 + 8, 160, 168)
    gus = prism_yz(x0, x1, [(y0, 160), (y0 - 6, 160), (y0, 148)])
    holes = [cyl_y(CAMERA_X, z, y0 - 1, y0 + 9, M3_R) for z in (18, 34)]
    holes += [cyl_z(CAMERA_X, y, 159, 169, M3_R) for y in (-14, 14)]
    return (post + arm + gus) - union(holes)


def camera_plate():
    z0, z1 = 157.0, 160.0
    plate = box(CAMERA_X - 25, CAMERA_X + 25, -20, 20, z0, z1)
    holes = [cyl_z(CAMERA_X, y, z0 - 1, z1 + 1, M3_R) for y in (-14, 14)]
    # Raspberry Pi camera v1/v2/v3 hole pattern (21 x 12.5, M2)
    holes += [cyl_z(CAMERA_X + dx, dy, z0 - 1, z1 + 1, 1.1, 16) for dx in (-10.5, 10.5) for dy in (-6.25, 6.25)]
    # zip-tie slots for USB webcams / ESP32-CAM boards
    holes += [box(CAMERA_X + sx - 1.5, CAMERA_X + sx + 1.5, sy - 4, sy + 4, z0 - 1, z1 + 1)
              for sx in (-21, 21) for sy in (-11, 11)]
    return plate - union(holes)


# ------------------------------------------------------------- small parts
def tie_bar(x, joint=False):
    w = 24.0 if joint else 14.0
    b = box(x - w / 2, x + w / 2, -W_IN + 0.2, W_IN - 0.2, FLANGE_T, FLANGE_T + 4)
    xs = [x - 7, x + 7] if joint else [x]
    holes = [cyl_z(hx, s * 18, 0, 20, M3_R) for hx in xs for s in (-1, 1)]
    return b - union(holes)


def splice_plate(side):
    yb = sorted((side * W_OUT, side * (W_OUT + 4)))
    b = box(SPLIT_X - 25, SPLIT_X + 25, yb[0], yb[1], 6, 30)
    return b - union([cyl_y(x, z, yb[0] - 1, yb[1] + 1, M3_R) for x, z in SPLICE_HOLES])


# ------------------------------------------------------------- belt path
def belt_hinges():
    pts = []
    top_z, bot_z = AXZ + R_PITCH, AXZ - R_PITCH
    n_straight = int(round((X_DRIVE - X_IDLER) / P))
    half = N_TEETH // 2
    for k in range(n_straight):
        pts.append((X_IDLER + k * P, top_z))
    for j in range(half):
        a = math.radians(90 - 360 / N_TEETH * j)
        pts.append((X_DRIVE + R_PITCH * math.cos(a), AXZ + R_PITCH * math.sin(a)))
    for k in range(n_straight):
        pts.append((X_DRIVE - k * P, bot_z))
    for j in range(half):
        a = math.radians(270 - 360 / N_TEETH * j)
        pts.append((X_IDLER + R_PITCH * math.cos(a), AXZ + R_PITCH * math.sin(a)))
    return pts


def belt_assembled(link):
    pts = belt_hinges()
    out = []
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        dx, dz = b[0] - a[0], b[1] - a[1]
        phi = math.atan2(-dz, dx)
        c, s = math.cos(phi), math.sin(phi)
        mat = np.array([[c, 0, s, a[0]], [0, 1, 0, 0], [-s, 0, c, a[1]]], dtype=float)
        out.append(link.transform(mat))
    return out


# -------------------------------------------------------------- assembly
def assembly(deployed_gate=None):
    parts = {}
    parts["wall_load_A"] = wall(-1, 0, SPLIT_X)
    parts["wall_load_B"] = wall(1, 0, SPLIT_X)
    parts["wall_drive_A"] = wall(-1, SPLIT_X, L_TOTAL)
    parts["wall_drive_B"] = wall(1, SPLIT_X, L_TOTAL)
    parts["drum_drive"] = drum_to_assembly(drum_drive_print(), X_DRIVE)
    parts["drum_idler"] = drum_to_assembly(drum_idler_print(), X_IDLER)
    for i, xt in enumerate(TIE_X):
        parts[f"tie_{i}"] = tie_bar(xt, xt == SPLIT_X)
    parts["splice_A"] = splice_plate(-1)
    parts["splice_B"] = splice_plate(1)
    for i, xp in enumerate(GATES):
        f = fence(xp)
        if deployed_gate == i:
            f = rot_z_about(f, xp, GATE_PIVOT_Y, FENCE_DEPLOY_DEG)
        parts[f"fence_{i}"] = f
        parts[f"bracket_{i}"] = servo_bracket(xp)
        parts[f"chute_{i}"] = chute(xp)
    parts["camera_mast"] = camera_mast()
    parts["camera_plate"] = camera_plate()
    return parts


def dummies():
    """Rough motor / servo / camera bodies for the preview only."""
    d = []
    yb = -W_OUT
    d.append(cyl_y(X_DRIVE, AXZ - MOTOR_SHAFT_OFFSET, yb - 19, yb, 14))
    d.append(box(X_DRIVE - 21, X_DRIVE + 21, yb - 1, yb, AXZ - MOTOR_SHAFT_OFFSET - 3.5, AXZ - MOTOR_SHAFT_OFFSET + 3.5))
    for xp in GATES:
        d.append(box(xp - 16.6, xp + 5.9, GATE_PIVOT_Y - 5.9, GATE_PIVOT_Y + 5.9, SERVO_PLATE_TOP - 4.1, SERVO_PLATE_TOP + 18.6))
        d.append(box(xp - 21, xp + 10.6, GATE_PIVOT_Y - 5.9, GATE_PIVOT_Y + 5.9, SERVO_PLATE_TOP, SERVO_PLATE_TOP + 2.5))
    d.append(box(CAMERA_X - 12.5, CAMERA_X + 12.5, -12, 12, 151, 157))
    return d


# ------------------------------------------------------------------ main
def export_all(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    link = belt_link()
    link_print = on_bed(link.rotate([180, 0, 0]))
    plate = []
    for r in range(8):
        for c in range(4):
            plate.append(link_print.translate([r * 19.0, c * 44.0, 0]))
    gate_print = {
        "gate_fence": on_bed(fence()),
        "servo_bracket": on_bed(servo_bracket().rotate([90, 0, 0])),
        "chute": on_bed(chute().rotate([90, 0, 0])),
    }
    parts = {
        "wall_load_A_x1": on_bed(wall(-1, 0, SPLIT_X).rotate([90, 0, 0])),
        "wall_load_B_x1": on_bed(wall(1, 0, SPLIT_X).rotate([-90, 0, 0])),
        "wall_drive_A_x1": on_bed(wall(-1, SPLIT_X, L_TOTAL).rotate([90, 0, 0])),
        "wall_drive_B_x1": on_bed(wall(1, SPLIT_X, L_TOTAL).rotate([-90, 0, 0])),
        "drum_drive_x1": on_bed(drum_drive_print()),
        "drum_idler_x1": on_bed(drum_idler_print()),
        "drum_spacer_x3": on_bed(spacer_print()),
        "belt_link_x60": link_print,
        "belt_links_plate_32pcs_print_x2": union(plate),
        "gate_fence_x3": gate_print["gate_fence"],
        "servo_bracket_x3": gate_print["servo_bracket"],
        "chute_x3": gate_print["chute"],
        "tie_bar_x4": on_bed(tie_bar(0)),
        "tie_bar_joint_x1": on_bed(tie_bar(0, True)),
        "splice_plate_x2": on_bed(splice_plate(1).rotate([-90, 0, 0])),
        "camera_mast_x1": on_bed(camera_mast().rotate([0, 90, 0])),
        "camera_plate_x1": on_bed(camera_plate()),
    }
    print(f"{'part':38s} {'X':>6s} {'Y':>6s} {'Z':>6s}  {'cm3':>6s}  fits A1 mini")
    for name, m in parts.items():
        assert m.status().name == "NoError", name
        b = m.bounding_box()
        size = (b[3] - b[0], b[4] - b[1], b[5] - b[2])
        fits = all(s <= 180 for s in size)
        save_stl(m, os.path.join(out_dir, name + ".stl"))
        print(f"{name:38s} {size[0]:6.1f} {size[1]:6.1f} {size[2]:6.1f}  {m.volume() / 1000:6.1f}  {'yes' if fits else 'NO'}")
    print(f"belt links needed: {N_LINKS} (pitch {P} mm, drum pitch radius {R_PITCH:.2f} mm)")


def check():
    link = belt_link()
    belt = belt_assembled(link)
    print("\nInterference check (overlap volume, mm^3):")
    bad = 0
    for state in (None, 0, 1, 2):
        parts = assembly(state)
        names = list(parts)
        if state is None:
            pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        else:
            pairs = [(f"fence_{state}", b) for b in names if b != f"fence_{state}"]
        for a, b in pairs:
            v = (parts[a] ^ parts[b]).volume()
            if v > 0.05:
                bad += 1
                print(f"  [{'parked' if state is None else 'gate %d deployed' % state}] {a} x {b}: {v:.2f}")
        if state is None:
            frame = union([parts[n] for n in names if not n.startswith("drum")])
            for i, lk in enumerate(belt):
                v = (lk ^ frame).volume()
                if v > 0.05:
                    bad += 1
                    print(f"  belt link {i} x frame: {v:.2f}")
            for i in range(len(belt)):
                v = (belt[i] ^ belt[(i + 1) % len(belt)]).volume()
                if v > 0.05:
                    bad += 1
                    print(f"  belt link {i} x link {i + 1}: {v:.2f}")
    print("  no interferences" if bad == 0 else f"  {bad} interferences")
    return bad


def preview(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    global FN
    FN = 24
    parts = assembly(deployed_gate=1)
    colours = {"wall": "#9aa7b4", "drum": "#e0a040", "tie": "#7d8a96", "splice": "#7d8a96",
               "fence": "#d9534f", "bracket": "#5b8bd6", "chute": "#5cb85c", "camera": "#8e6cc9"}
    groups = []
    for n, m in parts.items():
        groups.append((m, next(c for k, c in colours.items() if n.startswith(k))))
    for lk in belt_assembled(belt_link()):
        groups.append((lk, "#f2f2ee"))
    for d in dummies():
        groups.append((d, "#3a3a3a"))
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)

    def draw(ax, elev, azim):
        tris, cols = [], []
        for m, col in groups:
            mesh = m.to_mesh()
            v = np.asarray(mesh.vert_properties)[:, :3]
            tri = v[np.asarray(mesh.tri_verts)]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
            base = np.array(matplotlib.colors.to_rgb(col))
            tris.append(tri)
            cols.append(np.clip(base[None, :] * shade[:, None], 0, 1))
        # one collection so matplotlib depth-sorts every triangle together
        ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=np.concatenate(cols),
                                             edgecolors="none", linewidths=0))
        ax.set_xlim(10, 340); ax.set_ylim(-150, 150); ax.set_zlim(-40, 180)
        ax.set_box_aspect((330, 300, 220))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

    fig = plt.figure(figsize=(18, 8), dpi=110)
    for i, (elev, azim) in enumerate(((30, -60), (55, 200))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_proj_type("ortho")
        ax.dist = 6
        draw(ax, elev, azim)
    fig.subplots_adjust(-0.05, -0.1, 1.05, 1.1, wspace=-0.15)
    fig.savefig(path, facecolor="white")
    print("preview written:", path)


def export_blender(out_dir):
    """Assembled model as a GLB (metres, one object per part) + params for animate_conveyor.py.

    Moving parts have their origin on their pivot so they can be animated directly:
    drums on the axle (spin about local Y), gate paddles on the servo shaft (swing about
    local Z), belt links on their trailing hinge.
    """
    import json
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals

    os.makedirs(out_dir, exist_ok=True)
    s = 0.001                                           # mm -> m
    # our Z-up -> glTF Y-up; Blender's importer converts it back, so transforms stay clean
    C = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    Ci = np.linalg.inv(C)
    mats = {}

    def material(name, rgb, rough=0.55, metal=0.0):
        if name not in mats:
            lin = [c ** 2.2 for c in rgb]                # glTF colours are linear
            mats[name] = PBRMaterial(name=name, baseColorFactor=[*lin, 1.0],
                                     roughnessFactor=rough, metallicFactor=metal)
        return mats[name]

    scene = trimesh.Scene()
    groups = set()

    def add(name, m, mat, origin=(0, 0, 0), rot_y_deg=0.0, parent=None):
        mesh = m.to_mesh()
        v = np.asarray(mesh.vert_properties)[:, :3] * s
        v = v @ C[:3, :3].T
        tm = trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.tri_verts), process=False)
        tm.visual = TextureVisuals(material=mat)
        a = math.radians(rot_y_deg)
        T = np.eye(4)
        T[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        T[:3, 3] = np.array(origin) * s
        kw = {}
        if parent:
            if parent not in groups:
                scene.graph.update(frame_to=parent, frame_from=scene.graph.base_frame, matrix=np.eye(4))
                groups.add(parent)
            kw["parent_node_name"] = parent
        scene.add_geometry(tm, node_name=name, geom_name=name, transform=C @ T @ Ci, **kw)

    frame_mat = material("Frame_PLA", (0.58, 0.63, 0.68))
    belt_mat = material("Belt_PLA_Blue", (0.08, 0.30, 0.75))
    drum_mat = material("Drum_PLA", (0.90, 0.60, 0.20))
    fence_mat = material("Gate_PLA", (0.85, 0.25, 0.22))
    bracket_mat = material("Bracket_PLA", (0.33, 0.52, 0.85))
    chute_mat = material("Chute_PLA", (0.30, 0.68, 0.33))
    cam_mat = material("Mast_PLA", (0.55, 0.42, 0.78))
    dark = material("Electronics", (0.06, 0.06, 0.07), rough=0.4)
    motor_mat = material("Motor_Steel", (0.62, 0.64, 0.66), rough=0.35, metal=0.8)
    servo_mat = material("Servo_Blue", (0.10, 0.25, 0.65), rough=0.45)

    parts = assembly()
    for n, m in parts.items():
        if n.startswith("drum") or n.startswith("fence"):
            continue
        if n.startswith("wall"):
            add(n.replace("wall_", "Wall_"), m, frame_mat, parent="Frame")
        elif n.startswith(("tie", "splice")):
            add(n.capitalize(), m, frame_mat, parent="Frame")
        elif n.startswith("bracket"):
            add(f"Servo_Bracket_{int(n[-1]) + 1}", m, bracket_mat, parent="Gates")
        elif n.startswith("chute"):
            add(f"Chute_{int(n[-1]) + 1}", m, chute_mat, parent="Gates")
        else:
            add(n.replace("camera_", "Camera_").title(), m, cam_mat, parent="Camera_Mount")

    drum_off = [0, 0, -AXZ]
    add("Drum_Drive", drum_to_assembly(drum_drive_print(), 0).translate(drum_off), drum_mat, (X_DRIVE, 0, AXZ))
    add("Drum_Idler", drum_to_assembly(drum_idler_print(), 0).translate(drum_off), drum_mat, (X_IDLER, 0, AXZ))
    for i, xp in enumerate(GATES):
        pivot = (xp, GATE_PIVOT_Y, FENCE_Z0)
        add(f"Gate_Paddle_{i + 1}", fence(xp).translate([-p for p in pivot]), fence_mat, pivot, parent="Gates")

    link = belt_link()
    pts = belt_hinges()
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        phi = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))
        add(f"Belt_Link_{i:02d}", link, belt_mat, (a[0], 0, a[1]), phi, parent="Belt")

    # stand-ins for bought parts
    yb = -W_OUT
    mz = AXZ - MOTOR_SHAFT_OFFSET
    motor = cyl_y(X_DRIVE, mz, yb - 19, yb, 14) + \
        box(X_DRIVE - 21, X_DRIVE + 21, yb - 1, yb, mz - 3.5, mz + 3.5) + \
        cyl_y(X_DRIVE, AXZ, yb - 0.5, yb + 1.5, 4.5)
    add("Motor_28BYJ48", motor, motor_mat, parent="Electronics")
    add("Motor_Cover", box(X_DRIVE - 8.5, X_DRIVE + 8.5, yb - 17, yb - 2, mz - 17, mz - 8), servo_mat, parent="Electronics")
    for i, xp in enumerate(GATES):
        sv = box(xp - 16.6, xp + 5.9, GATE_PIVOT_Y - 5.9, GATE_PIVOT_Y + 5.9, SERVO_PLATE_TOP - 4.1, SERVO_PLATE_TOP + 18.6) + \
            box(xp - 21, xp + 10.6, GATE_PIVOT_Y - 5.9, GATE_PIVOT_Y + 5.9, SERVO_PLATE_TOP, SERVO_PLATE_TOP + 2.5)
        add(f"Servo_SG90_{i + 1}", sv, servo_mat, parent="Electronics")
    camera = box(CAMERA_X - 12.5, CAMERA_X + 12.5, -12, 12, 151, 157) + cyl_z(CAMERA_X, 0, 147, 151, 4)
    add("Camera_Module", camera, dark, parent="Electronics")

    glb = os.path.join(out_dir, "conveyor_assembly.glb")
    scene.export(glb, include_normals=False)

    params = {
        "units": "glb is in metres; values below are in mm",
        "pitch": P, "belt_width": BW, "n_links": N_LINKS, "n_teeth": N_TEETH,
        "r_pitch": R_PITCH, "axis_z": AXZ, "x_idler": X_IDLER, "x_drive": X_DRIVE,
        "belt_top": BELT_TOP, "hinges_xz": belt_hinges(),
        "gates_x": GATES, "gate_pivot_y": GATE_PIVOT_Y, "gate_deploy_deg": FENCE_DEPLOY_DEG,
        "gate_open_before": 22.0, "gate_close_after": 80.0, "fence_t": FENCE_T,
        "wall_in": W_IN, "wall_out": W_OUT, "wall_h": W_H,
        "chute_sill_z": 55.0, "chute_floor_end": [W_OUT + 55, 25.0], "chute_open_dx": [34.0, 66.0],
        "camera_x": CAMERA_X, "length": L_TOTAL,
    }
    with open(os.path.join(out_dir, "conveyor_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=1)
    print("blender export:", glb)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if "--blender" in sys.argv:
        export_blender(os.path.join(here, "..", "blender"))
        sys.exit()
    export_all(os.path.join(here, "..", "stl"))
    if "--check" in sys.argv:
        check()
    if "--preview" in sys.argv:
        os.makedirs(os.path.join(here, "..", "docs"), exist_ok=True)
        preview(os.path.join(here, "..", "docs", "assembly.png"))
