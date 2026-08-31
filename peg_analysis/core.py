from pathlib import Path
import json
import numpy as np


def config(p):
    import yaml
    c = yaml.safe_load(open(p))
    for k in ("dataset_path", "output_dir", "main_camera_serial", "secondary_camera_serial"):
        if not c.get(k) or str(c[k]).startswith("ENTER_"):
            raise ValueError(f"Set {k}")
    return c


def demos(h):
    return sorted(k for k in h["demos"] if k.startswith("demo_") and bool(h["demos"][k].attrs.get("complete", True)))


def group(h, d, s):
    return h[f"demos/{d}/cameras/{s}"]


def detect_gap(t, a=8, b=3):
    dt = np.diff(np.asarray(t, dtype=np.int64)).astype(float)
    q = dt[dt > 0]
    med = np.median(q)
    mad = np.median(abs(q-med))
    ids = np.flatnonzero(dt > max(b*med, med+a*max(mad, 1)))
    if len(ids) != 1:
        raise ValueError(f"Expected one gap, found {len(ids)}")
    i = int(ids[0])
    return i, i+1, int(dt[i])


def feat(m):
    y, x = np.nonzero(m)
    if not len(x):
        return None
    p = np.c_[x, y].astype(np.float64)
    c = p.mean(axis=0)
    centered = p-c
    if len(p) >= 2:
        values, vectors = np.linalg.eigh(centered.T @ centered / len(p))
        axis = vectors[:, int(np.argmax(values))]
    else:
        axis = np.array([0.0, 1.0])
    if axis[1] < 0 or (abs(axis[1]) < 1e-12 and axis[0] < 0):
        axis = -axis
    axis = axis / np.linalg.norm(axis)
    projection = centered @ axis
    s_min, s_max = float(projection.min()), float(projection.max())
    endpoint_top = c + s_min*axis
    endpoint_bottom = c + s_max*axis
    if endpoint_top[1] > endpoint_bottom[1]:
        endpoint_top, endpoint_bottom = endpoint_bottom, endpoint_top
        s_min, s_max = -s_max, -s_min
        axis = -axis
    y_tip = int(y.max())
    x_tip = float(np.median(x[y == y_tip]))
    return dict(c=c, axis=axis, normal=np.array([-axis[1], axis[0]]),
                endpoint_top=endpoint_top, endpoint_bottom=endpoint_bottom,
                projection_min=s_min, projection_max=s_max,
                visible_length=float(s_max-s_min),
                thumb_tip=np.array([x_tip, float(y_tip)]), area=len(x))


def axial_above_thumb_length(peg, hand, eps=1e-6):
    ay = float(peg["axis"][1])
    if abs(ay) < eps:
        return np.nan
    lam = (float(hand["thumb_tip"][1])-float(peg["c"][1]))/ay
    return float(lam-peg["projection_min"])


def signed_point_axis_distance(point, axis_origin, axis_normal):
    return float((np.asarray(point)-np.asarray(axis_origin)) @ np.asarray(axis_normal))


def peg_angle_deg(axis):
    """Signed image-plane angle from downward image vertical."""
    return float(np.degrees(np.arctan2(float(axis[0]), float(axis[1]))))




def length_scale(f, physical_length_mm):
    if physical_length_mm is None or f is None or f["visible_length"] <= 0:
        return np.nan
    return float(physical_length_mm)/float(f["visible_length"])


def save(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    json.dump(x, open(p, "w"), indent=2)
