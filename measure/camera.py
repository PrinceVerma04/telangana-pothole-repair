"""Pinhole helpers shared by surface / geometry / synthetic code (PLAN 4.3).

Camera frame: x right, y down, z forward (OpenCV). Ray for pixel (u, v) is
d = ((u - cx)/fx, (v - cy)/fy, 1) -- *unnormalised, z-component 1* so that a point at
camera depth Z is simply Z * d.
"""

from __future__ import annotations

import numpy as np

from measure.types import CalibCert


def pixel_rays(K: np.ndarray, h: int, w: int, dtype=np.float64) -> np.ndarray:
    """(H, W, 3) unnormalised rays with z = 1."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = (np.arange(w, dtype=dtype) - cx) / fx
    v = (np.arange(h, dtype=dtype) - cy) / fy
    d = np.empty((h, w, 3), dtype=dtype)
    d[..., 0] = u[None, :]
    d[..., 1] = v[:, None]
    d[..., 2] = 1.0
    return d


def rays_at(K: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N, 3) rays for (possibly sub-pixel) coordinates."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(np.asarray(u, float) - cx) / fx, (np.asarray(v, float) - cy) / fy,
                     np.ones(np.shape(u))], axis=-1)


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W) camera-z depth -> (H, W, 3) points in the camera frame."""
    h, w = depth.shape
    return pixel_rays(K, h, w) * depth[..., None]


def cert_for_image(cert: CalibCert, img_w: int, img_h: int) -> tuple[CalibCert, dict]:
    """Return a cert whose intrinsics match an image of size (img_w, img_h).

    Same aspect ratio -> intrinsics scaled exactly. Different aspect ratio (e.g. a 720x720 photo
    with a 1280x720 demo cert) -> keep the horizontal field of view (fx/W), square pixels, principal
    point at the centre. This is an *assumption*; it is recorded in the returned info dict.
    """
    info = {"intrinsics": "as_certified"}
    if (img_w, img_h) == (cert.img_w, cert.img_h):
        return cert, info
    sx, sy = img_w / cert.img_w, img_h / cert.img_h
    d = cert.to_dict()
    if abs(sx - sy) < 1e-3:
        d.update(fx=cert.fx * sx, fy=cert.fy * sy, cx=cert.cx * sx, cy=cert.cy * sy, img_w=img_w, img_h=img_h)
        info = {"intrinsics": "scaled", "scale": sx}
    else:
        f = cert.fx / cert.img_w * img_w
        d.update(fx=f, fy=f, cx=img_w / 2.0, cy=img_h / 2.0, img_w=img_w, img_h=img_h)
        info = {"intrinsics": "hfov_preserved_aspect_mismatch", "from": [cert.img_w, cert.img_h],
                "to": [img_w, img_h], "fx": f}
    return CalibCert.from_dict(d), info


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def world_to_camera_rotation(pitch_deg: float, roll_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Rotation taking WORLD/ROAD vectors (X right, Y up, Z forward) into the CAMERA frame.

    Camera looks forward, tilted DOWN by ``pitch_deg`` (positive = looking down), then rolled about
    its optical axis by ``roll_deg`` (positive = clockwise in the image), yawed by ``yaw_deg``
    (positive = right). Columns = world axes expressed in camera coordinates.
    """
    # world (X right, Y up, Z fwd) -> camera at zero pitch (x right, y down, z fwd): flip Y.
    flip = np.diag([1.0, -1.0, 1.0])
    # pitch down by p: world forward-down (0, -sin p, cos p) must map to camera +z -> rot_x(+p).
    # world up then maps to n = (0, -cos p, -sin p).
    return rot_z(roll_deg) @ rot_x(pitch_deg) @ rot_y(-yaw_deg) @ flip


def plane_from_extrinsics(h_m: float, pitch_deg: float, roll_deg: float = 0.0) -> tuple[np.ndarray, float]:
    """Road plane (n, d) in the camera frame, n = road 'up' (unit), n.p + d = 0, d = h."""
    R = world_to_camera_rotation(pitch_deg, roll_deg)
    n = R @ np.array([0.0, 1.0, 0.0])
    return n / np.linalg.norm(n), float(h_m)
