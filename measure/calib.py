"""Calibration: certificate I/O, undistortion, checkerboard intrinsics, PnP extrinsics, drift (PLAN 4.1).

Road/object frame used for extrinsics: X right, Y up, Z forward on the road plane (y = 0 on the road),
origin directly below the camera is NOT required -- PnP returns the camera centre. Internally OpenCV
needs a right-handed object frame, so road points are passed as (x, -y, z).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from measure.surface import pitch_roll_from_normal
from measure.types import CalibCert, DriftRecord, TruthFit

INTRINSICS_MAX_RMS_PX = 0.3
KNOWN_OBJECT_MAX_ERR_PCT = 5.0
SCALE_DRIFT_ALERT = 0.05
PITCH_DRIFT_ALERT_DEG = 3.0


# ----------------------------------------------------------------------------- cert I/O
def load_cert(path: str | Path) -> CalibCert:
    return CalibCert.load(path)


def save_cert(cert: CalibCert, path: str | Path) -> None:
    cert.save(path)


def has_distortion(cert: CalibCert, eps: float = 1e-9) -> bool:
    return any(abs(x) > eps for x in cert.dist)


# ----------------------------------------------------------------------------- undistortion
def undistort_maps(cert: CalibCert, new_K: Optional[np.ndarray] = None) -> tuple[Any, Any, np.ndarray]:
    """cv2 remap tables for the raw -> undistorted frame. ``new_K`` defaults to the certified K (so pixel
    scale is preserved and K' = K); returns (map1, map2, K')."""
    import cv2

    K = cert.K_np()
    Kn = K.copy() if new_K is None else np.asarray(new_K, float)
    m1, m2 = cv2.initUndistortRectifyMap(K, cert.dist_np(), None, Kn, (cert.img_w, cert.img_h), cv2.CV_32FC1)
    return m1, m2, Kn


def undistort_image(img: np.ndarray, cert: CalibCert, new_K: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    if not has_distortion(cert):
        return img, cert.K_np()
    m1, m2, Kn = undistort_maps(cert, new_K)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR), Kn


def undistort_mask(mask: np.ndarray, cert: CalibCert, new_K: Optional[np.ndarray] = None) -> np.ndarray:
    import cv2

    if not has_distortion(cert):
        return mask
    m1, m2, _ = undistort_maps(cert, new_K)
    return cv2.remap(mask.astype(np.uint8), m1, m2, cv2.INTER_NEAREST).astype(bool)


def undistort_points(pts_px: np.ndarray, cert: CalibCert) -> np.ndarray:
    """Raw pixel coordinates (N, 2) (e.g. observation polygons) -> undistorted pixels in K."""
    import cv2

    if not has_distortion(cert):
        return np.asarray(pts_px, float)
    K = cert.K_np()
    p = cv2.undistortPoints(np.asarray(pts_px, np.float64).reshape(-1, 1, 2), K, cert.dist_np(), P=K)
    return p.reshape(-1, 2)


# ----------------------------------------------------------------------------- intrinsics
def board_object_points(pattern: tuple[int, int] = (9, 6), square_m: float = 0.035) -> np.ndarray:
    cols, rows = pattern
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return g


def find_checkerboard(img: np.ndarray, pattern: tuple[int, int] = (9, 6)) -> Optional[np.ndarray]:
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit).reshape(-1, 2)


def calibrate_intrinsics(image_points: Sequence[np.ndarray], image_size: tuple[int, int],
                         pattern: tuple[int, int] = (9, 6), square_m: float = 0.035,
                         fix_k3: bool = True) -> dict[str, Any]:
    """cv2.calibrateCamera on detected corner sets. Accept if RMS reprojection < 0.3 px (PLAN 4.1)."""
    import cv2

    obj = board_object_points(pattern, square_m)
    objs = [obj for _ in image_points]
    imgs = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in image_points]
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, image_size, None, None, flags=flags)
    return {"rms_px": float(rms), "K": K, "dist": dist.reshape(-1)[:5].tolist(), "n_views": len(imgs),
            "accepted": bool(rms < INTRINSICS_MAX_RMS_PX), "image_size": list(image_size)}


def calibrate_from_images(images: Iterable[np.ndarray], pattern: tuple[int, int] = (9, 6),
                          square_m: float = 0.035) -> dict[str, Any]:
    pts, size = [], None
    for im in images:
        size = (im.shape[1], im.shape[0])
        c = find_checkerboard(im, pattern)
        if c is not None:
            pts.append(c)
    if len(pts) < 3 or size is None:
        raise ValueError(f"need >= 3 checkerboard views, found {len(pts)}")
    return calibrate_intrinsics(pts, size, pattern, square_m)


# ----------------------------------------------------------------------------- extrinsics
def solve_extrinsics(road_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray,
                     dist: Optional[np.ndarray] = None) -> dict[str, Any]:
    """PnP on known road points (cone bases / painted square corners), road frame X right, Y up, Z fwd.

    Returns pitch (+down), roll, yaw (+right) in degrees, camera height h (m), reprojection RMS (px).
    """
    import cv2

    obj = np.asarray(road_pts, np.float64).copy()
    obj[:, 1] *= -1.0  # right-handed object frame (Y down) for OpenCV
    ip = np.asarray(img_pts, np.float64).reshape(-1, 1, 2)
    D = np.zeros(5) if dist is None else np.asarray(dist, float)
    coplanar = np.allclose(obj[:, 1], obj[0, 1], atol=1e-6)
    flag = cv2.SOLVEPNP_IPPE if (coplanar and len(obj) >= 4) else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, ip, K, D, flags=flag)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, ip, K, D, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).reshape(3)
    h = -float(C[1])  # Y down in object frame
    up_cam = R @ np.array([0.0, -1.0, 0.0])
    pitch, roll = pitch_roll_from_normal(up_cam)
    zc = R.T @ np.array([0.0, 0.0, 1.0])
    yaw = math.degrees(math.atan2(zc[0], zc[2]))
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1))))
    return {"pitch_deg": pitch, "roll_deg": roll, "yaw_deg": yaw, "h_m": h, "reproj_rms_px": rms,
            "camera_xz": [float(C[0]), float(C[2])], "rvec": rvec.reshape(3).tolist(), "tvec": tvec.reshape(3).tolist()}


def pixel_to_road(uv: np.ndarray, K: np.ndarray, h_m: float, pitch_deg: float, roll_deg: float = 0.0,
                  yaw_deg: float = 0.0) -> np.ndarray:
    """Undistorted pixels (N, 2) -> road-plane (x, z) metres (camera foot point = origin)."""
    from measure.camera import rays_at, world_to_camera_rotation

    uv = np.asarray(uv, float).reshape(-1, 2)
    R_wc = world_to_camera_rotation(pitch_deg, roll_deg, yaw_deg)
    r = rays_at(K, uv[:, 0], uv[:, 1]) @ R_wc  # world directions
    t = -h_m / r[:, 1]
    p = np.array([0.0, h_m, 0.0]) + t[:, None] * r
    return p[:, [0, 2]]


def polygon_area(xz: np.ndarray) -> float:
    x, z = xz[:, 0], xz[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


def known_object_error(corners_px: np.ndarray, true_area_m2: float, K: np.ndarray, h_m: float, pitch_deg: float,
                       roll_deg: float = 0.0, yaw_deg: float = 0.0) -> dict[str, Any]:
    """PLAN 4.1 step 2 acceptance: the 1 m^2 painted square at 5 m must reconstruct within 5 %."""
    xz = pixel_to_road(corners_px, K, h_m, pitch_deg, roll_deg, yaw_deg)
    a = polygon_area(xz)
    err = 100.0 * abs(a - true_area_m2) / true_area_m2
    return {"area_m2": a, "err_pct": err, "accepted": bool(err <= KNOWN_OBJECT_MAX_ERR_PCT)}


# ----------------------------------------------------------------------------- drift
def drift_check(cert: CalibCert, trip_id: str, h_hats: Sequence[float], s_nominal: float = 1.0,
                pitch_ests: Optional[Sequence[float]] = None) -> tuple[DriftRecord, list[str]]:
    """PLAN 4.1 step 3: scale_drift = |median(h_hat) * s_nominal - h_true| / h_true (alert > 5 %);
    pitch drift = |median(pitch_est) - certified pitch| (alert > 3 deg).

    ``h_hats`` are per-frame RANSAC heights in raw depth units; ``s_nominal`` is the device's nominal
    depth-scale (median s over the certification drive; 1.0 for a perfectly metric model).
    """
    hh = np.asarray([h for h in h_hats if np.isfinite(h)], float)
    alerts: list[str] = []
    sd = float(abs(np.median(hh) * s_nominal - cert.h_true_m) / cert.h_true_m) if len(hh) else float("nan")
    if np.isfinite(sd) and sd > SCALE_DRIFT_ALERT:
        alerts.append(f"scale_drift {sd:.1%} > {SCALE_DRIFT_ALERT:.0%}: re-measure camera height / remount")
    pd = None
    if pitch_ests:
        pe = np.asarray([p for p in pitch_ests if np.isfinite(p)], float)
        if len(pe):
            pd = float(abs(np.median(pe) - cert.pitch_deg))
            if pd > PITCH_DRIFT_ALERT_DEG:
                alerts.append(f"pitch_drift {pd:.1f} deg > {PITCH_DRIFT_ALERT_DEG} deg: camera moved")
    rec = DriftRecord(trip_id=trip_id, scale_drift=sd, pitch_drift=pd, at=datetime.now(timezone.utc))
    return rec, alerts


def append_drift(cert: CalibCert, rec: DriftRecord, keep: int = 200) -> CalibCert:
    d = cert.to_dict()
    hist = d.get("drift_history", []) + [rec.to_dict()]
    d["drift_history"] = hist[-keep:]
    return CalibCert.from_dict(d)


# ----------------------------------------------------------------------------- certificate
def make_cert(calib_id: str, device_id: str, intr: dict[str, Any], extr: dict[str, Any], *, vehicle_kind: str = "bus",
              h_true_m: Optional[float] = None, known_object_err_pct: Optional[float] = None,
              calibrated_by: Optional[str] = None, lateral_offset_m: float = 0.0,
              wheel_tracks: tuple[float, float] = (-0.9, 0.9), notes: Optional[str] = None) -> CalibCert:
    """Assemble a certificate. ``certified`` only if intrinsics RMS < 0.3 px AND the known object
    reconstructs within 5 % (PLAN 4.1); the tape-measured ``h_true_m`` overrides the PnP height."""
    K = np.asarray(intr["K"], float)
    w, h = intr["image_size"]
    certified = bool(intr.get("accepted")) and known_object_err_pct is not None and known_object_err_pct <= KNOWN_OBJECT_MAX_ERR_PCT
    return CalibCert(calib_id=calib_id, device_id=device_id, vehicle_kind=vehicle_kind, fx=float(K[0, 0]),
                     fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), dist=list(intr["dist"])[:5],
                     img_w=int(w), img_h=int(h), reproj_rms_px=float(intr["rms_px"]),
                     h_true_m=float(h_true_m if h_true_m is not None else extr["h_m"]),
                     pitch_deg=float(extr["pitch_deg"]), roll_deg=float(extr["roll_deg"]),
                     yaw_deg=float(extr["yaw_deg"]), lateral_offset_m=lateral_offset_m,
                     wheel_track_left_m=wheel_tracks[0], wheel_track_right_m=wheel_tracks[1],
                     known_object_err_pct=known_object_err_pct, calibrated_at=datetime.now(timezone.utc),
                     calibrated_by=calibrated_by, certified=certified, truth_fit=TruthFit(), notes=notes)
