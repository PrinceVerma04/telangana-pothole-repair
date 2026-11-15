"""Pure-numpy detection/segmentation metrics for ml/eval.py (no torch).

* Matching is COCO-style: per image and class, predictions in descending confidence take the unmatched ground truth
  with the highest IoU, at each of the 10 IoU thresholds 0.50:0.05:0.95.
* AP uses the 101-point interpolated precision envelope (same as Ultralytics ``compute_ap(method='interp')`` and COCO).
* Mask IoU is computed at a reduced resolution (``Evaluator.mask_max_side``) to keep memory flat; Ultralytics' own val
  does the same at proto resolution.

The evaluator is incremental (``add_image``) so masks are never held for the whole split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4), xyxy."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def mask_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boolean masks a (N,H,W) and b (M,H,W) of the same shape."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.ndim == 2:
        a = a[None]
    if b.ndim == 2:
        b = b[None]
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    fa = a.reshape(len(a), -1).astype(np.float32)
    fb = b.reshape(len(b), -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(1)[:, None] + fb.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match(iou: np.ndarray, pred_conf: np.ndarray, thresholds: np.ndarray = IOU_THRESHOLDS) -> np.ndarray:
    """Greedy COCO matching for one image and one class.

    iou: (P, G) IoU matrix; pred_conf: (P,). Returns tp (P, T) bool, in the original prediction order.
    """
    P = iou.shape[0]
    tp = np.zeros((P, len(thresholds)), bool)
    if P == 0 or iou.shape[1] == 0:
        return tp
    order = np.argsort(-np.asarray(pred_conf), kind="stable")
    for ti, t in enumerate(thresholds):
        taken = np.zeros(iou.shape[1], bool)
        for p in order:
            cand = np.where(~taken & (iou[p] >= t - 1e-9), iou[p], -1.0)
            g = int(np.argmax(cand))
            if cand[g] >= 0:
                taken[g] = True
                tp[p, ti] = True
    return tp


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP from a recall/precision curve (recall ascending)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def ap_per_threshold(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> tuple[np.ndarray, dict[str, float]]:
    """AP at each IoU threshold for one class, plus P/R at conf >= 0.25 and the max-F1 point (IoU 0.5)."""
    T = tp.shape[1] if tp.ndim == 2 else len(IOU_THRESHOLDS)
    if n_gt == 0:
        return np.full(T, np.nan), {}
    if len(conf) == 0:
        return np.zeros(T), {"p_at_025": 0.0, "r_at_025": 0.0, "best_f1": 0.0, "best_f1_conf": None}
    order = np.argsort(-conf, kind="stable")
    tp = tp[order]
    c = conf[order]
    ctp = np.cumsum(tp, axis=0)
    cfp = np.cumsum(~tp, axis=0)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    aps = np.array([compute_ap(recall[:, t], precision[:, t]) for t in range(T)])
    k = int(np.searchsorted(-c, -0.25, side="right"))  # predictions with conf >= 0.25
    p25 = float(precision[k - 1, 0]) if k > 0 else 0.0
    r25 = float(recall[k - 1, 0]) if k > 0 else 0.0
    f1 = 2 * precision[:, 0] * recall[:, 0] / np.maximum(precision[:, 0] + recall[:, 0], 1e-12)
    bi = int(np.argmax(f1))
    return aps, {"p_at_025": p25, "r_at_025": r25, "best_f1": float(f1[bi]), "best_f1_conf": float(c[bi])}


def downsample_mask(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour bool mask resize (no cv2 needed)."""
    mask = np.asarray(mask, bool)
    H, W = mask.shape
    h, w = out_hw
    if (H, W) == (h, w):
        return mask
    yi = np.minimum((np.arange(h) + 0.5) * H / h, H - 1).astype(int)
    xi = np.minimum((np.arange(w) + 0.5) * W / w, W - 1).astype(int)
    return mask[yi][:, xi]


def eval_hw(h: int, w: int, max_side: int) -> tuple[int, int]:
    s = min(1.0, max_side / float(max(h, w)))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


@dataclass
class _ClassStats:
    tp_box: list[np.ndarray] = field(default_factory=list)
    tp_mask: list[np.ndarray] = field(default_factory=list)
    conf: list[np.ndarray] = field(default_factory=list)
    conf_mask: list[np.ndarray] = field(default_factory=list)
    n_gt: int = 0
    n_gt_mask: int = 0
    matched_mask_ious: list[float] = field(default_factory=list)


@dataclass
class Evaluator:
    """Incremental evaluator. Classes are indices into ``names``.

    add_image(gt_cls, gt_boxes, pred_cls, pred_conf, pred_boxes, gt_masks=None, pred_masks=None, image_hw=None)
    Masks (when both are given) must be full-image bool arrays of shape image_hw; they are downsampled here.
    """

    names: Sequence[str]
    mask_max_side: int = 256
    op_conf: float = 0.25  # operating point for matched-mask IoU and pixel IoU
    stats: dict[int, _ClassStats] = field(default_factory=dict)
    n_images: int = 0
    n_images_with_masks: int = 0
    pix_inter: dict[int, float] = field(default_factory=dict)
    pix_union: dict[int, float] = field(default_factory=dict)

    def _s(self, c: int) -> _ClassStats:
        if c not in self.stats:
            self.stats[c] = _ClassStats()
        return self.stats[c]

    def add_image(self, gt_cls: Sequence[int], gt_boxes: np.ndarray, pred_cls: Sequence[int],
                  pred_conf: Sequence[float], pred_boxes: np.ndarray, gt_masks: Optional[Sequence[np.ndarray]] = None,
                  pred_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
                  image_hw: Optional[tuple[int, int]] = None) -> None:
        self.n_images += 1
        gt_cls = np.asarray(gt_cls, int).reshape(-1)
        pred_cls = np.asarray(pred_cls, int).reshape(-1)
        pred_conf = np.asarray(pred_conf, float).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
        pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
        use_masks = (gt_masks is not None and pred_masks is not None and image_hw is not None
                     and all(m is not None for m in pred_masks))
        if use_masks:
            self.n_images_with_masks += 1
            hw = eval_hw(image_hw[0], image_hw[1], self.mask_max_side)
            gm_all = np.array([downsample_mask(m, hw) for m in gt_masks], bool).reshape(len(gt_masks), *hw)
            pm_all = np.array([downsample_mask(m, hw) for m in pred_masks], bool).reshape(len(pred_masks), *hw)
        for c in set(gt_cls.tolist()) | set(pred_cls.tolist()):
            gi = np.where(gt_cls == c)[0]
            pi = np.where(pred_cls == c)[0]
            st = self._s(int(c))
            st.n_gt += len(gi)
            iou_b = box_iou(pred_boxes[pi], gt_boxes[gi])
            st.tp_box.append(match(iou_b, pred_conf[pi]))
            st.conf.append(pred_conf[pi])
            if use_masks:
                st.n_gt_mask += len(gi)
                iou_m = mask_iou(pm_all[pi], gm_all[gi])
                st.tp_mask.append(match(iou_m, pred_conf[pi]))
                st.conf_mask.append(pred_conf[pi])
                # matched-pair mask IoU at the operating point: box-matched (IoU>=0.5) pairs, conf-greedy
                keep = pred_conf[pi] >= self.op_conf
                if keep.any() and len(gi):
                    sub_b = iou_b[keep]
                    sub_m = iou_m[keep]
                    order = np.argsort(-pred_conf[pi][keep], kind="stable")
                    taken = np.zeros(len(gi), bool)
                    for p in order:
                        cand = np.where(~taken & (sub_b[p] >= 0.5), sub_b[p], -1.0)
                        g = int(np.argmax(cand))
                        if cand[g] >= 0:
                            taken[g] = True
                            st.matched_mask_ious.append(float(sub_m[p, g]))
                gu = gm_all[gi].any(0) if len(gi) else np.zeros(hw, bool)
                pu = pm_all[pi][keep].any(0) if keep.any() else np.zeros(hw, bool)
                self.pix_inter[int(c)] = self.pix_inter.get(int(c), 0.0) + float((gu & pu).sum())
                self.pix_union[int(c)] = self.pix_union.get(int(c), 0.0) + float((gu | pu).sum())

    def summary(self, classes: Optional[Sequence[int]] = None) -> dict:
        """Per-class and mean metrics. ``classes`` restricts the mean (default: classes with ground truth)."""
        per: dict[str, dict] = {}
        for c, st in sorted(self.stats.items()):
            tp = np.concatenate(st.tp_box) if st.tp_box else np.zeros((0, 10), bool)
            conf = np.concatenate(st.conf) if st.conf else np.zeros(0)
            aps, pr = ap_per_threshold(tp, conf, st.n_gt)
            name = self.names[c] if 0 <= c < len(self.names) else str(c)
            d = {"n_gt": st.n_gt, "n_pred": int(len(conf)),
                 "box_ap50": _f(aps[0]), "box_ap50_95": _f(np.nanmean(aps)) if st.n_gt else None, **_r(pr)}
            if st.tp_mask:
                tpm = np.concatenate(st.tp_mask)
                confm = np.concatenate(st.conf_mask)
                apm, _ = ap_per_threshold(tpm, confm, st.n_gt_mask)
                d.update({"mask_ap50": _f(apm[0]), "mask_ap50_95": _f(np.nanmean(apm)) if st.n_gt_mask else None,
                          "mean_matched_mask_iou": _f(np.mean(st.matched_mask_ious)) if st.matched_mask_ious else None,
                          "n_matched_masks": len(st.matched_mask_ious),
                          "pixel_iou": _f(self.pix_inter.get(c, 0.0) / self.pix_union[c])
                          if self.pix_union.get(c) else None})
            per[name] = d
        sel = [c for c in (classes if classes is not None else self.stats) if c in self.stats and self.stats[c].n_gt]
        sel_names = [self.names[c] for c in sel]

        def mean(key: str) -> Optional[float]:
            vals = [per[n][key] for n in sel_names if per[n].get(key) is not None]
            return _f(np.mean(vals)) if vals else None

        return {"n_images": self.n_images, "n_images_with_masks": self.n_images_with_masks, "per_class": per,
                "mean_over": sel_names, "map50": mean("box_ap50"), "map50_95": mean("box_ap50_95"),
                "mask_map50": mean("mask_ap50"), "mask_map50_95": mean("mask_ap50_95")}


def _f(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def _r(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
