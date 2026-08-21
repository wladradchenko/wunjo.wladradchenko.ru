"""Faces described by the editor, not found again by the plugin.

The Face Region effect already carries where the face is on every frame — it was
detected once, in the app, and the user can correct it by hand. So nothing here
searches for faces. What the three engines still need beyond a rectangle:

* five landmark points, to align a crop the way the swap model was trained;
* a 512-d identity vector of the *source* photo, which is literally the second
  input of ``faceswap.onnx`` and cannot be derived from a rectangle.

The points come from the landmark model this plugin already ships for Live
Portrait, run on the small region crop. The identity vector is computed once,
when a face preset is registered, and travels inside the preset afterwards.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

# The template arcface was trained on, in 112x112 space.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def pts_from_box(crop: Optional[Dict], shape) -> np.ndarray:
    """Five seed points laid out inside the face box, in ordinary proportions.

    Good enough to aim the landmark model, which then predicts the real points.
    Order is the one every arcface-derived model expects: eyes, nose, mouth.
    """
    height, width = shape[:2]
    x, y, w, h = 0.0, 0.0, 0.0, 0.0
    if crop:
        x, y = float(crop.get("x", 0)), float(crop.get("y", 0))
        w, h = float(crop.get("width", 0)), float(crop.get("height", 0))
    if w <= 1 or h <= 1:
        x, y, w, h = width * 0.25, height * 0.2, width * 0.5, height * 0.6
    return np.array(
        [
            [x + 0.30 * w, y + 0.38 * h],
            [x + 0.70 * w, y + 0.38 * h],
            [x + 0.50 * w, y + 0.55 * h],
            [x + 0.35 * w, y + 0.75 * h],
            [x + 0.65 * w, y + 0.75 * h],
        ],
        dtype=np.float32,
    )


def norm_crop(img: np.ndarray, kps: np.ndarray, size: int) -> (np.ndarray, np.ndarray):
    """Align a face onto the arcface template and return the crop and matrix.

    The template is not simply scaled to the requested size. Arcface's own is
    112, and the swap model's 128 crop is that same 112 template shifted 8 px
    right inside a wider frame — not a 128/112 zoom. Scale it as if it were and
    the face lands too large and off-centre: the model then rebuilds a face that
    does not fit the hole it came from, and the frame comes back barely changed.
    """
    if size % 112 == 0:
        ratio, offset_x = float(size) / 112.0, 0.0
    else:
        ratio, offset_x = float(size) / 128.0, 8.0 * (float(size) / 128.0)
    dst = ARCFACE_TEMPLATE * ratio
    dst[:, 0] += offset_x
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(kps, dtype=np.float32), dst, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("could not align the face crop")
    return cv2.warpAffine(img, matrix, (size, size), borderValue=0.0), matrix


class BoxFace:
    """What the engines call a "face": a box, its points, and nothing hidden."""

    def __init__(self, bbox, kps, embedding: Optional[np.ndarray] = None):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = np.asarray(kps, dtype=np.float32)
        # kept under the donor's name so its cropping code needs no changes
        self.landmark_2d_106 = self.kps
        self.normed_embedding = embedding
        self.mean_distance = 0.0
        self.dynamic_threshold = 1.0
        self.det_score = 1.0

    def get(self, key, default=None):
        return {"bbox": self.bbox, "kps": self.kps}.get(key, default)

    def __getitem__(self, key):
        return {"bbox": self.bbox, "kps": self.kps}[key]


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4) -> List[int]:
    """Plain greedy non-maximum suppression over score-sorted boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        overlap = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = overlap / (areas[i] + areas[order[1:]] - overlap)
        order = order[1:][iou <= threshold]
    return keep


class FaceDetector:
    """SCRFD (``det_10g.onnx``) driven directly, without the insightface package.

    Both donors were trained against this detector: Wav2Lip is fed its box and
    the swap model is aligned on its five points. Neither is an arbitrary
    rectangle the caller may choose — the shape of the box is part of the input
    the model expects — so the detector runs here rather than being replaced by
    the effect's region. What the region decides is *which* face, not where it is.
    """

    def __init__(self, model_path: str, providers: Optional[List[str]] = None, input_size: int = 640,
                 threshold: float = 0.5):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = int(input_size)
        self.threshold = float(threshold)
        self.strides = (8, 16, 32)
        self.num_anchors = 2

    def detect(self, img_bgr: np.ndarray) -> List[tuple]:
        """Every face in the frame as ``(bbox, kps, score)``, in image pixels."""
        size = self.input_size
        height, width = img_bgr.shape[:2]
        scale = min(size / max(1.0, width), size / max(1.0, height))
        resized = cv2.resize(img_bgr, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        fmc = len(self.strides)
        all_boxes, all_points, all_scores = [], [], []
        for index, stride in enumerate(self.strides):
            scores = outputs[index].reshape(-1)
            keep = np.where(scores >= self.threshold)[0]
            if keep.size == 0:
                continue
            # the grid this stride predicts from, one point per anchor
            cells = size // stride
            centers = np.stack(np.mgrid[:cells, :cells][::-1], axis=-1).astype(np.float32) * stride
            centers = np.repeat(centers.reshape((-1, 2)), self.num_anchors, axis=0)[keep]
            # distances from the anchor, in stride units
            deltas = outputs[index + fmc].reshape((-1, 4))[keep] * stride
            points = outputs[index + fmc * 2].reshape((-1, 5, 2))[keep] * stride
            all_boxes.append(
                np.stack(
                    [
                        centers[:, 0] - deltas[:, 0],
                        centers[:, 1] - deltas[:, 1],
                        centers[:, 0] + deltas[:, 2],
                        centers[:, 1] + deltas[:, 3],
                    ],
                    axis=-1,
                )
            )
            all_points.append(points + centers[:, None, :])
            all_scores.append(scores[keep])

        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes) / scale
        points = np.concatenate(all_points) / scale
        scores = np.concatenate(all_scores)
        return [(boxes[i], points[i], float(scores[i])) for i in nms(boxes, scores)]


class RegionFaces:
    """The detector, pointed at the face the effect's rectangle picked out.

    Kept call-compatible with what the donor engines expect, so their cropping
    and pasting code is untouched. When the detector finds nothing — the head is
    turned away, or the shot is too dark — the rectangle answers on its own, with
    points guessed from the landmark model, so a lost frame is a slightly worse
    frame rather than a hole.
    """

    def __init__(self, landmark_model: str = "", device: str = "cpu", refine: bool = False,
                 detector_model: str = ""):
        self.landmark_model = landmark_model if refine and os.path.isfile(landmark_model) else ""
        self.device = device
        self._runner = None
        if not detector_model and landmark_model:
            detector_model = os.path.join(os.path.dirname(landmark_model), "buffalo_l", "det_10g.onnx")
        self._detector: Optional[FaceDetector] = None
        self._detector_model = detector_model if os.path.isfile(detector_model or "") else ""

    @property
    def detector(self) -> Optional[FaceDetector]:
        if self._detector is None and self._detector_model:
            providers = ["CUDAExecutionProvider"] if self.device != "cpu" else ["CPUExecutionProvider"]
            self._detector = FaceDetector(self._detector_model, providers)
        return self._detector

    @staticmethod
    def _pick(faces: List[tuple], crop: Optional[Dict]) -> tuple:
        """The detected face the region asks for: the one it sits on."""
        if not crop or float(crop.get("width", 0)) <= 1:
            return max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
        center_x = float(crop.get("x", 0)) + float(crop.get("width", 0)) / 2
        center_y = float(crop.get("y", 0)) + float(crop.get("height", 0)) / 2
        inside = [f for f in faces if f[0][0] <= center_x <= f[0][2] and f[0][1] <= center_y <= f[0][3]]
        return min(
            inside or faces,
            key=lambda f: ((f[0][0] + f[0][2]) / 2 - center_x) ** 2 + ((f[0][1] + f[0][3]) / 2 - center_y) ** 2,
        )

    def _refine(self, img_bgr: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Turn seed points into real ones with the landmark model, if we have it."""
        if not self.landmark_model:
            return seed
        if self._runner is None:
            from portrait_animation.utils.human_landmark_runner import LandmarkRunner

            self._runner = LandmarkRunner(ckpt_path=self.landmark_model, onnx_provider=self.device, device_id=0)
            self._runner.warmup()
        pts = self._runner.run(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), seed)
        if isinstance(pts, dict):
            pts = pts.get("pts")
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < 203:
            return seed
        # 203-point layout, as the cropper reads it: eye rings and lip corners.
        left_eye = pts[[0, 6, 12, 18]].mean(axis=0)
        right_eye = pts[[24, 30, 36, 42]].mean(axis=0)
        mouth = sorted([pts[48], pts[66]], key=lambda p: p[0])
        # No nose index is documented here; the midpoint is close enough because
        # the alignment below fits robustly and leans on the four sure points.
        nose = ((left_eye + right_eye) / 2 + (mouth[0] + mouth[1]) / 2) / 2
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye
        return np.stack([left_eye, right_eye, nose, mouth[0], mouth[1]]).astype(np.float32)

    def get(self, img_bgr: np.ndarray, **kwargs) -> List[BoxFace]:
        crop = kwargs.get("crop") or {}
        detector = self.detector
        if detector is not None:
            faces = detector.detect(img_bgr)
            if faces:
                bbox, kps, _score = self._pick(faces, crop)
                return [BoxFace(bbox, kps)]

        # Nothing found: answer from the rectangle, with points the landmark
        # model guesses inside it. Worse than a detection, better than a hole.
        seed = pts_from_box(crop, img_bgr.shape)
        kps = self._refine(img_bgr, seed)
        if crop and float(crop.get("width", 0)) > 1:
            x, y = float(crop.get("x", 0)), float(crop.get("y", 0))
            bbox = [x, y, x + float(crop["width"]), y + float(crop.get("height", 0))]
        else:
            bbox = [float(kps[:, 0].min()), float(kps[:, 1].min()), float(kps[:, 0].max()), float(kps[:, 1].max())]
        return [BoxFace(bbox, kps)]

    # the donor calls these between frames; there is no state to keep now
    def prepare(self, *args, **kwargs):
        pass

    def warmup(self, *args, **kwargs):
        pass

    def clear(self):
        pass

    def update_center(self, crop=None):
        pass

    @staticmethod
    def calculate_distance(first, second):
        return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


class ArcFace:
    """The identity vector faceswap.onnx asks for. Run once, per registered photo."""

    def __init__(self, model_path: str, providers: List[str]):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(self.session.get_inputs()[0].shape[2] or 112)

    def embed(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        aligned, _ = norm_crop(img_bgr, kps, self.size)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        embedding = self.session.run(None, {self.input_name: blob})[0].flatten()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding


class InSwapper:
    """faceswap.onnx driven directly, without the insightface package."""

    def __init__(self, model_path: str, providers: List[str]):
        import onnx
        import onnxruntime
        from onnx import numpy_helper

        graph = onnx.load(model_path).graph
        # the model carries the matrix that maps an identity vector into its own
        # latent space as its last initializer
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.size = 128

    def get(self, img: np.ndarray, target_face: BoxFace, source_embedding, paste_back: bool = True):
        # callers hold either the vector itself or a face carrying it
        source_embedding = getattr(source_embedding, "normed_embedding", source_embedding)
        aligned, matrix = norm_crop(img, target_face.kps, self.size)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255.0, (self.size, self.size), (0.0, 0.0, 0.0), swapRB=True)
        latent = np.asarray(source_embedding, dtype=np.float32).reshape((1, -1)) @ self.emap
        latent /= np.linalg.norm(latent)
        pred = self.session.run(None, {self.input_names[0]: blob, self.input_names[1]: latent.astype(np.float32)})[0]
        swapped = np.clip(255.0 * pred.transpose((0, 2, 3, 1))[0], 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return swapped, matrix

        # Put the crop back where it came from, fading out at the edges so the
        # seam does not show.
        inverse = cv2.invertAffineTransform(matrix)
        height, width = img.shape[:2]
        warped = cv2.warpAffine(swapped, inverse, (width, height), borderValue=0.0)
        mask = np.ones((self.size, self.size), dtype=np.float32)
        border = max(2, self.size // 16)
        mask[:border, :] = mask[-border:, :] = mask[:, :border] = mask[:, -border:] = 0
        mask = cv2.warpAffine(mask, inverse, (width, height), borderValue=0.0)
        blur = max(3, int(0.05 * np.sqrt((mask > 0).sum())) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)[:, :, None]
        return (warped * mask + img * (1 - mask)).astype(np.uint8)
