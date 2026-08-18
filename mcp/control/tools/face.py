"""Built-in face detection (RetinaFace ONNX, no venv) and face-driven blur.

The detector runs in the background per bin clip and stores results as
<projectDataFolder>/faces/<binId>.json; coordinates are normalized 0..1.
blur_face applies the approved Motion Tracker (opencv.tracker) effect with
keyframes pre-filled from the detected track — no analysis pass, no freeze.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    def _app(ctx: Context):
        return helpers.get_resolve(ctx)._app

    @mcp.tool()
    def set_face_detection(ctx: Context, bin_id: str, enabled: bool = True,
                           first_frame: int = -1, last_frame: int = -1) -> str:
        """Enable/disable face detection on a bin clip (analysis runs in background).

        Args:
            bin_id: Media pool clip id.
            enabled: True to analyse and show face boxes, False to turn off.
            first_frame, last_frame: The part to analyse before the rest, in
                frames of the source. Give the range of the timeline clip you
                are working on: a ten-minute file trimmed to five seconds is
                otherwise analysed from its first frame, and the piece in the
                edit gets its faces last. The whole file is still covered.
        """
        try:
            ok = _app(ctx)._call("scriptSetFaceDetection", bin_id, enabled,
                                 int(first_frame), int(last_frame))
            if not ok:
                return f"ERROR: could not toggle face detection for bin_id={bin_id}"
            return f"Face detection {'enabled' if enabled else 'disabled'} for bin_id={bin_id}. " \
                   "Poll get_face_detection_status until complete."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def get_face_detection_status(ctx: Context, bin_id: str) -> str:
        """Report whether face analysis is enabled/complete and how many frames are done.

        Args:
            bin_id: Media pool clip id.
        """
        try:
            status = _app(ctx)._call("scriptGetFaceDetectionStatus", bin_id)
            return json.dumps(dict(status), default=str) if status else "{}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def get_faces_at_frame(ctx: Context, bin_id: str, position: int) -> str:
        """List faces detected at a source frame (normalized 0..1 rects).

        Args:
            bin_id: Media pool clip id.
            position: Source frame (0-based) within the clip.

        Returns a JSON list; index into it for blur_face's face_index.
        """
        try:
            faces = _app(ctx)._call("scriptGetFacesAtFrame", bin_id, position) or []
            faces = [dict(f) for f in faces]
            if not faces:
                return "No faces at this frame (not analysed yet, or none present)."
            return json.dumps(faces, default=str)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def apply_face_effect(ctx: Context, clip_id: int, face_index: int = 0, at_frame: int = 0,
                          effect: str = "opencv.tracker", params: dict | None = None) -> str:
        """Select a detected face and apply the chosen effect, tracked across the clip.

        Picks the face at (source at_frame, face_index), builds a motion track from
        the detection data and applies `effect` following it. Default is the "Hide
        Face" Motion Tracker (opencv.tracker); pass params to pick the mode/strength
        or override defaults.

        Args:
            clip_id: Timeline clip id (the instance to add the effect to).
            face_index: Which face at that frame (see get_faces_at_frame).
            at_frame: Source frame to pick the face from.
            effect: Effect id. Default "opencv.tracker" (blur/pixelate/fill on the face).
            params: Effect parameters to override defaults. For opencv.tracker:
                blur_type (0=blur, 2=pixelate, 3=fill), blur (strength), shape_width.
        """
        try:
            import json as _json
            ok = _app(ctx)._call("scriptApplyFaceEffect", clip_id, face_index, at_frame,
                                  effect, _json.dumps(params or {}))
            if not ok:
                return "ERROR: could not apply face effect (no face at that frame, or clip has no detection data)."
            return f"Applied '{effect}' on face #{face_index} of clip {clip_id} (tracked from frame {at_frame})."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
