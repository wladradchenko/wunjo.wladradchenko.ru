"""Constants for the vendored api package (adapted for Wunjo Make)."""

import os

# Where the editor listens. ScriptingServer takes this name under
# XDG_RUNTIME_DIR; a second instance appends its pid. Set WUNJO_SOCKET to an
# absolute path to talk to one particular copy.
SOCKET_NAME = "online.wunjo.make.scripting"
SOCKET_ENV_VAR = "WUNJO_SOCKET"

# Track types (Resolve uses these strings)
TRACK_VIDEO = "video"
TRACK_AUDIO = "audio"
TRACK_SUBTITLE = "subtitle"

# Marker / guide categories (Kdenlive default palette)
MARKER_PURPLE = 0
MARKER_BLUE = 1
MARKER_GREEN = 2
MARKER_YELLOW = 3
MARKER_ORANGE = 4
MARKER_RED = 5

# Resolve marker color name → Kdenlive category mapping
MARKER_COLOR_MAP = {
    "Purple": MARKER_PURPLE,
    "Blue": MARKER_BLUE,
    "Cyan": MARKER_BLUE,
    "Green": MARKER_GREEN,
    "Yellow": MARKER_YELLOW,
    "Orange": MARKER_ORANGE,
    "Red": MARKER_RED,
    # Resolve extras — map to nearest
    "Fuchsia": MARKER_PURPLE,
    "Rose": MARKER_RED,
    "Lavender": MARKER_PURPLE,
    "Sky": MARKER_BLUE,
    "Mint": MARKER_GREEN,
    "Lemon": MARKER_YELLOW,
    "Sand": MARKER_ORANGE,
    "Cocoa": MARKER_ORANGE,
    "Cream": MARKER_YELLOW,
}

# Reverse map: category → first matching color name
MARKER_CATEGORY_TO_COLOR = {
    MARKER_PURPLE: "Purple",
    MARKER_BLUE: "Blue",
    MARKER_GREEN: "Green",
    MARKER_YELLOW: "Yellow",
    MARKER_ORANGE: "Orange",
    MARKER_RED: "Red",
}

# Scene duration at 25 fps (5 seconds)
SCENE_DURATION_FRAMES = 125

# Default transition duration (frames at 25 fps)
DEFAULT_MIX_DURATION = 13  # ~0.5s

# Project defaults
DEFAULT_FPS = 25.0
DEFAULT_WIDTH = 1536
DEFAULT_HEIGHT = 864

# ── Resolve Export Type constants ──────────────────────────────────────
# Used by timeline.Export() and resolve.EXPORT_* attributes
EXPORT_AAF = 0
EXPORT_DRT = 1
EXPORT_EDL = 2
EXPORT_FCP_7_XML = 3
EXPORT_FCPXML_1_3 = 4
EXPORT_FCPXML_1_4 = 5
EXPORT_FCPXML_1_5 = 6
EXPORT_FCPXML_1_6 = 7
EXPORT_FCPXML_1_7 = 8
EXPORT_FCPXML_1_8 = 9
EXPORT_FCPXML_1_9 = 10
EXPORT_FCPXML_1_10 = 11
EXPORT_HDL = 12
EXPORT_TEXT_CSV = 13
EXPORT_TEXT_TAB = 14
EXPORT_DOLBY_VISION_VER_2_9 = 15
EXPORT_DOLBY_VISION_VER_4_0 = 16
EXPORT_OTIO = 17

# Export AAF sub-types
EXPORT_AAF_NEW = 0
EXPORT_AAF_EXISTING = 1

# Export EDL sub-types
EXPORT_CDL = 0
EXPORT_SDL = 1
EXPORT_MISSING_CLIPS = 2
