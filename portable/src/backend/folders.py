import os
import sys

# This configuration sets the path for the .wunjo folder based on the operating system.
# Set the HOME variable to the desired path before running this script.
# Example paths:
# - Windows: "N:\\Wunjo"
# - Linux: "/mnt/another_drive/Wunjo"
# - macOS: "/Volumes/AnotherDrive/Wunjo"
# Ensure the path exists and is writable on your system.

HOME = None  # Set this to the desired path.

if HOME:
    if sys.platform == 'win32':
        os.environ["USERPROFILE"] = HOME
    elif sys.platform in ['darwin', 'linux']:
        os.environ["HOME"] = HOME

HOME_FOLDER = os.path.expanduser('~')

MEDIA_FOLDER = os.path.join(HOME_FOLDER, '.wunjo')
if not os.path.exists(MEDIA_FOLDER):
    os.makedirs(MEDIA_FOLDER)

SETTING_FOLDER = os.path.join(MEDIA_FOLDER, 'setting')
if not os.path.exists(SETTING_FOLDER):
    os.makedirs(SETTING_FOLDER)

ALL_MODEL_FOLDER = os.path.join(MEDIA_FOLDER, 'all_models')
if not os.path.exists(ALL_MODEL_FOLDER):
    os.makedirs(ALL_MODEL_FOLDER)

TMP_FOLDER = os.path.join(MEDIA_FOLDER, 'tmp')
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

# CONTENT FOLDERS
CONTENT_FOLDER = os.path.join(MEDIA_FOLDER, "content")
if not os.path.exists(CONTENT_FOLDER):
    os.makedirs(CONTENT_FOLDER)

CONTENT_FACE_SWAP_FOLDER_NAME = "face_swap"
CONTENT_LIP_SYNC_FOLDER_NAME = "lip_sync"
CONTENT_REMOVE_BACKGROUND_FOLDER_NAME = "remove_background"
CONTENT_REMOVE_OBJECT_FOLDER_NAME = "remove_object"
CONTENT_ENHANCEMENT_FOLDER_NAME = "enhancement"
CONTENT_AVATARS_FOLDER_NAME = "avatars"
CONTENT_ANALYSIS_NAME = "analysis"
CONTENT_SEPARATOR_FOLDER_NAME = "separator"
