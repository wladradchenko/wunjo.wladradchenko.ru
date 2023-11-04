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

AVATAR_FOLDER = os.path.join(MEDIA_FOLDER, "avatar")
if not os.path.exists(AVATAR_FOLDER):
    os.makedirs(AVATAR_FOLDER)

DEEPFAKE_MODEL_FOLDER = os.path.join(MEDIA_FOLDER, 'deepfake')
if not os.path.exists(DEEPFAKE_MODEL_FOLDER):
    os.makedirs(DEEPFAKE_MODEL_FOLDER)

VOICE_FOLDER = os.path.join(MEDIA_FOLDER, 'voice')
if not os.path.exists(VOICE_FOLDER):
    os.makedirs(VOICE_FOLDER)

RTVC_VOICE_FOLDER = os.path.join(MEDIA_FOLDER, 'rtvc')
if not os.path.exists(RTVC_VOICE_FOLDER):
    os.makedirs(RTVC_VOICE_FOLDER)

CUSTOM_VOICE_FOLDER = os.path.join(MEDIA_FOLDER, 'user_trained_voice')
if not os.path.exists(CUSTOM_VOICE_FOLDER):
    os.makedirs(CUSTOM_VOICE_FOLDER)

TMP_FOLDER = os.path.join(MEDIA_FOLDER, 'tmp')
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

# CONTENT FOLDERS
CONTENT_FOLDER = os.path.join(MEDIA_FOLDER, "content")
if not os.path.exists(CONTENT_FOLDER):
    os.makedirs(CONTENT_FOLDER)

CONTENT_MEDIA_EDIT_FOLDER = os.path.join(CONTENT_FOLDER, "media_edit")
if not os.path.exists(CONTENT_MEDIA_EDIT_FOLDER):
    os.makedirs(CONTENT_MEDIA_EDIT_FOLDER)

CONTENT_AUDIO_SEPARATOR_FOLDER = os.path.join(CONTENT_FOLDER, "audio_separator")
if not os.path.exists(CONTENT_AUDIO_SEPARATOR_FOLDER):
    os.makedirs(CONTENT_AUDIO_SEPARATOR_FOLDER)

CONTENT_DIFFUSER_FOLDER = os.path.join(CONTENT_FOLDER, "diffuser")
if not os.path.exists(CONTENT_DIFFUSER_FOLDER):
    os.makedirs(CONTENT_DIFFUSER_FOLDER)

CONTENT_RETOUCH_FOLDER = os.path.join(CONTENT_FOLDER, "retouch")
if not os.path.exists(CONTENT_RETOUCH_FOLDER):
    os.makedirs(CONTENT_RETOUCH_FOLDER)

CONTENT_FACE_SWAP_FOLDER = os.path.join(CONTENT_FOLDER, "face_swap")
if not os.path.exists(CONTENT_FACE_SWAP_FOLDER):
    os.makedirs(CONTENT_FACE_SWAP_FOLDER)

CONTENT_ANIMATION_TALK_FOLDER = os.path.join(CONTENT_FOLDER, "animation_talk")
if not os.path.exists(CONTENT_ANIMATION_TALK_FOLDER):
    os.makedirs(CONTENT_ANIMATION_TALK_FOLDER)

CONTENT_SPEECH_FOLDER = os.path.join(CONTENT_FOLDER, "speech")
if not os.path.exists(CONTENT_SPEECH_FOLDER):
    os.makedirs(CONTENT_SPEECH_FOLDER)
