import os

HOME_FOLDER = os.path.expanduser('~')

MEDIA_FOLDER = os.path.join(HOME_FOLDER, '.wunjo')
if not os.path.exists(MEDIA_FOLDER):
    os.makedirs(MEDIA_FOLDER)

WAVES_FOLDER = os.path.join(MEDIA_FOLDER, 'waves')
if not os.path.exists(WAVES_FOLDER):
    os.makedirs(WAVES_FOLDER)

AVATAR_FOLDER = os.path.join(MEDIA_FOLDER, "avatar")
if not os.path.exists(AVATAR_FOLDER):
    os.makedirs(AVATAR_FOLDER)

TALKER_FOLDER = os.path.join(MEDIA_FOLDER, 'video')
if not os.path.exists(TALKER_FOLDER):
    os.makedirs(TALKER_FOLDER)

TALKER_MODEL_FOLDER = os.path.join(MEDIA_FOLDER, 'talker')
if not os.path.exists(TALKER_MODEL_FOLDER):
    os.makedirs(TALKER_MODEL_FOLDER)

VOICE_FOLDER = os.path.join(MEDIA_FOLDER, 'voice')
if not os.path.exists(VOICE_FOLDER):
    os.makedirs(VOICE_FOLDER)

TMP_FOLDER = os.path.join(MEDIA_FOLDER, 'tmp')
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)
