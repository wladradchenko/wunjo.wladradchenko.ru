import shutil
import uuid

import os

import cv2


def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = [] 
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break 
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames


def save_video_with_audio(video, audio, save_path):
    file_name = str(uuid.uuid4())+'.mp4'
    save_file = os.path.join(save_path, file_name)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -c:v libx264 -c:a aac -crf 23 -preset medium -movflags +faststart "%s"' % (video, audio, save_file)
    os.system(cmd)
    return file_name


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}.{:03}".format(int(hours), int(minutes), int(seconds), int((seconds % 1) * 1000))


def cut_start_video(video, video_start):
    hms_format = seconds_to_hms(video_start)
    print("Video will start from %s ".format(hms_format))
    new_video = f"{video}_cut.mp4"
    cmd = f"ffmpeg -ss {hms_format} -i {video} -c copy {new_video}"
    os.system(cmd)
    return new_video
