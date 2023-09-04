import shutil
import uuid

import os

import cv2
import numpy as np


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
    # if audio length than video, when will show video last frame
    # cmd = r'ffmpeg -y -i "%s" -i "%s" -c:v libx264 -c:a aac -crf 23 -preset medium -movflags +faststart "%s"' % (video, audio, save_file)
    # if audio length than video, when will audio cut
    cmd = r'ffmpeg -y -i "%s" -i "%s" -c:v libx264 -c:a aac -crf 23 -preset medium -movflags +faststart -shortest "%s"' % (video, audio, save_file)
    os.system(cmd)
    return file_name


def save_video_from_frames(frame_names, save_path, fps):
    # frames has to have name from 0
    frame_path = os.path.join(save_path, frame_names)
    file_name = str(uuid.uuid4())+'.mp4'
    save_file = os.path.join(save_path, file_name)
    cmd = f"ffmpeg -framerate {fps} -i {frame_path} -c:v libx264 -pix_fmt yuv420p {save_file}"
    os.system(cmd)
    return file_name


def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cmd = f'ffmpeg -i "{video_path}" "{output_folder}/%d.png"'
    os.system(cmd)


def extract_audio_from_video(video_path, save_path):
    file_name = str(uuid.uuid4()) + '.wav'
    save_file = os.path.join(save_path, file_name)
    cmd = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{save_file}" -y'
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


def check_media_type(file_path):
    # Initialize a VideoCapture object
    cap = cv2.VideoCapture(file_path)

    # Count the number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clean up
    cap.release()

    # Check the number of frames to determine the type of media
    if frame_count > 1:
        return "animated"
    else:
        return "static"


def get_first_frame(file_path):
    """
    Get first frame from content
    :param file_path: file path
    :return: frame or NOne
    """
    type_file = check_media_type(file_path)
    if type_file == "static":
        # It's an image or GIF
        img = cv2.imread(file_path)
        if img is not None:
            return img
    elif type_file == "animated":
        # It's a video
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        if ret:
            return frame
        else:
            raise ValueError("Could not read the video file.")
    else:
        raise ValueError("Unsupported file format.")

    return None


def get_frames(video: str, rotate: int, crop: list, resize_factor: int):
    """
    Extract frames from a video, apply resizing, rotation, and cropping.

    :param video: path to the video file
    :param rotate: number of 90-degree rotations
    :param crop: list with cropping coordinates [y1, y2, x1, x2]
    :param resize_factor: factor by which the frame should be resized
    :return: list of processed frames, fps of the video
    """
    print("Start reading video")

    video_stream = cv2.VideoCapture(video)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []

    try:
        while True:
            still_reading, frame = video_stream.read()

            if not still_reading:
                break

            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            for _ in range(rotate):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            x2 = x2 if x2 != -1 else frame.shape[1]
            y2 = y2 if y2 != -1 else frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    finally:
        video_stream.release()

    print(f"Number of frames available for inference: {len(full_frames)}")

    return full_frames, fps
