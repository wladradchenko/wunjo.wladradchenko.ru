import shutil
import uuid

import os

import cv2
from tqdm import tqdm
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


def encrypted(video_path: str, save_dir: str, device: str, fn: int = 0):
    # Video objects for src
    src = cv2.VideoCapture(video_path)
    src_w = int(src.get(3))
    src_h = int(src.get(4))
    src_fps = src.get(cv2.CAP_PROP_FPS)
    src_frame_cnt = src.get(cv2.CAP_PROP_FRAME_COUNT)

    # Load a dummy image to get the shape attributes
    sec_frame_original = np.zeros((src_h, src_w, 3), dtype=np.uint8)

    if not os.path.exists(os.path.join(save_dir, 'enc')):
        os.mkdir(os.path.join(save_dir, 'enc'))

    # Create a progress bar
    pbar = tqdm(total=int(src_frame_cnt), unit='frames')

    while True:
        ret, src_frame = src.read()

        if ret == False:
            break

        # Create a copy of the dummy frame
        sec_frame = sec_frame_original.copy()

        # Put the text "FAKE" onto the dummy frame
        font_scale = max(src_w, src_h) // 400
        font_thickness = int(font_scale // 0.5)

        # Define text position
        text_x = src_w // 4
        text_y = src_h // 2

        # Get the region where the text will be placed
        text_size = cv2.getTextSize(device, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        region = src_frame[text_y - text_size[1]:text_y, text_x:text_x + text_size[0]]

        # Compute the mean color of the region
        mean_color = cv2.mean(region)[:3]

        # Compute the contrasting color
        contrast_color = tuple([255 - int(x) for x in mean_color])

        # Adjusting the coordinates to correctly position the text
        height, width = src_frame.shape[:2]

        # Get the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(device, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Center positions
        center_x = width // 2
        center_y = height // 2

        # Position the text at different locations keeping it fully inside the bounds of the image
        cv2.putText(sec_frame, device, (0, text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, contrast_color,
                    font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (0, center_y + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    contrast_color, font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (center_x - text_width // 2, text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    contrast_color, font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (center_x - text_width // 2, center_y + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, contrast_color, font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (width - text_width, text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    contrast_color, font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (0, height - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, contrast_color,
                    font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (width - text_width, height - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    contrast_color, font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (center_x - text_width // 2, height - baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, contrast_color, font_thickness, cv2.LINE_AA)
        cv2.putText(sec_frame, device, (width - text_width, center_y + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, contrast_color, font_thickness, cv2.LINE_AA)

        # Encryption for LSB 3 bits
        encrypted_img = (src_frame & 0b11111000) | (sec_frame >> 5 & 0b00000111)

        fn = fn + 1
        cv2.imwrite(os.path.join(save_dir, "enc", "{}.png".format(fn)), encrypted_img)

        pbar.update(1)

    pbar.close()
    src.release()

    # Delete encrypted video if already exists
    file_name = str(uuid.uuid4())+'.mp4'
    file_path = os.path.join(save_dir, file_name)

    # Save the video using ffmpeg as a lossless video; frame rate is kept the same
    cmd = f"ffmpeg -framerate {src_fps} -i {os.path.join(save_dir, 'enc', '%d.png')} -c:v copy {file_path}"
    os.system(cmd)

    # Delete the temporary image sequence folder
    shutil.rmtree(f"rm -r {os.path.join(save_dir, 'enc')}")
    return file_path
