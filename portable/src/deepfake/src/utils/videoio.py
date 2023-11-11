import shutil
import uuid
import time

import os
import subprocess

import cv2
import random
from tqdm import tqdm
import numpy as np
from PIL import Image


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
    file_name = str(uuid.uuid4()) + '.mp4'
    save_file = os.path.join(save_path, file_name)
    if os.path.exists(audio):
        # If there is an audio file, include it in the ffmpeg command
        cmd = r'ffmpeg -y -i "%s" -i "%s" -c:v libx264 -c:a aac -crf 23 -preset medium -movflags +faststart -shortest "%s"' % (video, audio, save_file)
    else:
        # If there is no audio file, omit the audio input and codec options
        cmd = r'ffmpeg -y -i "%s" -c:v libx264 -crf 23 -preset medium -movflags +faststart "%s"' % (video, save_file)
    if os.environ.get('DEBUG', 'False') == 'True':
        # not silence run
        os.system(cmd)
    else:
        # silence run
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return file_name


def save_video_from_frames(frame_names, save_path, fps, alternative_save_path=None):
    # frames has to have name from 0
    frame_path = os.path.join(save_path, frame_names)
    file_name = str(uuid.uuid4())+'.mp4'
    if alternative_save_path:
        save_file = os.path.join(alternative_save_path, file_name)
    else:
        save_file = os.path.join(save_path, file_name)
    cmd = f'ffmpeg -framerate {fps} -i {frame_path} -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {save_file}'
    if os.environ.get('DEBUG', 'False') == 'True':
        # not silence run
        os.system(cmd)
    else:
        # silence run
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return file_name


def video_to_frames(video_path, output_folder, start_seconds=0, end_seconds=None, extract_nth_frame=1, reduce_size=1):
    start_hms = seconds_to_hms(start_seconds)
    if end_seconds is not None:
        select_cmd = f"select=between(t\\,{start_seconds}\\,{end_seconds})*not(mod(n\\,{extract_nth_frame}))"
    else:
        select_cmd = f"select=not(mod(n\\,{extract_nth_frame}))"
    # Add the scale filter to reduce frame size by a factor
    vf_cmd = f"{select_cmd},scale=iw/{int(reduce_size)}:ih/{int(reduce_size)}" if reduce_size > 1 else select_cmd
    cmd = f'ffmpeg -ss {start_hms} -i "{video_path}" -vf "{vf_cmd}" -vsync vfr "{output_folder}/%d.png"'
    if os.environ.get('DEBUG', 'False') == 'True':
        # not silence run
        os.system(cmd)
    else:
        # silence run
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_audio_from_video(video_path, save_path):
    # Check if the file is a GIF
    try:
        with Image.open(video_path) as img:
            if img.format == 'GIF':
                print(f"Skipping audio extraction because the file is a GIF")
                return None
    except Exception as e:
        print(f"Unable to determine image format: {e}")

    # If not a GIF, proceed with audio extraction
    file_name = str(uuid.uuid4()) + '.wav'
    save_file = os.path.join(save_path, file_name)
    cmd = f'ffmpeg -i "{video_path}" -q:a 0 -map a? "{save_file}" -y'
    if os.environ.get('DEBUG', 'False') == 'True':
        # not silence run
        os.system(cmd)
    else:
        # silence run
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return file_name


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}.{:03}".format(int(hours), int(minutes), int(seconds), int((seconds % 1) * 1000))


def cut_start_video(video, video_start, video_end):
    if video_start == video_end:
        return video
    time.sleep(5)
    hms_start_format = seconds_to_hms(video_start)
    hms_end_format = seconds_to_hms(video_end)
    print(f"Video will start from {hms_start_format} and end at {hms_end_format}")
    new_video = f"{video}_cut.mp4"
    cmd = f"ffmpeg -y -ss {hms_start_format} -to {hms_end_format} -i {video} -c copy {new_video}"
    if os.environ.get('DEBUG', 'False') == 'True':
        # not silence run
        os.system(cmd)
    else:
        # silence run
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def get_first_frame(file_path, source_current_time: float = 0):
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
        # Get the frames per second of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the frame number
        num_frame = int(fps * source_current_time)
        # Set the video capture to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)

        ret, frame = cap.read()
        if ret:
            return frame
        else:
            raise ValueError("Could not read the video file.")
    else:
        raise ValueError("Unsupported file format.")

    return None


def save_frames(video: str, output_dir: str, rotate: int, crop: list, resize_factor: int):
    """
    Extract frames from a video, apply resizing, rotation, and cropping, and save them to an output directory.

    :param video: path to the video file
    :param output_dir: path to the directory where frames should be saved
    :param rotate: number of 90-degree rotations
    :param crop: list with cropping coordinates [y1, y2, x1, x2]
    :param resize_factor: factor by which the frame should be resized
    :return: fps of the video, path to the directory containing frames
    """
    print("Start reading video")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    video_stream = cv2.VideoCapture(video)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    try:
        while True:
            still_reading, frame = video_stream.read()

            if not still_reading:
                break

            if resize_factor > 1:
                frame = cv2.resize(frame, (int(frame.shape[1] // resize_factor), int(frame.shape[0] // resize_factor)))

            for _ in range(rotate):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            x2 = x2 if x2 != -1 else frame.shape[1]
            y2 = y2 if y2 != -1 else frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            # Save the frame to the output directory
            frame_filename = os.path.join(output_dir, f'frame{frame_count:04}.png')
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

    finally:
        video_stream.release()

    print(f"Number of frames saved: {frame_count}")

    return fps, output_dir



def encrypted(video_path: str, save_dir: str, fn: int = 0):

    name = str(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    media_type = check_media_type(video_path)

    if media_type == "animated":
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

            # Put the text onto the dummy frame
            font_scale = int(max(src_w, src_h) // 600)
            font_thickness = int(font_scale // 0.5)

            # Define text position
            text_x = src_w // 4
            text_y = src_h // 2

            # Get the region where the text will be placed
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            region = src_frame[text_y - text_size[1]:text_y, text_x:text_x + text_size[0]]

            # Compute the mean color of the region
            mean_color = cv2.mean(region)[:3]

            # Compute the contrasting color
            contrast_color = tuple([255 - int(x) for x in mean_color])

            # Adjusting the coordinates to correctly position the text
            height, width = src_frame.shape[:2]

            # Get the size of the text
            (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Center positions
            center_x = width // 2
            center_y = height // 2

            # Define potential positions with padding along the edges (adjust as necessary)
            positions = [
                (0, text_height),
                (0, center_y + text_height // 2),
                (center_x - text_width // 2, text_height),
                (width - text_width, text_height),
                (0, height - baseline),
                (width - text_width, height - baseline),
                (center_x - text_width // 2, height - baseline),
                (width - text_width, center_y + text_height // 2),
                (int(width * 0.25) - text_width // 2, text_height),
                (int(width * 0.75) - text_width // 2, text_height),
                (int(width * 0.25) - text_width // 2, height - baseline),
                (int(width * 0.75) - text_width // 2, height - baseline),
                None, None, None
            ]

            # In each call to the method, select one random positions from the list
            selected_positions = random.sample(positions, 1)

            # Put text at the selected positions
            for pos in selected_positions:
                if pos is not None:
                    cv2.putText(sec_frame, name, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, contrast_color, font_thickness, cv2.LINE_AA)

            # Encryption for LSB 3 bits
            encrypted_img = (src_frame & 0b11111000) | (sec_frame >> 6 & 0b00000111)

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
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Delete the temporary image sequence folder
        shutil.rmtree(os.path.join(save_dir, 'enc'))

    else:
        # If the media is an image
        src_frame = cv2.imread(video_path)
        src_h, src_w = src_frame.shape[:2]

        # Load a dummy image to get the shape attributes
        sec_frame_original = np.zeros((src_h, src_w, 3), dtype=np.uint8)

        # Create a copy of the dummy frame
        sec_frame = sec_frame_original.copy()

        # Put the text onto the dummy frame
        font_scale = int(max(src_w, src_h) // 500)
        font_thickness = int(font_scale // 0.5)

        # Define text position
        text_x = src_w // 4
        text_y = src_h // 2

        # Get the region where the text will be placed
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        region = src_frame[text_y - text_size[1]:text_y, text_x:text_x + text_size[0]]

        # Compute the mean color of the region
        mean_color = cv2.mean(region)[:3]

        # Compute the contrasting color
        contrast_color = tuple([255 - int(x) for x in mean_color])

        # Adjusting the coordinates to correctly position the text
        height, width = src_frame.shape[:2]

        # Get the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              font_thickness)

        # Center positions
        center_x = width // 2
        center_y = height // 2

        # Define potential positions with padding along the edges (adjust as necessary)
        positions = [
            (0, text_height),
            (0, center_y + text_height // 2),
            (center_x - text_width // 2, text_height),
            (width - text_width, text_height),
            (0, height - baseline),
            (width - text_width, height - baseline),
            (center_x - text_width // 2, height - baseline),
            (width - text_width, center_y + text_height // 2),
            (int(width * 0.25) - text_width // 2, text_height),
            (int(width * 0.75) - text_width // 2, text_height),
            (int(width * 0.25) - text_width // 2, height - baseline),
            (int(width * 0.75) - text_width // 2, height - baseline),
        ]

        # In each call to the method, select one random positions from the list
        selected_positions = random.sample(positions, 1)

        # Put text at the selected positions
        for pos in selected_positions:
            cv2.putText(sec_frame, name, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, contrast_color, font_thickness, cv2.LINE_AA)

        # Encryption for LSB 3 bits
        encrypted_img = (src_frame & 0b11111000) | (sec_frame >> 6 & 0b00000111)

        # Save the encrypted image
        file_name = str(uuid.uuid4()) + '.png'
        file_path = os.path.join(save_dir, file_name)
        cv2.imwrite(file_path, encrypted_img)

    return file_name
