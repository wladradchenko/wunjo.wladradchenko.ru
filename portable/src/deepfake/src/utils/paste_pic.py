import cv2, os
import numpy as np
from tqdm import tqdm
import uuid

from src.utils.videoio import VideoManipulation

def paste_pic(video_path, pic_path, crop_info, new_audio_path, video_save_dir, video_format = '.mp4', preprocess = "crop", pic_path_type = "static"):

    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path_type == "static":
        # loader for first frame
        full_img = cv2.imread(pic_path)
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(pic_path)
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break 
            break 
        full_img = frame

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)
    
    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        clx, cly, crx, cry = crop_info[1]
        oy1, oy2, ox1, ox2 = cly, cry, clx, crx

    tmp_path = os.path.join(video_save_dir, str(uuid.uuid4())+video_format)

    if video_format == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_format == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if preprocess == "resize":
        frame_w = ox2 - ox1
        frame_h = oy2 - oy1
    else:
        frame_h = full_img.shape[0]
        frame_w = full_img.shape[1]

    out_tmp = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))
    for crop_frame in tqdm(crop_frames, 'seamlessClone:'):
        p = cv2.resize(crop_frame.astype(np.uint8), (crx-clx, cry - cly)) 

        mask = 255*np.ones(p.shape, p.dtype)
        location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
        gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
        if preprocess == "resize":
            gen_img = gen_img[oy1:oy2, ox1:ox2]

        out_tmp.write(gen_img)

    out_tmp.release()

    new_video_name = VideoManipulation.save_video_with_audio(tmp_path, new_audio_path, video_save_dir)
    os.remove(tmp_path)

    return new_video_name