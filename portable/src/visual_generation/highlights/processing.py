import os
import re
import cv2
import json
import torch
import random
import subprocess
from typing import Literal
from transformers import AutoProcessor, AutoModelForImageTextToText
import whisper_timestamped as whisper

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    FLASH_ATTN_2_AVAILABLE = True
except ImportError:
    FLASH_ATTN_2_AVAILABLE = False


SYSTEM_MESSAGES = {
    "video_description": "You are a helpful assistant that can understand videos. Describe what type of video this is and what's happening in it.",
    "highlight_editor": "You are a highlight editor. List archetypal dramatic moments that would make compelling highlights if they appear in the video. Each moment should be specific enough to be recognizable but generic enough to potentially exist in other videos of this type.",
    "highlight_assistant": "You are a helpful visual-language assistant that can understand videos and edit. You are tasked helping the user to create highlight reels for videos. Highlights should be rare and important events in the video in question.",
    "highlight_analyzer": "You are a video highlight analyzer. Your role is to identify moments that have high dramatic value, focusing on displays of skill, emotion, personality, or tension. Be categorical and choose only the brightest moments. Compare video segments against provided example highlights to find moments with similar emotional impact and visual interest, even if the specific actions differ."
}


class SmolVLM2:
    def __init__(self, model_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct", device: str = "cuda", download_root: str = None):
        self.device = device
        self.download_root = download_root
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            _attn_implementation="flash_attention_2" if FLASH_ATTN_2_AVAILABLE else None
        )  # .to(self.device)  # CAN BE CPU???

    def generate(self, messages: list, max_new_tokens: int = 64):
        inputs = self.processor.apply_chat_template(
            messages[:8192],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        generated_output = self.processor.decode(generated_ids[0], skip_special_tokens=True).lower().split("assistant: ")[1]
        return generated_output

    @staticmethod
    def prompt_video_analyser(video_path: str, prompt: str = None):
        prompt = "What type of video is this and what's happening in it? Be specific about the content type and general activities you observe." if prompt is None or not isinstance(prompt, str) else prompt
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGES["video_description"]}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    def process_video_analyser(self, video_path: str, prompt: str = None) -> str:
        messages = self.prompt_video_analyser(video_path, prompt)
        response = self.generate(messages, max_new_tokens=512)
        return response

    @staticmethod
    def prompt_text_analyser(text: str, prompt: str = None):
        prompt = "Retell the text very briefly, what happens in it? Be specific about the content type and general activities you observe." if prompt is None or not isinstance(prompt, str) else prompt
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGES["video_description"]}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    def process_text_analyser(self, text: str, prompt: str = None) -> str:
        messages = self.prompt_text_analyser(text, prompt)
        response = self.generate(messages, max_new_tokens=512)
        return response

    @staticmethod
    def prompt_highlight(description: str, highlight_description: str = None, behaviour: int = None):
        prompts = {
            1: "List potential highlight moments to look for in this video:",
            2: "List dramatic moments that would make compelling highlights if they appear in the video. Each moment should be specific enough to be recognizable but generic enough to potentially exist in any video of this type:"
        }
        behaviour = 1 if highlight_description is None and behaviour is None else 2
        highlight_description = prompts[behaviour] if highlight_description is None else highlight_description
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGES["highlight_editor" if behaviour == 1 else "highlight_assistant"]}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"""Here is a description of a video:\n\n{description}\n\n{highlight_description}"""}]
            }
        ]

    def process_highlight(self, description: str, highlight_description: str = None) -> str:
        if highlight_description is not None:
            highlight_description = f"List potential {highlight_description.lower()} to look for in this video:"
        highlight_response_messages = self.prompt_highlight(description, highlight_description)
        highlight_response = self.generate(highlight_response_messages, max_new_tokens=256)
        return highlight_response

    @staticmethod
    def prompt_video_segment(video_segment_path: str, highlight_types: str):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGES["highlight_analyzer"]}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_segment_path},
                    {"type": "text", "text": f"""Given these highlight examples:\n{highlight_types}\n\nDoes this video contain a moment that matches the core action of one of the highlights? Answer with:\n'yes' or 'no'\nIf yes, justify it"""}]
            }
        ]

    def process_video_segment(self, video_segment_path: str, highlight_types: str) -> bool:
        messages = self.prompt_video_segment(video_segment_path, highlight_types)
        response = self.generate(messages, max_new_tokens=64)
        print(f"Segment response {response}")
        return "yes" in response

    @staticmethod
    def prompt_text_segment(text: str, highlight_types: str):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGES["highlight_analyzer"]}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "text", "text": f"""Given these highlight examples:\n{highlight_types}\n\nDoes this text contain a moment that matches the core action of one of the highlights? Answer with:\n'yes' or 'no'\nIf yes, justify it"""}]
            }
        ]

    def process_text_segment(self, text: str, highlight_types: str) -> bool:
        messages = self.prompt_text_segment(text, highlight_types)
        response = self.generate(messages, max_new_tokens=64)
        print(f"Segment response {response}")
        return "yes" in response

    @staticmethod
    def get_video_duration_seconds(video_path: str) -> float:
        """Use ffprobe to get video duration in seconds."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    @staticmethod
    def concatenate_scenes(video_path: str, scene_times: list, output_path: str):
        """Concatenate selected scenes into final video."""
        if not scene_times:
            print("No scenes to concatenate, skipping.")
            return

        filter_complex_parts = []
        concat_inputs = []
        for i, (start_sec, end_sec) in enumerate(scene_times):
            filter_complex_parts.append(
                f"[0:v]trim=start={start_sec}:end={end_sec},"
                f"setpts=PTS-STARTPTS[v{i}];"
            )
            filter_complex_parts.append(
                f"[0:a]atrim=start={start_sec}:end={end_sec},"
                f"asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[v{i}][a{i}]")

        concat_filter = f"{''.join(concat_inputs)}concat=n={len(scene_times)}:v=1:a=1[outv][outa]"
        filter_complex = "".join(filter_complex_parts) + concat_filter

        cmd = r'ffmpeg -y -i "%s" -filter_complex "%s" -map [outv] -map [outa] -c:v libx264 -profile:v high -pix_fmt yuv420p -c:a aac -b:a 128k -ar 44100 -ac 2 "%s"' % (video_path, filter_complex, output_path)

        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def concatenate_audio(audio_path: str, segments: list, output_path: str):
        """Concatenate selected scenes into final video."""
        if not segments:
            print("No scenes to concatenate, skipping.")
            return

        filter_parts = []
        concat_inputs = []
        for i, (start, end) in enumerate(segments):
            filter_parts.append(
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[a{i}]")

        concat_filter = f"{''.join(concat_inputs)}concat=n={len(segments)}:v=0:a=1[outa]"
        filter_complex = "".join(filter_parts) + concat_filter

        cmd = f'ffmpeg -y -i "{audio_path}" -filter_complex "{filter_complex}" -map [outa] -c:a pcm_s16le "{output_path}"'

        if os.environ.get('DEBUG', 'False') == 'True':
            os.system(cmd)
        else:
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def cut_scenes(video_path: str, output_path: str, segment_time: int = 10):
        output_segment_path = os.path.join(output_path, "segment_%03d.mp4")
        cmd = 'ffmpeg -i "%s" -f segment -segment_time "%s" -c:v libx264 -preset ultrafast -pix_fmt yuv420p "%s"' % (video_path, segment_time, output_segment_path)
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def strip_repeats(text, min_words=2, max_words=6):
        pattern = re.compile(r'\b((?:\w+\s*){' + str(min_words) + ',' + str(max_words) + r'})\1+', flags=re.IGNORECASE)
        cleaned_text = pattern.sub(r'\1', text)
        return cleaned_text

    def generate_visual_highlight_moments(self, save_dir: str, video_path: str, segment_length: int = 10, duration_limit_min:int = 1, highlight_description: str = None, progress_callback=None):
        # Get video duration
        if progress_callback:
            progress_callback(0, f"Get video duration")
        duration = self.get_video_duration_seconds(video_path)

        # Get video description
        if progress_callback:
            progress_callback(25, f"Get video description")
        description = self.process_video_analyser(video_path)

        # Get highlight types
        if progress_callback:
            progress_callback(50, f"Get highlight types")
        highlight_response = self.process_highlight(description, highlight_description)

        # Parameters
        kept_segments = []

        # Get highlight types
        if progress_callback:
            progress_callback(75, f"Cut video on segments")
        self.cut_scenes(video_path, save_dir, segment_length)

        segments_path = [os.path.join(save_dir, fname) for fname in os.listdir(save_dir) if fname.endswith('.mp4')]

        # Analyser segments
        duration_limit_sec = duration_limit_min * 60  # min to sec
        duration_highlight = 0

        # Create a list of indexes and shuffle them
        indices = list(range(len(segments_path)))
        random.shuffle(indices)

        # Split it into chunks
        chunk_size = 10
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        num_chunks = len(chunks)

        for num, chunk in enumerate(chunks):
            for i in chunk:
                segment_path = segments_path[i]
                start_time = float(i * segment_length)
                end_time = min(float(i * segment_length + segment_length), duration)

                if progress_callback:
                    progress_callback(round(num / num_chunks * 100, 0), f"Get highlight response")

                if duration_highlight + segment_length > duration_limit_sec:
                    break

                if self.process_video_segment(segment_path, highlight_response):
                    kept_segments.append((start_time, end_time))
                    duration_highlight += segment_length
            else:
                continue  # If it isn't reached the limit, we continue processing the next chunk
            break  # If the limit is reached, break

        # Restoring order kept_segments
        kept_segments.sort(key=lambda x: x[0])

        # Get highlight types
        if progress_callback:
            progress_callback(90, f"Video analysing finished")

        return kept_segments

    def generate_text_highlight_moments(self, video_path: str, text: str, timestamp: list, segment_length: int = 10, duration_limit_min: int = 1, highlight_description: str = None, progress_callback=None):
        if not text or not timestamp:
            # Get video description
            if progress_callback:
                progress_callback(100, f"Not found speech")
            return

        # Get video duration
        if progress_callback:
            progress_callback(0, f"Get video duration")
        duration = self.get_video_duration_seconds(video_path)

        # Get video description
        description = ""
        text_length = len(text)
        for i in range(0, text_length, 8192):
            if progress_callback:
                progress_callback(round(i / text_length * 100, 0), f"Text analyser")
            part_description = self.process_text_analyser(video_path, text[i:i + 8192])
            description += self.strip_repeats(part_description)
        else:
            if progress_callback:
                progress_callback(100, f"Text analyser finished")
            description = self.process_text_analyser(video_path, description)

        # Get highlight types
        if progress_callback:
            progress_callback(50, f"Get highlight types")
        highlight_response = self.process_highlight(description, highlight_description)

        # Parameters
        kept_segments = []

        # Analyser segments
        duration_limit_sec = duration_limit_min * 60  # min to sec
        duration_highlight = 0

        # Create a list of indexes and shuffle them
        indices = list(range(len(timestamp)))
        random.shuffle(indices)

        # Split it into chunks
        chunk_size = 10
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        num_chunks = len(chunks)

        for num, chunk in enumerate(chunks):
            for i in chunk:
                segment_text = timestamp[i].get("text")
                start_time = float(timestamp[i].get("start"))
                end_time = min(float(timestamp[i].get("end")), duration)

                if progress_callback:
                    progress_callback(round(num / num_chunks * 100, 0), f"Get highlight response")

                if duration_highlight + segment_length > duration_limit_sec:
                    break

                if self.process_text_segment(segment_text, highlight_response):
                    kept_segments.append((start_time, end_time))
                    duration_highlight += segment_length
            else:
                continue  # If it isn't reached the limit, we continue processing the next chunk
            break  # If the limit is reached, break

        # Restoring order kept_segments
        kept_segments.sort(key=lambda x: x[0])

        # Get highlight types
        if progress_callback:
            progress_callback(90, f"Sound analysing finished")

        return kept_segments


class Whisper:
    def __init__(self, download_root: str, model_id: str = "large-v3-turbo", device: str = "cuda"):
        self.model = whisper.load_model(model_id, device=device, download_root=download_root)

    @staticmethod
    def get_audio(video_path: str, output_path: str):
        audio_path = os.path.join(output_path, f"{os.path.basename(video_path)}.mp3")
        cmd = 'ffmpeg -i "%s" "%s"' % (video_path, audio_path)
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path

    @staticmethod
    def load_audio(audio_path: str):
        return whisper.load_audio(audio_path)

    def __call__(self, audio_path: str, language: str = None, task: Literal["translate", "transcribe"] = "transcribe"):
        audio = self.load_audio(audio_path)
        # detect_disfluencies (Whether to detect and mark disfluencies (hesitations, filler words, etc.) in the transcription.)
        # vad (Whether to use Voice Activity Detection (VAD) to remove non-speech segments.)
        output = whisper.transcribe(self.model, audio, detect_disfluencies=True, vad=True, language=language, task=task)
        return output

    def get(self, whisper_output: dict, segments_length: int = 30, mode: Literal[None, "timestamp", "words"] = None):
        if mode == "timestamp":
            return self.get_timestamp(whisper_output.get("segments", []), segments_length)
        elif mode == "words":
            return sum([segment.get("words", []) for segment in whisper_output.get("segments", [])], [])
        return whisper_output.get("text", "").strip()

    @staticmethod
    def get_timestamp(segments: list, segments_length: int = 30) -> list:
        current_segment_start = 0
        current_text = []
        current_timestamp = []

        for segment in segments:
            start, end = segment['start'], segment['end']
            text = segment['text'].strip()

            if text:
                current_text.append(text)

            if end >= current_segment_start + segments_length:
                if current_text:
                    current_timestamp.append({"start": round(current_segment_start, 2), "end": round(end, 2), "text": ' '.join(current_text)})
                    current_text = []
                    current_segment_start = end

        if current_text:
            last_segment = segments[-1]
            current_timestamp.append({"start": round(current_segment_start, 2), "end": round(last_segment['end'], 2), "text": ' '.join(current_text)})
        return current_timestamp
