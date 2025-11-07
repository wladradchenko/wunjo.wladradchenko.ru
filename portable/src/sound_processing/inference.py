import os
import uuid
import shutil
import torch
from gtts import gTTS
from time import strftime

from sound_processing.separator.mdx.model import MDXSeparator

from backend.folders import ALL_MODEL_FOLDER, TMP_FOLDER, MEDIA_FOLDER
from backend.download import download_model, check_download_size, get_nested_url, is_connected
from backend.config import ModelsConfig, Settings
from backend.general_utils import lprint, get_vram_gb, get_ram_gb, ensure_memory_capacity, get_best_device


file_models_config = ModelsConfig()


# TODO Check what with "https://github.com/snakers4/silero-vad/zipball/master" to /home/user/.wunjo/deepfake/clone/hub/master.zip will not be problems on Windows else write custom method

class BaseModule:
    @staticmethod
    def get_media_url(path: str):
        return path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")

    @staticmethod
    def is_valid_audio(file):
        import librosa
        # Check if the file has a recognized audio extension
        audio_extensions = ['.wav', '.mp3']  # Add more if needed
        _, ext = os.path.splitext(file)
        if ext.lower() in audio_extensions:
            return True
        # Try to load the file as audio and check if it's valid
        try:
            # Using librosa
            librosa.load(file, sr=None)
            return True
        except Exception as e:
            pass
        return False

    @staticmethod
    def get_model_info(model_path, url_key):
        return {
            "path": model_path,
            "url": get_nested_url(file_models_config(), url_key) if isinstance(url_key, list) else url_key,
            "is_exist": os.path.exists(model_path)
        }

    @staticmethod
    def validate_models(model_path, model_url, user_name, method_name: str, progress_callback=None):
        """
        Helper function to check if the model exists, validate its size, and download if needed.
        """
        if not os.path.exists(model_path):
            if is_connected(model_url):
                lprint(msg=f"Model {model_path} not found. Downloading model.", user=user_name, level="info")
                BaseModule.update_model_download_checker(method_name)
                download_model(model_path, model_url, progress_callback=progress_callback)
            else:
                lprint(msg=f"No internet connection to download model {model_url}.", user=user_name, level="error")
                raise ConnectionError(f"Cannot download model {model_path}")
        else:
            # Only check the download size if the setting is enabled
            if getattr(Settings, method_name, False):
                check_download_size(model_path, model_url, progress_callback=progress_callback)

    @staticmethod
    def update_model_download_checker(method_name: str):
        """
        Update settings to enable model download size checking.
        """
        lprint(msg=f"Updating download size checker for method: {method_name}.")
        # Update config with method-specific download size check flag
        Settings.config_manager.update_config({method_name: True})
        # Set the dynamic attribute in Settings to True
        setattr(Settings, method_name, True)


class Separator(BaseModule):
    @staticmethod
    def inspect() -> dict:
        hub_dir_checkpoints = os.path.join(ALL_MODEL_FOLDER, "separator")
        model_data = {}

        model_info_configs = [
            ("Separator", "MDX", os.path.join(hub_dir_checkpoints, "MDX_Inst_HQ_3.onnx"), ["separator", "MDX_Inst_HQ_3.onnx"]),
        ]

        for category, sub_category, file_path, url_key in model_info_configs:
            model_info = Separator.get_model_info(file_path, url_key)
            if not model_info.get("is_exist"):
                if model_data.get(category) is None:
                    model_data[category] = {}
                if model_data[category].get(sub_category) is None:
                    model_data[category][sub_category] = []
                model_data[category][sub_category].append(model_info)

        return model_data

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, user_save_name: str = None, target: str = "vocals", trim_silence=False, progress_callback=None):
        # method
        method = "Separator"
        # create local folder and user
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # load model
        # hub_dir = torch.hub.get_dir()
        # hub_dir_checkpoints = os.path.join(hub_dir, "checkpoints")
        hub_dir_checkpoints = os.path.join(ALL_MODEL_FOLDER, "separator")
        if not os.path.exists(hub_dir_checkpoints):
            os.makedirs(hub_dir_checkpoints, exist_ok=True)

        model_name = 'MDX_Inst_HQ_3'

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        if target in ["vocals", "residual"]:
            # link_audio_separator = "https://huggingface.co/wladradchenko/wunjo.wladradchenko.ru/resolve/main/separator/mdx/MDX_Inst_HQ_3.onnx"
            link_audio_separator = get_nested_url(file_models_config(), ["separator", "MDX_Inst_HQ_3.onnx"])
            model_audio_separator = os.path.join(hub_dir_checkpoints, f"{model_name}.onnx")
            Separator.validate_models(model_audio_separator, link_audio_separator, user_name, "CHECK_DOWNLOAD_SIZE_SEPARATOR")
        else:
            model_audio_separator = None

        if model_audio_separator is None:
            lprint(msg="Model of separator not found.",level="error")
            raise ValueError("Model of separator not found.")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inspection models...")

        msg = f"Before start audio separator memory"
        memory_message(method, msg, user_name, "info", device=device)

        # Updating progress
        if progress_callback:
            progress_callback(0, "Ensure memory capacity...")

        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_SEPARATOR_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load separator model.", level="error", user=user_name)
            raise RuntimeError(f"Memory out of necessary for {method}.")

        common_params = {
            "torch_device": torch.device("cuda") if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else torch.device("cpu"),
            "torch_device_cpu": torch.device("cpu"),
            "torch_device_mps": None,  # torch.device("mps"),
            "onnx_execution_provider": None,  # ["CUDAExecutionProvider", "CPUExecutionProvider"]
            "model_name": model_name,
            "model_path": model_audio_separator,
            "model_data": {'compensate': 1.022, 'mdx_dim_f_set': 3072, 'mdx_dim_t_set': 8, 'mdx_n_fft_scale_set': 6144, 'primary_stem': 'Instrumental'},  # special for each model from mdx_model_data.json by hash
            "output_format": "WAV",
            "output_dir": local_save_dir,
            "normalization_threshold": 0.9,
            "output_single_stem": None,
            "invert_using_spec": False,
            "sample_rate": 44100,
        }

        # Updating progress
        if progress_callback:
            progress_callback(0, "Separator...")

        separator = MDXSeparator(common_config=common_params)
        output_files = separator.separate(source_file, progress_callback=progress_callback)
        output_vocal, output_instrumental = output_files
        output_vocal_path = os.path.join(local_save_dir, output_vocal)
        output_instrumental_path = os.path.join(local_save_dir, output_instrumental)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Separator...")

        del separator
        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        msg = f"After finished and clear memory"
        memory_message(method, msg, user_name, "info", device=device)

        user_save_name = str(user_save_name) if user_save_name is not None else str(uuid.uuid4())
        user_save_name_vocal = f"{user_save_name}_vocal.wav"
        user_save_name_instrumental = f"{user_save_name}_instrumental.wav"
        shutil.move(output_vocal_path, os.path.join(user_save_dir, user_save_name_vocal))
        shutil.move(output_instrumental_path, os.path.join(user_save_dir, user_save_name_instrumental))
        # remove tmp dir (not remove tmp file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"[{method}] Synthesis audio separator finished.", user=user_name, level="info")

        if target in ["vocals", "residual"]:
            # Update config about models if process was successfully
            if Settings.CHECK_DOWNLOAD_SIZE_SEPARATOR and device != "cpu":
                lprint(msg="Update checker of download size separator models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SEPARATOR": False})
                Settings.CHECK_DOWNLOAD_SIZE_SEPARATOR = False

        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished.")

        output_vocal_path = os.path.join(user_save_dir, user_save_name_vocal)
        output_instrumental_path = os.path.join(user_save_dir, user_save_name_instrumental)
        media_vocal_path = Separator.get_media_url(output_vocal_path)
        media_instrumental_path = Separator.get_media_url(output_instrumental_path)

        return {"file": [output_vocal_path, output_instrumental_path], "media": [media_vocal_path, media_instrumental_path]}


def memory_message(method: str, msg: str, user: str, level: str, device="cuda:0"):
    _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
    lprint(msg=f"[{method}] {msg}. Available {available_vram_gb} GB and busy {busy_vram_gb} GB VRAM.", user=user, level=level)
    _, available_ram_gb, busy_ram_gb = get_ram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_ram_gb} GB and busy {busy_ram_gb} GB RAM.", user=user, level=level)