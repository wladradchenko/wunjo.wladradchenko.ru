import os
import uuid
import shutil
import torch
from time import strftime

from speech.separator.mdx.model import MDXSeparator

from backend.folders import ALL_MODEL_FOLDER, TMP_FOLDER, MEDIA_FOLDER
from backend.download import download_model, unzip, check_download_size, get_nested_url, is_connected
from backend.config import ModelsConfig, Settings
from backend.general_utils import lprint, get_vram_gb, get_ram_gb, ensure_memory_capacity


file_models_config = ModelsConfig()


class Separator:
    @staticmethod
    def is_valid_audio(file):
        import librosa
        # Check if the file has a recognized audio extension
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']  # Add more if needed
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
    def inspect() -> dict:
        models_information = {"Separator": {"MDX": []}}

        hub_dir_checkpoints = os.path.join(ALL_MODEL_FOLDER, "separator")
        link_audio_separator = get_nested_url(file_models_config(), ["separator", "MDX_Inst_HQ_3.onnx"])
        model_audio_separator = os.path.join(hub_dir_checkpoints, "MDX_Inst_HQ_3.onnx")
        if not os.path.exists(model_audio_separator):
            separator_information = {"path": model_audio_separator, "url": link_audio_separator, "is_exist": False}
        else:
            separator_information = {"path": model_audio_separator, "url": link_audio_separator, "is_exist": True}
        models_information["Separator"]["MDX"].append(separator_information)

        return models_information

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, user_save_name: str = None, target: str = "vocals", trim_silence=False):
        # method
        method = "Separator"
        # create local folder and user
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

        lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
        device = "cpu"

        # load model
        hub_dir_checkpoints = os.path.join(ALL_MODEL_FOLDER, "separator")
        if not os.path.exists(hub_dir_checkpoints):
            os.makedirs(hub_dir_checkpoints, exist_ok=True)

        model_name = 'MDX_Inst_HQ_3'

        if target in ["vocals", "residual"]:
            link_audio_separator = get_nested_url(file_models_config(), ["separator", "MDX_Inst_HQ_3.onnx"])
            model_audio_separator = os.path.join(hub_dir_checkpoints, f"{model_name}.onnx")

            if not os.path.exists(model_audio_separator):
                # check what is internet access
                if is_connected(model_audio_separator):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SEPARATOR": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SEPARATOR = True
                    # download pre-trained models from url
                    download_model(model_audio_separator, link_audio_separator)
                    lprint("Audio separator model installed in PyTorch Hub Cache Directory:", model_audio_separator, user=user_name, level="info")
                else:
                    lprint(msg=f"Not internet connection to download model {link_audio_separator} in {model_audio_separator}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SEPARATOR:
                    check_download_size(model_audio_separator, link_audio_separator)
        else:
            model_audio_separator = None

        if model_audio_separator is None:
            lprint(msg="Model of separator not found.",level="error")
            raise

        msg = f"Before start audio separator memory"
        memory_message(method, msg, user_name, "info")

        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_SEPARATOR_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load separator model.", level="error", user=user_name)
            raise f"Memory out of necessary for {method}."

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

        separator = MDXSeparator(common_config=common_params)
        output_files = separator.separate(source_file)
        output_vocal, output_instrumental = output_files
        output_vocal_path = os.path.join(local_save_dir, output_vocal)
        output_instrumental_path = os.path.join(local_save_dir, output_instrumental)
        # trim silence before analysis
        if trim_silence:
            output_vocal_path = separator.trim_silence(output_vocal_path, local_save_dir)
            output_instrumental_path = separator.trim_silence(output_instrumental_path, local_save_dir)

        del separator
        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        msg = f"After finished and clear memory"
        memory_message(method, msg, user_name, "info")

        user_save_name = str(user_save_name) if user_save_name is not None else str(uuid.uuid4())
        user_save_name_vocal = f"{user_save_name}_vocal.wav"
        user_save_name_instrumental = f"{user_save_name}_instrumental.wav"
        shutil.move(output_vocal_path, os.path.join(user_save_dir, user_save_name_vocal))
        shutil.move(output_instrumental_path, os.path.join(user_save_dir, user_save_name_instrumental))
        # remove tmp files
        if os.path.exists(source_file):
            os.remove(source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"[{method}] Synthesis audio separator finished.", user=user_name, level="info")

        if target in ["vocals", "residual"]:
            # Update config about models if process was successfully
            if Settings.CHECK_DOWNLOAD_SIZE_SEPARATOR and device == "cuda":
                lprint(msg="Update checker of download size separator models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SEPARATOR": False})
                Settings.CHECK_DOWNLOAD_SIZE_SEPARATOR = False

        return {"file": os.path.join(user_save_dir, user_save_name_vocal)}


def memory_message(method: str, msg: str, user: str, level: str):
    _, available_vram_gb, busy_vram_gb = get_vram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_vram_gb} GB and busy {busy_vram_gb} GB VRAM.", user=user, level=level)
    _, available_ram_gb, busy_ram_gb = get_ram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_ram_gb} GB and busy {busy_ram_gb} GB RAM.", user=user, level=level)