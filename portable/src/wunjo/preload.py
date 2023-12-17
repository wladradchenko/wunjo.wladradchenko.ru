import os
import sys
import json
import torch
import dearpygui.dearpygui as dpg
from denoiser.pretrained import MASTER_64_URL

from backend.folders import DEEPFAKE_MODEL_FOLDER, VOICE_FOLDER, MEDIA_FOLDER, RTVC_VOICE_FOLDER, SETTING_FOLDER
from backend.download import get_nested_url
from backend.general_utils import set_settings


def _log(sender, app_data, user_data):
    print(f"sender: {sender}, \t app_data: {app_data}, \t user_data: {user_data}")


def local_deepfake_config():
    if not os.path.isfile(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json')):
        deepfake = {}
    else:
        with open(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json'), 'r', encoding="utf8") as file:
            deepfake = json.load(file)
    return deepfake


def local_diffusion_config():
    # create default.json
    custom_diffusion_file = os.path.join(DEEPFAKE_MODEL_FOLDER, "custom_diffusion.json")
    # Check if file exists
    if not os.path.exists(custom_diffusion_file):
        # If the file doesn't exist, create and write to it
        with open(custom_diffusion_file, 'w') as f:
            json.dump({}, f)
    else:
        print(f"{custom_diffusion_file} already exists!")

    with open(custom_diffusion_file, 'r') as f:
        custom_diffusion = json.load(f)
    return custom_diffusion


def local_tts_config():
    def cover_windows_media_path_from_str(path_str: str):
        path_sys = MEDIA_FOLDER
        for folder_or_file in path_str.split("/"):
            if folder_or_file:
                path_sys = os.path.join(path_sys, folder_or_file)
        return path_sys

    if not os.path.isfile(os.path.join(VOICE_FOLDER, 'voice.json')):
        voices_config = {}
    else:
        with open(os.path.join(VOICE_FOLDER, 'voice.json'), 'r', encoding="utf8") as file:
            voices_config = json.load(file)
    if voices_config.get("Unknown") is not None:
        voices_config.pop("Unknown")

    voice_names = []
    for key in voices_config.keys():
        if voices_config[key].get("engine") and voices_config[key].get("vocoder"):
            tacotron2 = voices_config[key]["engine"]["tacotron2"]["model_path"]
            full_tacotron2_path = cover_windows_media_path_from_str(tacotron2)
            tacotron2_download_link = voices_config[key].get("checkpoint_download")

            waveglow = voices_config[key]["vocoder"]["waveglow"]["model_path"]
            full_waveglow_path = cover_windows_media_path_from_str(waveglow)
            waveglow_download_link = voices_config[key].get("waveglow_download")

            voice_names += [
                {
                    key: {
                        "tacotron2": {"model_path": full_tacotron2_path, "download_link": tacotron2_download_link},
                        "waveglow": {"model_path": full_waveglow_path, "download_link": waveglow_download_link}
                    }
                }
            ]

    return voice_names


def local_rtvc_config():
    if not os.path.isfile(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json')):
        rtvc_models = {}
    else:
        with open(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json'), 'r', encoding="utf8") as file:
            rtvc_models = json.load(file)
    return rtvc_models


class App:
    # Close status
    is_close = False
    is_start = False

    def create_app(self):
        # Config
        self.user_settings = set_settings()
        self.deepfake_config = local_deepfake_config()
        self.tts_config = local_tts_config()
        self.rtvc_config = local_rtvc_config()
        self.diffusion_config = local_diffusion_config()
        # Create application
        dpg.create_context()
        self.load_themes()
        # Style windows
        win_width = 410
        win_height = 385

        # Create main Window
        with dpg.window(tag="_main", autosize=True, no_resize=True, no_background=True, no_title_bar=True, min_size=(win_width + 35, win_height), max_size=(win_width + 35, win_height)) as window:
            dpg.bind_item_theme(window, 'app_theme')  # Set style
            with dpg.group(horizontal=True):
                dpg.add_text("CPU / GPU" if torch.cuda.is_available() else "Only CPU")
                dpg.add_spacer(width=160)
                dpg.add_checkbox(label="Offline", callback=self._checkbox_offline, default_value=self.user_settings.get("offline_mode", False))
                dpg.add_checkbox(label="Not show", callback=self._checkbox_not_show, default_value=self.user_settings.get("not_show_preload_window", False))

            dpg.add_spacer(height=50)
            with dpg.drawlist(width=300, height=50):
                dpg.draw_text(pos=(135, 0), text='Wunjo AI', size=36, color=(0, 0, 0))
            dpg.add_text("Where is run app?")
            dpg.add_radio_button(items=("Default Browser", "WebGUI"), default_value=self._get_user_browser(), callback=self._set_user_browser, horizontal=True)
            if not self.inspect_browser_exist() and self._get_user_browser() == "webgui":
                msg = "WebGUI not found. Once clicked run, the files will be downloaded if not turn on offline mode. It can take some time."
            else:
                msg = ""
            dpg.add_text(msg, wrap=win_width, color=(255, 79, 139), tag="_msg_webgui")
            dpg.add_spacer(height=25)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Face Animation", tag="_collapsing_header_face_animation", width=int(win_width / 3))
                dpg.add_button(label="Mouth Animation", tag="_collapsing_header_mouth_animation", width=int(win_width / 3))
                dpg.add_button(label="Face Swap", tag="_collapsing_header_face_swap", width=int(win_width / 3))

            # Face Animation
            with dpg.popup("_collapsing_header_face_animation", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_face_animation"):
                # Insert the new message
                offline_status, model_list = self.inspect_face_animation()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_face_animation", status=offline_status if a else not a)
                msg = self.update_message(offline_status, model_list)
                dpg.add_text(msg, wrap=win_width - 30, tag="_msg_face_animation")
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_face_animation", show=False))

            # Mouth Animation
            with dpg.popup("_collapsing_header_mouth_animation", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_mouth_animation"):
                # Insert the new message
                offline_status, model_list = self.inspect_mouth_animation()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_mouth_animation", status=offline_status if a else not a)
                msg = self.update_message(offline_status, model_list)
                dpg.add_text(msg, wrap=win_width - 30, tag="_msg_mouth_animation")
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_mouth_animation", show=False))

            # Face Swap
            with dpg.popup("_collapsing_header_face_swap", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_face_swap"):
                # Insert the new message
                offline_status, model_list = self.inspect_face_swap()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_face_swap", status=offline_status if a else not a)
                msg = self.update_message(offline_status, model_list)
                dpg.add_text(msg, wrap=win_width - 30, tag="_msg_face_swap")
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_face_swap", show=False))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Video Diffusion", tag="_collapsing_header_video_diffusion", width=int(win_width / 3))
                dpg.add_button(label="Remove Objects", tag="_collapsing_header_remove_objects", width=int(win_width / 3))
                dpg.add_button(label="Enhancer", tag="_collapsing_header_enhancer", width=int(win_width / 3))

            # Video Diffusion
            with dpg.popup("_collapsing_header_video_diffusion", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_video_diffusion"):
                # Insert the new message
                offline_status, model_list = self.inspect_diffusion()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_video_diffusion", status=offline_status if a else not a)
                msg = self.update_message(offline_status, model_list)
                dpg.add_text(msg, wrap=win_width - 30, tag="_msg_diffusion")
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_video_diffusion", show=False))

            # Remove Objects
            with dpg.popup("_collapsing_header_remove_objects", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_remove_objects"):
                # Insert the new message
                offline_status, model_list = self.inspect_retouch()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_remove_objects", status=offline_status if a else not a)
                msg = self.update_message(offline_status, model_list)
                dpg.add_text(msg, wrap=win_width - 30, tag="_msg_retouch")
                dpg.add_separator()
                dpg.add_button(label="Close", width=75,  callback=lambda: dpg.configure_item("_popup_remove_objects", show=False))

            # Enhancer
            with dpg.popup("_collapsing_header_enhancer", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_enhancer"):
                # Insert the new message
                offline_status, model_list = self.inspect_enhancer()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_enhancer", status=offline_status if a else not a)
                msg = self.update_message(offline_status, model_list)
                dpg.add_text(msg, wrap=win_width - 30, tag="_msg_enhancer")
                dpg.add_separator()
                dpg.add_button(label="Close", width=75,  callback=lambda: dpg.configure_item("_popup_enhancer", show=False))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Clone Voice", tag="_collapsing_header_clone_voice", width=int(win_width / 3))
                dpg.add_button(label="Speech Synthesis", tag="_collapsing_header_speech_synthesis", width=int(win_width / 3))
                dpg.add_button(label="Google Translate", tag="_collapsing_header_google_translate", width=int(win_width / 3))

            # Clone Voice
            with dpg.popup("_collapsing_header_clone_voice", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_clone_voice"):
                offline_status, model_list, online_language, offline_language = self.inspect_clone_voice()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_clone_voice", status=offline_status if a else not a)
                if offline_status and len(offline_language) > 0:
                    new_message = "Languages " + ", ".join(l for l in offline_language) + " can be used in offline mode."
                    if len(online_language) > 0:
                        new_message += "However, the following languages are not fully loaded for offline mode " + ", ".join(l for l in online_language)
                else:
                    if self.user_settings['offline_mode']:
                        new_message = "There are no models available for offline use. Download models manually, or run online for automatic downloading."
                    else:
                        new_message = "There will be automatically downloading models, but if you wish, you can download them manually by link."
                    if len(model_list) > 0:
                        new_message += "".join(f"\nDownload in {m[0]} model from link {m[1]}" for m in model_list)
                    if len(online_language) > 0:
                        new_message += "The following languages are not fully loaded " + ", ".join(l for l in online_language)
                dpg.add_text(new_message, wrap=win_width - 30, tag="_msg_clone_voice")
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_clone_voice", show=False))

            # Speech Synthesis
            with dpg.popup("_collapsing_header_speech_synthesis", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_speech_synthesis"):
                offline_status = self.inspect_speech_synthesis()
                a = self.user_settings.get("offline_mode", False)
                self._set_color_header(tag="_collapsing_header_speech_synthesis", status=offline_status if a else not a)
                if offline_status:
                    dpg.add_text("Can be used in offline mode.", wrap=win_width - 30, tag="_msg_speech_synthesis")
                else:
                    dpg.add_text("To use offline default models, you must download at least one voice. More information in documentation https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki/How-manually-install-model-for-text-to-speech", wrap=win_width - 30)
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_speech_synthesis", show=False))

            # Google Translate
            with dpg.popup("_collapsing_header_google_translate", modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_google_translate"):
                self._set_color_header(tag="_collapsing_header_google_translate", status=not self.user_settings.get("offline_mode", False))
                dpg.add_text("Google Translator is used for automatic translation in speech synthesis and voice cloning. Not available offline.", wrap=win_width - 30)
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_google_translate", show=False))
            dpg.bind_item_theme("_popup_google_translate", 'app_theme')  # Set style
            dpg.add_spacer(height=25)
            btn_start = dpg.add_button(label="Save Settings & Run", callback=self._start, width=win_width + 15)
            self._set_color_button_theme(btn_start)

        # Render
        dpg.create_viewport(title="Settings Wunjo AI", width=win_width + 30, height=win_height, resizable=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("_main", True)
        dpg.start_dearpygui()
        dpg.minimize_viewport()
        dpg.destroy_context()
        if not self.is_start:
            sys.exit()

    def inspect_face_animation(self) -> tuple:
        models_is_not_exist = []
        offline_status = False
        if not self.deepfake_config:
            return offline_status, models_is_not_exist

        checkpoint_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
        path_of_net_recon_model = os.path.join(checkpoint_dir_full, 'epoch_20.pth')
        if not os.path.exists(path_of_net_recon_model):
            link_of_net_recon_model = get_nested_url(self.deepfake_config, ["checkpoints", "epoch_20.pth"])
            models_is_not_exist += [(path_of_net_recon_model, link_of_net_recon_model)]

        dir_of_BFM_fitting = os.path.join(checkpoint_dir_full, 'BFM_Fitting')
        if not os.path.exists(dir_of_BFM_fitting):
            link_of_BFM_fitting = get_nested_url(self.deepfake_config, ["checkpoints", "BFM_Fitting.zip"])
            models_is_not_exist += [(dir_of_BFM_fitting, link_of_BFM_fitting)]

        dir_of_hub_models = os.path.join(checkpoint_dir_full, 'hub')
        if not os.path.exists(dir_of_hub_models):
            link_of_hub_models = get_nested_url(self.deepfake_config, ["checkpoints", "hub.zip"])
            models_is_not_exist += [(dir_of_hub_models, link_of_hub_models)]

        wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
        if not os.path.exists(wav2lip_checkpoint):
            link_wav2lip_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "wav2lip.pth"])
            models_is_not_exist += [(wav2lip_checkpoint, link_wav2lip_checkpoint)]

        audio2pose_checkpoint = os.path.join(checkpoint_dir_full, 'auido2pose_00140-model.pth')
        if not os.path.exists(audio2pose_checkpoint):
            link_audio2pose_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "auido2pose_00140-model.pth"])
            models_is_not_exist += [(audio2pose_checkpoint, link_audio2pose_checkpoint)]

        audio2exp_checkpoint = os.path.join(checkpoint_dir_full, 'auido2exp_00300-model.pth')
        if not os.path.exists(audio2exp_checkpoint):
            link_audio2exp_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "auido2exp_00300-model.pth"])
            models_is_not_exist += [(audio2exp_checkpoint, link_audio2exp_checkpoint)]

        free_view_checkpoint = os.path.join(checkpoint_dir_full, 'facevid2vid_00189-model.pth.tar')
        if not os.path.exists(free_view_checkpoint):
            link_free_view_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "facevid2vid_00189-model.pth.tar"])
            models_is_not_exist += [(free_view_checkpoint, link_free_view_checkpoint)]

        mapping_checkpoint = os.path.join(checkpoint_dir_full, 'mapping_00109-model.pth.tar')
        if not os.path.exists(mapping_checkpoint):
            link_mapping_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "mapping_00109-model.pth.tar"])
            models_is_not_exist += [(mapping_checkpoint, link_mapping_checkpoint)]

        mapping_checkpoint_crop = os.path.join(checkpoint_dir_full, 'mapping_00229-model.pth.tar')
        if not os.path.exists(mapping_checkpoint_crop):
            link_mapping_checkpoint_crop = get_nested_url(self.deepfake_config, ["checkpoints", "mapping_00229-model.pth.tar"])
            models_is_not_exist += [(mapping_checkpoint_crop, link_mapping_checkpoint_crop)]

        model_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "model")
        face_recognition = os.path.join(model_dir_full, 'buffalo_l')
        if not os.path.exists(face_recognition):
            BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
            name = 'buffalo_l'
            link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
            models_is_not_exist += [(face_recognition, link_face_recognition)]

        offline_status = False if len(models_is_not_exist) > 0 else True

        return offline_status, models_is_not_exist

    def inspect_mouth_animation(self) -> tuple:
        models_is_not_exist = []
        offline_status = False
        if not self.deepfake_config:
            return offline_status, models_is_not_exist

        checkpoint_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
        wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
        if not os.path.exists(wav2lip_checkpoint):
            link_wav2lip_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "wav2lip.pth"])
            models_is_not_exist += [(models_is_not_exist, link_wav2lip_checkpoint)]

        emo2lip_checkpoint = os.path.join(checkpoint_dir_full, 'emo2lip.pth')
        if not os.path.exists(emo2lip_checkpoint):
            link_emo2lip_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "emo2lip.pth"])
            models_is_not_exist += [(emo2lip_checkpoint, link_emo2lip_checkpoint)]

        model_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "models")
        face_recognition = os.path.join(model_dir_full, 'buffalo_l')
        if not os.path.exists(face_recognition):
            BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
            name = 'buffalo_l'
            link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
            models_is_not_exist += [(face_recognition, link_face_recognition)]

        offline_status = False if len(models_is_not_exist) > 0 else True

        return offline_status, models_is_not_exist

    def inspect_face_swap(self):
        models_is_not_exist = []
        offline_status = False
        if not self.deepfake_config:
            return offline_status, models_is_not_exist

        checkpoint_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
        faceswap_checkpoint = os.path.join(checkpoint_dir_full, 'faceswap.onnx')
        if not os.path.exists(faceswap_checkpoint):
            link_faceswap_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "faceswap.onnx"])
            models_is_not_exist += [(faceswap_checkpoint, link_faceswap_checkpoint)]

        model_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "models")
        face_recognition = os.path.join(model_dir_full, 'buffalo_l')
        if not os.path.exists(face_recognition):
            BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
            name = 'buffalo_l'
            link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
            models_is_not_exist += [(face_recognition, link_face_recognition)]

        offline_status = False if len(models_is_not_exist) > 0 else True

        return offline_status, models_is_not_exist

    def inspect_retouch(self):
        # retouch and remove objects
        models_is_not_exist = []
        offline_status = False
        if not self.deepfake_config:
            return offline_status, models_is_not_exist

        checkpoint_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
        model_pro_painter_path = os.path.join(checkpoint_folder, "ProPainter.pth")
        if not os.path.exists(model_pro_painter_path):
            link_pro_painter = get_nested_url(self.deepfake_config, ["checkpoints", "ProPainter.pth"])
            models_is_not_exist += [(model_pro_painter_path, link_pro_painter)]

        model_raft_things_path = os.path.join(checkpoint_folder, "raft-things.pth")
        if not os.path.exists(model_raft_things_path):
            link_raft_things = get_nested_url(self.deepfake_config, ["checkpoints", "raft-things.pth"])
            models_is_not_exist += [(model_raft_things_path, link_raft_things)]

        model_recurrent_flow_path = os.path.join(checkpoint_folder, "recurrent_flow_completion.pth")
        if not os.path.exists(model_recurrent_flow_path):
            link_recurrent_flow = get_nested_url(self.deepfake_config, ["checkpoints", "recurrent_flow_completion.pth"])
            models_is_not_exist += [(model_recurrent_flow_path, link_recurrent_flow)]

        model_retouch_face_path = os.path.join(checkpoint_folder, "retouch_face.pth")
        if not os.path.exists(model_retouch_face_path):
            link_model_retouch_face = get_nested_url(self.deepfake_config, ["checkpoints", f"retouch_face.pth"])
            models_is_not_exist += [(model_retouch_face_path, link_model_retouch_face)]

        model_retouch_object_path = os.path.join(checkpoint_folder, "retouch_object.pth")
        if not os.path.exists(model_retouch_object_path):
            link_model_retouch_object = get_nested_url(self.deepfake_config, ["checkpoints", f"retouch_object.pth"])
            models_is_not_exist += [(model_retouch_object_path, link_model_retouch_object)]

        # Large model
        sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        if not os.path.exists(sam_vit_checkpoint):
            link_sam_vit_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "sam_vit_h.pth"])
            models_is_not_exist += [(sam_vit_checkpoint, link_sam_vit_checkpoint)]

        onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        if not os.path.exists(onnx_vit_checkpoint):
            link_onnx_vit_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "vit_h_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_checkpoint, link_onnx_vit_checkpoint)]

        # Small model
        sam_vit_small_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
        if not os.path.exists(sam_vit_small_checkpoint):
            link_sam_vit_small_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "sam_vit_b.pth"])
            models_is_not_exist += [(sam_vit_small_checkpoint, link_sam_vit_small_checkpoint)]

        onnx_vit_small_checkpoint = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
        if not os.path.exists(onnx_vit_small_checkpoint):
            link_onnx_vit_small_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "vit_b_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_small_checkpoint, link_onnx_vit_small_checkpoint)]

        # Get models for text detection
        vgg16_baseline_path = os.path.join(checkpoint_folder, "vgg16_baseline.pth")
        if not os.path.exists(vgg16_baseline_path):
            link_vgg16_baseline = get_nested_url(self.deepfake_config, ["checkpoints", "vgg16_baseline.pth"])
            models_is_not_exist += [(vgg16_baseline_path, link_vgg16_baseline)]

        vgg16_east_path = os.path.join(checkpoint_folder, "vgg16_east.pth")
        if not os.path.exists(vgg16_east_path):
            link_vgg16_east = get_nested_url(self.deepfake_config, ["checkpoints", "vgg16_east.pth"])
            models_is_not_exist += [(vgg16_east_path, link_vgg16_east)]

        offline_status = False if len(models_is_not_exist) > 0 else True

        return offline_status, models_is_not_exist

    def inspect_enhancer(self):
        models_is_not_exist = []
        offline_status = False
        if not self.deepfake_config:
            return offline_status, models_is_not_exist

        local_model_path = os.path.join(DEEPFAKE_MODEL_FOLDER, 'gfpgan', 'weights')
        gfpgan_model_path = os.path.join(local_model_path, 'GFPGANv1.4.pth')
        if not os.path.exists(gfpgan_model_path):
            gfpgan_url = get_nested_url(self.deepfake_config, ["gfpgan", "GFPGANv1.4.pth"])
            models_is_not_exist += [(gfpgan_model_path, gfpgan_url)]

        animesgan_model_path = os.path.join(local_model_path, 'realesr-animevideov3.pth')
        if not os.path.exists(animesgan_model_path):
            animesgan_url = get_nested_url(self.deepfake_config, ["gfpgan", "realesr-animevideov3.pth"])
            models_is_not_exist += [(animesgan_model_path, animesgan_url)]

        general_model_path = os.path.join(local_model_path, 'realesr-general-x4v3.pth')
        if not os.path.exists(general_model_path):
            general_url = get_nested_url(self.deepfake_config, ["gfpgan", "realesr-general-x4v3.pth"])
            models_is_not_exist += [(general_model_path, general_url)]

        wdn_model_path = os.path.join(local_model_path, 'realesr-general-wdn-x4v3.pth')
        if not os.path.exists(wdn_model_path):
            wdn_url = get_nested_url(self.deepfake_config, ["gfpgan", "realesr-general-wdn-x4v3.pth"])
            models_is_not_exist += [(wdn_model_path, wdn_url)]

        offline_status = False if len(models_is_not_exist) > 0 else True

        return offline_status, models_is_not_exist

    def inspect_diffusion(self):
        models_is_not_exist = []
        offline_status = False
        # Check models of deepfake
        if not self.deepfake_config:
            return offline_status, models_is_not_exist
        # Check models of stable diffusion
        if not self.diffusion_config:
            return offline_status, models_is_not_exist

        checkpoint_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
        # Large model
        sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        if not os.path.exists(sam_vit_checkpoint):
            link_sam_vit_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "sam_vit_h.pth"])
            models_is_not_exist += [(sam_vit_checkpoint, link_sam_vit_checkpoint)]

        onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        if not os.path.exists(onnx_vit_checkpoint):
            link_onnx_vit_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "vit_h_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_checkpoint, link_onnx_vit_checkpoint)]

        # Small model
        sam_vit_small_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
        if not os.path.exists(sam_vit_small_checkpoint):
            link_sam_vit_small_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "sam_vit_b.pth"])
            models_is_not_exist += [(sam_vit_small_checkpoint, link_sam_vit_small_checkpoint)]

        onnx_vit_small_checkpoint = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
        if not os.path.exists(onnx_vit_small_checkpoint):
            link_onnx_vit_small_checkpoint = get_nested_url(self.deepfake_config, ["checkpoints", "vit_b_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_small_checkpoint, link_onnx_vit_small_checkpoint)]

        diffuser_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "diffusion")
        controlnet_model_path = os.path.join(diffuser_folder, "control_sd15_hed.pth")
        if not os.path.exists(controlnet_model_path):
            link_controlnet_model = get_nested_url(self.deepfake_config, ["diffusion", "control_sd15_hed.pth"])
            models_is_not_exist += [(controlnet_model_path, link_controlnet_model)]

        controlnet_model_annotator_path = os.path.join(diffuser_folder, "ControlNetHED.pth")
        if not os.path.exists(controlnet_model_annotator_path):
            link_controlnet_model_annotator = get_nested_url(self.deepfake_config, ["diffusion", "ControlNetHED.pth"])
            models_is_not_exist += [(controlnet_model_annotator_path, link_controlnet_model_annotator)]

        controlnet_model_canny_path = os.path.join(diffuser_folder, "control_sd15_canny.pth")
        if not os.path.exists(controlnet_model_canny_path):
            link_controlnet_model_canny = get_nested_url(self.deepfake_config, ["diffusion", "control_sd15_canny.pth"])
            models_is_not_exist += [(controlnet_model_canny_path, link_controlnet_model_canny)]

        vae_model_path = os.path.join(diffuser_folder, "vae-ft-mse-840000-ema-pruned.ckpt")
        if not os.path.exists(vae_model_path):
            link_vae_model = get_nested_url(self.deepfake_config, ["diffusion", "vae-ft-mse-840000-ema-pruned.ckpt"])
            models_is_not_exist += [(vae_model_path, link_vae_model)]

        # load gmflow model
        gmflow_model_path = os.path.join(diffuser_folder, "gmflow_sintel-0c07dcb3.pth")
        if not os.path.exists(gmflow_model_path):
            link_gmflow_model = get_nested_url(self.deepfake_config, ["diffusion", "gmflow_sintel-0c07dcb3.pth"])
            models_is_not_exist += [(gmflow_model_path, link_gmflow_model)]

        ebsynth_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "ebsynth")
        if sys.platform == 'win32':
            ebsynth_path = os.path.join(ebsynth_folder, "EbSynth.exe")
            if not os.path.exists(ebsynth_path):
                link_ebsynth = get_nested_url(self.deepfake_config, ["ebsynth", "EbSynth-Beta-Win.zip"])
                models_is_not_exist += [(ebsynth_folder, link_ebsynth)]
        elif sys.platform == 'linux':
            ebsynth_path = os.path.join(ebsynth_folder, "ebsynth_linux_cu118")
            if not os.path.exists(ebsynth_path):
                link_ebsynth = get_nested_url(self.deepfake_config, ["ebsynth", "ebsynth_linux_cu118"])
                models_is_not_exist += [(ebsynth_folder, link_ebsynth)]

        for key, val in self.diffusion_config.items():
            if os.path.exists(os.path.join(diffuser_folder, str(val))):
                print("Is minimum one model for stable diffusion")
                break
        else:
            sd_model_path = os.path.join(diffuser_folder, "Realistic_Vision_V5.1.safetensors")
            if not os.path.exists(sd_model_path):
                link_sd_model = get_nested_url(self.deepfake_config, ["diffusion", "Realistic_Vision_V5.1.safetensors"])
                models_is_not_exist += [(sd_model_path, link_sd_model)]

        offline_status = False if len(models_is_not_exist) > 0 else True

        return offline_status, models_is_not_exist

    def inspect_clone_voice(self):
        offline_status = False
        models_is_not_exist = []
        offline_language = []
        online_language = []
        if not self.rtvc_config:
            return offline_status, models_is_not_exist, online_language, offline_language
        # English
        english_models_is_not_exist = []
        english_models = os.path.join(RTVC_VOICE_FOLDER, "en")
        english_encoder = os.path.join(english_models, "encoder.pt")
        if not os.path.exists(english_encoder):
            link_english_encoder = get_nested_url(self.rtvc_config, ["en", "encoder"])
            english_models_is_not_exist += [(english_encoder, link_english_encoder)]

        english_synthesizer = os.path.join(english_models, "synthesizer.pt")
        if not os.path.exists(english_synthesizer):
            link_english_synthesizer = get_nested_url(self.rtvc_config, ["en", "synthesizer"])
            english_models_is_not_exist += [(english_synthesizer, link_english_synthesizer)]

        english_vocoder = os.path.join(english_models, "vocoder.pt")
        if not os.path.exists(english_vocoder):
            link_english_vocoder = get_nested_url(self.rtvc_config, ["en", "vocoder"])
            english_models_is_not_exist += [(english_vocoder, link_english_vocoder)]

        # Russian
        russian_models_is_not_exist = []
        russian_models = os.path.join(RTVC_VOICE_FOLDER, "ru")
        russian_encoder = os.path.join(russian_models, "encoder.pt")
        if not os.path.exists(russian_encoder):
            link_russian_encoder = get_nested_url(self.rtvc_config, ["ru", "encoder"])
            russian_models_is_not_exist += [(russian_encoder, link_russian_encoder)]

        russian_synthesizer = os.path.join(russian_models, "synthesizer.pt")
        if not os.path.exists(russian_synthesizer):
            link_russian_synthesizer = get_nested_url(self.rtvc_config, ["ru", "synthesizer"])
            russian_models_is_not_exist += [(russian_synthesizer, link_russian_synthesizer)]

        russian_vocoder = os.path.join(russian_models, "vocoder.pt")
        if not os.path.exists(russian_vocoder):
            link_russian_vocoder = get_nested_url(self.rtvc_config, ["ru", "vocoder"])
            russian_models_is_not_exist += [(russian_vocoder, link_russian_vocoder)]

        # Chinese
        chinese_models_is_not_exist = []
        chinese_models = os.path.join(RTVC_VOICE_FOLDER, "zh")
        chinese_encoder = os.path.join(chinese_models, "encoder.pt")
        if not os.path.exists(chinese_encoder):
            link_chinese_encoder = get_nested_url(self.rtvc_config, ["zh", "encoder"])
            chinese_models_is_not_exist += [(chinese_encoder, link_chinese_encoder)]

        chinese_synthesizer = os.path.join(chinese_models, "synthesizer.pt")
        if not os.path.exists(chinese_synthesizer):
            link_chinese_synthesizer = get_nested_url(self.rtvc_config, ["zh", "synthesizer"])
            chinese_models_is_not_exist += [(chinese_synthesizer, link_chinese_synthesizer)]

        chinese_vocoder = os.path.join(chinese_models, "vocoder.pt")
        if not os.path.exists(chinese_vocoder):
            link_chinese_vocoder = get_nested_url(self.rtvc_config, ["zh", "vocoder"])
            chinese_models_is_not_exist += [(chinese_vocoder, link_chinese_vocoder)]

        # General
        general_models = os.path.join(RTVC_VOICE_FOLDER, "general")
        signature = os.path.join(general_models, "signature.pt")
        if not os.path.exists(signature):
            link_signature = get_nested_url(self.rtvc_config, ["general", "signature"])
            models_is_not_exist += [(signature, link_signature)]

        hubert = os.path.join(general_models, "hubert.pt")
        if not os.path.exists(hubert):
            link_hubert = get_nested_url(self.rtvc_config, ["general", "hubert"])
            models_is_not_exist += [(hubert, link_hubert)]

        trimed = os.path.join(general_models, "trimed.pt")
        if not os.path.exists(trimed):
            link_trimed = get_nested_url(self.rtvc_config, ["general", "trimed.pt"])
            models_is_not_exist += [(trimed, link_trimed)]

        voicefixer = os.path.join(general_models, "voicefixer.ckpt")
        if not os.path.exists(voicefixer):
            link_voicefixer = get_nested_url(self.rtvc_config, ["general", "voicefixer.ckpt"])
            models_is_not_exist += [(voicefixer, link_voicefixer)]

        hub_dir = torch.hub.get_dir()
        hub_dir_checkpoints = os.path.join(hub_dir, "checkpoints")
        unmix_vocals_voice = os.path.join(hub_dir_checkpoints, "vocals-bccbd9aa.pth")
        if not os.path.exists(unmix_vocals_voice):
            link_unmix_vocals_voice = get_nested_url(self.rtvc_config, ["general", "vocals-bccbd9aa.pth"])
            models_is_not_exist += [(unmix_vocals_voice, link_unmix_vocals_voice)]

        master_64_path = os.path.join(hub_dir_checkpoints, MASTER_64_URL.split("/")[-1])
        if not os.path.exists(master_64_path):
            models_is_not_exist += [(master_64_path, MASTER_64_URL)]

        # Language
        if len(english_models_is_not_exist) > 0:
            online_language += ["English"]
        else:
            offline_language += ["English"]
        if len(russian_models_is_not_exist) > 0:
            online_language += ["Russian"]
        else:
            offline_language += ["Russian"]
        if len(chinese_models_is_not_exist) > 0:
            online_language += ["Chinese"]
        else:
            offline_language += ["Chinese"]

        offline_status = False if len(models_is_not_exist) > 0 else True

        return offline_status, models_is_not_exist, online_language, offline_language

    def inspect_speech_synthesis(self):
        offline_status = False
        if not self.tts_config:
            return offline_status

        for voice_config in self.tts_config:
            for key, val in voice_config.items():
                voice = voice_config[key]
                if voice.get("tacotron2") and voice.get("waveglow"):
                    tacotron2_path = voice["tacotron2"].get("model_path")
                    waveglow_path = voice["waveglow"].get("model_path")
                    if os.path.exists(tacotron2_path) and os.path.exists(waveglow_path):
                        offline_status = True
                        break

        return offline_status

    def update_message(self, offline_status: bool, model_list: list):
        if offline_status:
            new_message = "Can be used in offline mode."
        else:
            if self.user_settings['offline_mode']:
                new_message = "There are no models available for offline use. Download models manually, or run online for automatic downloading."
            else:
                new_message = "There will be automatically downloading models, but if you wish, you can download them manually by link."
            if len(model_list) > 0:
                new_message += ''.join([f"\nDownload in {m[0]} model from link {m[1]}" if ".zip" not in m[1] else f"\nDownload archive by link {m[1]} and unzip as {m[0]}, not remove .zip too" for m in model_list])
        return new_message

    def _start(self, s, a, u):
        self.update_config_command()
        self.is_start = True
        dpg.stop_dearpygui()
        dpg.minimize_viewport()

    def _checkbox_not_show(self, s, a, u):
        self.user_settings["not_show_preload_window"] = a

    def update_config_command(self):
        # Save settings
        setting_file = os.path.join(SETTING_FOLDER, "settings.json")
        with open(setting_file, 'w') as f:
            self.user_settings["browser"] = self.user_settings.get("browser", "default")
            self.user_settings["offline_mode"] = self.user_settings.get("offline_mode", False)
            self.user_settings["not_show_preload_window"] = self.user_settings.get("not_show_preload_window", False)
            json.dump(self.user_settings, f)

    def _get_user_browser(self):
        _user_browser_bone = {"default": "Default Browser", "webgui": "WebGUI"}
        user_browser = self.user_settings.get("browser", "default")
        return _user_browser_bone.get(user_browser)

    def _set_user_browser(self, s, a, u):
        _sys_browser_bone = {"Default Browser": "default", "WebGUI": "webgui"}
        sys_browser = _sys_browser_bone.get(a)
        self.user_settings["browser"] = sys_browser
        if not self.inspect_browser_exist() and sys_browser == "webgui":
            msg = "WebGUI not found. Once clicked run, the files will be downloaded if not turn on offline mode. It can take some time."
        else:
            msg = ""
        dpg.set_value("_msg_webgui", msg)

    @staticmethod
    def inspect_browser_exist():
        # Message for WebGUI
        if sys.platform == 'win32':
            browser_path = os.path.join(SETTING_FOLDER, 'webgui', 'webgui.exe')
        elif sys.platform == 'linux':
            browser_path = os.path.join(SETTING_FOLDER, 'webgui.AppImage')
        else:
            browser_path = None
        if not os.path.exists(str(browser_path)):
            return False
        return True

    def _checkbox_offline(self, s, a, u):
        self.user_settings['offline_mode'] = a
        # Face Animation
        offline_status, model_list = self.inspect_face_animation()
        dpg.set_value("_msg_face_animation", self.update_message(offline_status, model_list))
        self._set_color_header(tag="_collapsing_header_face_animation", status=offline_status if a else not a)
        # Mouth Animation
        offline_status, model_list = self.inspect_mouth_animation()
        dpg.set_value("_msg_mouth_animation", self.update_message(offline_status, model_list))
        self._set_color_header(tag="_collapsing_header_mouth_animation", status=offline_status if a else not a)
        # Face Swap
        offline_status, model_list = self.inspect_face_swap()
        dpg.set_value("_msg_face_swap", self.update_message(offline_status, model_list))
        self._set_color_header(tag="_collapsing_header_face_swap", status=offline_status if a else not a)
        # Video Diffusion
        offline_status, model_list = self.inspect_diffusion()
        dpg.set_value("_msg_diffusion", self.update_message(offline_status, model_list))
        self._set_color_header(tag="_collapsing_header_video_diffusion", status=offline_status if a else not a)
        # Remove Objects
        offline_status, model_list = self.inspect_retouch()
        dpg.set_value("_msg_retouch", self.update_message(offline_status, model_list))
        self._set_color_header(tag="_collapsing_header_remove_objects", status=offline_status if a else not a)
        # Enhancer
        offline_status, model_list = self.inspect_enhancer()
        dpg.set_value("_msg_enhancer", self.update_message(offline_status, model_list))
        self._set_color_header(tag="_collapsing_header_enhancer", status=offline_status if a else not a)
        # Clone Voice
        offline_status, model_list, online_language, offline_language = self.inspect_clone_voice()
        if offline_status and len(offline_language) > 0:
            msg = "Languages " + ", ".join(l for l in offline_language) + " can be used in offline mode."
            if len(online_language) > 0:
                msg += "However, the following languages are not fully loaded for offline mode " + ", ".join(l for l in online_language)
        else:
            if self.user_settings['offline_mode']:
                msg = "There are no models available for offline use. Download models manually, or run online for automatic downloading."
            else:
                msg = "There will be automatically downloading models, but if you wish, you can download them manually by link."
            if len(model_list) > 0:
                msg += "".join(f"\nDownload in {m[0]} model from link {m[1]}" for m in model_list)
            if len(online_language) > 0:
                msg += "The following languages are not fully loaded " + ", ".join(l for l in online_language)
        dpg.set_value("_msg_clone_voice", msg)
        self._set_color_header(tag="_collapsing_header_clone_voice", status=offline_status if a else not a)
        # Speech Synthesis
        offline_status = self.inspect_speech_synthesis()
        if offline_status:
            msg = "Can be used in offline mode."
        else:
            msg = "To use offline default models, you must download at least one voice. More information in documentation https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki/How-manually-install-model-for-text-to-speech"
        dpg.set_value("_msg_speech_synthesis", msg)
        self._set_color_header(tag="_collapsing_header_speech_synthesis", status=offline_status if a else not a)
        # Google Translate
        self._set_color_header(tag="_collapsing_header_google_translate", status=not a)

    @staticmethod
    def _make_img(file, x_pos, y_pos, layer):
        width, height = 300, 100
        with dpg.texture_registry():
            image_width, image_height, image_channels, image_buffer = dpg.load_image(file)
            texture = dpg.add_static_texture(image_width, image_height, image_buffer)
            dpg.draw_image(texture_tag=texture, pmin=(x_pos, y_pos), pmax=(x_pos + width, y_pos + height), parent=layer)

    @staticmethod
    def _set_color_button_theme(btn):
        colour_yellow = (247, 219, 77)
        colour_yellow_ac = (230, 200, 50)
        colour_yellow_hv = (255, 235, 100)

        with dpg.theme() as item_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, colour_yellow)  # button bg
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colour_yellow_hv)  # button hover
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colour_yellow_ac)  # button active
        dpg.bind_item_theme(btn, item_theme)

    @staticmethod
    def _set_color_header(tag, status=True):
        if status:
            colour = (66, 171, 255)
            colour_ac = (0, 119, 200)
            colour_hv = (29, 151, 236)
        else:
            colour = (255, 79, 139)
            colour_ac = (220, 50, 100)
            colour_hv = (255, 130, 180)

        with dpg.theme() as item_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, colour)  # button bg
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colour_hv)  # button hover
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colour_ac)  # button active
        dpg.bind_item_theme(tag, item_theme)

    @staticmethod
    def load_themes():
        colour_red = (164, 6, 6)
        colour_blue = (66, 171, 255)
        colour_blue_ac = (0, 119, 200)
        colour_blue_hv = (29, 151, 236)
        colour_orange = (217, 131, 46)
        colour_black = (0, 0, 0)
        colour_grey = (40, 40, 40)
        colour_orange_muted = (217, 131, 46, 200)
        colour_white = (236, 240, 243)
        colour_white_bg = (212, 216, 218)
        colour_yellow = (247, 219, 77)
        colour_yellow_ac = (230, 200, 50)
        colour_yellow_hv = (255, 235, 100)

        with dpg.theme(tag='listbox_theme'):
            with dpg.theme_component():
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, colour_orange, category=dpg.mvThemeCat_Core)  # listbox item: selected
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, colour_red,  category=dpg.mvThemeCat_Core)  # listbox item: hovered
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, colour_red, category=dpg.mvThemeCat_Core)  # listbox item: hovered + clicked
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, colour_black, category=dpg.mvThemeCat_Core)  # listbox background colour

        with dpg.theme(tag='app_theme'):
            with dpg.theme_component():
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, colour_white, category=dpg.mvThemeCat_Core)  # background colour
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0,  category=dpg.mvThemeCat_Core)  # window border thickness
                dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)  # apply listbox only theme
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, colour_black, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, colour_grey, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, colour_yellow, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, colour_yellow, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, colour_black, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, colour_blue, category=dpg.mvThemeCat_Core)  # checkbox
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, colour_white_bg, category=dpg.mvThemeCat_Core)  # radio
                dpg.add_theme_color(dpg.mvThemeCol_Header, colour_blue, category=dpg.mvThemeCat_Core)  # collapsing_header bg
                dpg.add_theme_color(dpg.mvThemeCol_Button, colour_blue, category=dpg.mvThemeCat_Core)  # button bg
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colour_blue_ac, category=dpg.mvThemeCat_Core)  # button hover
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colour_blue_hv, category=dpg.mvThemeCat_Core)  # button active
                # modal
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, colour_white, category=dpg.mvThemeCat_Core)  # button active

        with dpg.theme(tag='button_theme'):
            with dpg.theme_component():
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, colour_orange, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colour_red, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colour_red, category=dpg.mvThemeCat_Core)

        with dpg.theme(tag='song_playing_theme'):
            with dpg.theme_component():
                # set the colour of all states of a button to the background colour
                dpg.add_theme_color(dpg.mvThemeCol_Button, colour_black, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colour_black, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colour_black, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, colour_orange, category=dpg.mvThemeCat_Core)


def main():
    # Create the Dear PyGui application
    # Get user settings
    user_settings = set_settings()
    if not user_settings.get("not_show_preload_window"):
        App().create_app()
        user_settings = set_settings()
    if user_settings.get("offline_mode"):
        os.environ['WUNJO_OFFLINE_MODE'] = 'True'
    else:
        os.environ['WUNJO_OFFLINE_MODE'] = 'False'
