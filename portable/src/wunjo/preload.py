import os
import sys
import json
import torch
import customtkinter
from denoiser.pretrained import MASTER_64_URL

from backend.folders import DEEPFAKE_MODEL_FOLDER, VOICE_FOLDER, MEDIA_FOLDER, RTVC_VOICE_FOLDER, SETTING_FOLDER
from backend.download import get_nested_url
from backend.general_utils import set_settings


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


class Window:
    def __init__(self, root, user_settings):
        # Load config files
        self.user_settings = user_settings
        self.deepfake_config = local_deepfake_config()
        self.tts_config = local_tts_config()
        self.rtvc_config = local_rtvc_config()
        self.diffusion_config = local_diffusion_config()
        # Style
        self.style()
        # Root
        self.root = root
        # Setting title
        self.root.title("Settings Wunjo AI")
        # Logical for close window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Setting window size
        width = 600
        height = 570
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)
        self.root.resizable(width=False, height=False)
        # Fonts
        h_font = customtkinter.CTkFont(family='Poppins', size=18, weight="normal")
        element_font = customtkinter.CTkFont(family='Poppins', size=14)
        # Label browser
        label_browser = customtkinter.CTkLabel(self.root, text="Where run Wunjo AI?", anchor="nw",
                                               font=h_font, justify="center", width=200, height=25)
        label_browser.place(x=220, y=100)
        # Radio buttons
        if not sys.platform == "darwin":
            browser_var = customtkinter.StringVar()
            self.user_browser_var = user_settings.get("browser", "default")
            use_default_browser = customtkinter.CTkRadioButton(self.root, font=element_font, text="Default Browser",
                                                               value="default", variable=browser_var, width=150, height=25,
                                                               command=self.use_default_browser_command, fg_color="#42abff")
            use_default_browser.place(x=170, y=140)

            self.use_webgui = customtkinter.CTkRadioButton(self.root, font=element_font, text="WebGUI",
                                                      value="webgui", variable=browser_var, fg_color="#42abff",
                                                      command=self.use_webgui_command, width=150, height=25)
            self.use_webgui.place(x=340, y=140)
            # Set user value
            browser_var.set(self.user_browser_var)
        else:
            self.user_browser_var = "default"

        # Message about CPU/GPU status
        message_accelerator = "CPU / GPU" if torch.cuda.is_available() else "Only CPU"
        status_accelerator = customtkinter.CTkLabel(self.root, font=element_font, text=message_accelerator, height=25, width=100)
        status_accelerator.place(x=10, y=10)

        # Style checkboxes
        checkbox_style = {
            "width": 140,
            "height": 25,
            "font": element_font,
            "hover_color": "#37a3fa",
            "border_color": "#2896ed",
            "text_color": "#31344b",
            "fg_color": "#42abff",
        }

        # Offline mode
        self.user_offline_mode_val = user_settings.get("offline_mode", False)
        offline_mode_var = customtkinter.BooleanVar(value=self.user_offline_mode_val)
        offline_mode = customtkinter.CTkCheckBox(self.root, text="Offline mode", **checkbox_style, offvalue=False, onvalue=True, command=self.offline_mode_command, variable=offline_mode_var)
        offline_mode.place(x=330, y=10)

        # Not show setting window more
        self.not_show_preload_window = user_settings.get("not_show_preload_window", False)
        not_show_preload_var = customtkinter.BooleanVar(value=self.not_show_preload_window)
        not_show = customtkinter.CTkCheckBox(self.root, text="Not show", **checkbox_style, offvalue=False, onvalue=True, command=self.now_show_command, variable=not_show_preload_var)
        not_show.place(x=470, y=10)

        # Features
        label_features = customtkinter.CTkLabel(self.root, font=element_font, width=543, height=30,
                                                text="Click on the buttons to find out which models are missing for the offline")
        label_features.place(x=30, y=210)

        self.feature_active_style = {
            "font": element_font,
            "width": 160,
            "height": 25,
            "hover_color": "#37a3fa",
            "border_color": "#2896ed",
            "text_color": "#31344b",
            "fg_color": "#42abff",
            "corner_radius": 1,
            "border_width": 4
        }

        self.feature_disable_style = {
            "font": element_font,
            "width": 160,
            "height": 25,
            "hover_color": "#ff4f8b",
            "border_color": "#ff4f8b",
            "text_color": "#31344b",
            "fg_color": "transparent",
            "corner_radius": 1,
            "border_width": 4
        }

        self.face_animation_button = customtkinter.CTkButton(self.root, text="Face Animation",
                                                             command=self.face_animation_command,
                                                             **self.feature_active_style)
        self.face_animation_button.place(x=30, y=250)

        self.mouth_animation_button = customtkinter.CTkButton(self.root, text="Mouth Animation",
                                                              command=self.mouth_animation_command,
                                                              **self.feature_active_style)
        self.mouth_animation_button.place(x=30, y=290)

        self.face_swap_button = customtkinter.CTkButton(self.root, text="Face Swap", command=self.face_swap_command,
                                                        **self.feature_active_style)
        self.face_swap_button.place(x=410, y=290)

        self.remove_objects_button = customtkinter.CTkButton(self.root, text="Remove Objects",
                                                             command=self.remove_objects_command,
                                                             **self.feature_active_style)
        self.remove_objects_button.place(x=220, y=290)

        self.video_diffusion_button = customtkinter.CTkButton(self.root, text="Video Diffusion",
                                                              command=self.video_diffusion_command,
                                                              **self.feature_active_style)
        self.video_diffusion_button.place(x=30, y=330)

        self.clone_voice_button = customtkinter.CTkButton(self.root, text="Clone Voice",
                                                          command=self.clone_voice_command,
                                                          **self.feature_active_style)
        self.clone_voice_button.place(x=220, y=330)

        self.speech_synthesis_button = customtkinter.CTkButton(self.root, text="Speech Synthesis",
                                                               command=self.speech_synthesis_command,
                                                               **self.feature_active_style)
        self.speech_synthesis_button.place(x=220, y=250)

        self.enhancer_button = customtkinter.CTkButton(self.root, text="Enhancer", command=self.enhancer_command,
                                                       **self.feature_active_style)
        self.enhancer_button.place(x=410, y=250)

        self.google_button = customtkinter.CTkButton(self.root, text="Google Translate", command=self.google_command,
                                                     **self.feature_active_style)
        self.google_button.place(x=410, y=330)

        # Update style
        self.set_style_feature_by_status()

        # Main message about status
        if not self.inspect_browser_exist() and self.user_browser_var == "webgui":
            start_status_message = "WebGUI not found. Once clicked run, the files will be downloaded if not turn on offline mode. It can take some time."
        else:
            start_status_message = ""
        self.status_message = customtkinter.CTkTextbox(self.root, font=element_font, width=544, height=87)
        self.status_message.insert(customtkinter.END, start_status_message)
        self.status_message.place(x=30, y=380)

        # Run button
        start_button_styles = {
            "width": 160,
            "height": 40,
            "corner_radius": 6,
            "border_width": 0,
            "bg_color": "#f7db4d",
            "fg_color": "#31344b",
            "hover_color": "#e0c431",
            "text_color": "#31344b",
            "text": "Start Wunjo AI",
            "font": h_font,
            "command": self.start_command,
        }

        start_button = customtkinter.CTkButton(self.root, font=h_font, text="Start Wunjo AI", fg_color="#f7db4d",
                                               hover_color="#f2d644", corner_radius=6, border_width=0,
                                               command=self.start_command, width=160, height=40, text_color="#31344b")
        start_button.place(x=220, y=500)

    @staticmethod
    def style():
        # TODO custom theme did after creating window
        customtkinter.set_appearance_mode("light")
        customtkinter.set_default_color_theme("dark-blue")

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
            link_face_recognition = "%s/%s.zip"%(BASE_REPO_URL, name)
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

    def set_style_feature_by_status(self):
        if self.user_offline_mode_val:
            face_animation_status, _ = self.inspect_face_animation()
            self.face_animation_button.configure(**self.feature_active_style) if face_animation_status else self.face_animation_button.configure(**self.feature_disable_style)

            mouth_animation_status, _ = self.inspect_mouth_animation()
            self.mouth_animation_button.configure(**self.feature_active_style) if mouth_animation_status else self.mouth_animation_button.configure(**self.feature_disable_style)

            face_swap_status, _ = self.inspect_face_swap()
            self.face_swap_button.configure(**self.feature_active_style) if face_swap_status else self.face_swap_button.configure(**self.feature_disable_style)

            self.google_button.configure(**self.feature_disable_style)

            enhancer_status, _ = self.inspect_enhancer()
            self.enhancer_button.configure(**self.feature_active_style) if enhancer_status else self.enhancer_button.configure(**self.feature_disable_style)

            retouch_status, _ = self.inspect_retouch()
            self.remove_objects_button.configure(**self.feature_active_style) if retouch_status else self.remove_objects_button.configure(**self.feature_disable_style)

            diffusion_status, _ = self.inspect_diffusion()
            self.video_diffusion_button.configure(**self.feature_active_style) if diffusion_status else self.video_diffusion_button.configure(**self.feature_disable_style)

            rtvc_status, _, _, offline_language = self.inspect_clone_voice()
            self.clone_voice_button.configure(**self.feature_active_style) if rtvc_status and len(offline_language) > 0 else self.clone_voice_button.configure(**self.feature_disable_style)

            tts_status = self.inspect_speech_synthesis()
            self.speech_synthesis_button.configure(**self.feature_active_style) if tts_status else self.speech_synthesis_button.configure(**self.feature_disable_style)

            if not sys.platform == "darwin":
                if not self.inspect_browser_exist():
                    self.use_webgui.configure(**{"fg_color": "#ff4f8b"})
                else:
                    self.use_webgui.configure(**{"fg_color": "#42abff"})
        else:
            self.face_animation_button.configure(**self.feature_active_style)

            self.mouth_animation_button.configure(**self.feature_active_style)

            self.face_swap_button.configure(**self.feature_active_style)

            self.google_button.configure(**self.feature_active_style)

            self.enhancer_button.configure(**self.feature_active_style)

            self.remove_objects_button.configure(**self.feature_active_style)

            self.video_diffusion_button.configure(**self.feature_active_style)

            self.clone_voice_button.configure(**self.feature_active_style)

            self.speech_synthesis_button.configure(**self.feature_active_style)

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

    def use_default_browser_command(self):
        self.user_browser_var = "default"
        # Clear the existing message
        self.status_message.delete("1.0", customtkinter.END)

    def use_webgui_command(self):
        self.user_browser_var = "webgui"
        # Clear the existing message
        self.status_message.delete("1.0", customtkinter.END)
        # Message for WebGUI
        if not self.inspect_browser_exist():
            new_message = "WebGUI not found. Once clicked run, the files will be downloaded if not turn on offline mode. It can take some time."
            self.status_message.insert(customtkinter.END, new_message)

    def offline_mode_command(self):
        self.user_offline_mode_val = not self.user_offline_mode_val
        self.set_style_feature_by_status()

    def now_show_command(self):
        self.not_show_preload_window = not self.not_show_preload_window

    def update_message(self, offline_status: bool, model_list: list):
        # Clear the existing message
        self.status_message.delete("1.0", customtkinter.END)
        if offline_status:
            new_message = "Can be used in offline mode."
        else:
            if self.user_offline_mode_val:
                new_message = "There are no models available for offline use. Download models manually, or run online for automatic downloading."
            else:
                new_message = "There will be automatically downloading models, but if you wish, you can download them manually by link."
            if len(model_list) > 0:
                new_message += "".join(f"\nDownload in {m[0]} model from link {m[1]}" if ".zip" not in m[1] else f"\nDownload archive by link {m[1]} and unzip as {m[0]}, not remove .zip too" for m in model_list)
        self.status_message.insert(customtkinter.END, new_message)

    def face_animation_command(self):
        # Insert the new message
        offline_status, model_list = self.inspect_face_animation()
        self.update_message(offline_status, model_list)

    def mouth_animation_command(self):
        # Insert the new message
        offline_status, model_list = self.inspect_mouth_animation()
        self.update_message(offline_status, model_list)


    def face_swap_command(self):
        # Insert the new message
        offline_status, model_list = self.inspect_face_swap()
        self.update_message(offline_status, model_list)


    def remove_objects_command(self):
        # Insert the new message
        offline_status, model_list = self.inspect_retouch()
        self.update_message(offline_status, model_list)

    def video_diffusion_command(self):
        # Insert the new message
        offline_status, model_list = self.inspect_diffusion()
        self.update_message(offline_status, model_list)
        self.status_message.insert(customtkinter.END, "\nHowever, Wunjo AI may need to download supporting files from the Huggingface to the .cache folder if such files have been cleared previously.")

    def clone_voice_command(self):
        offline_status, model_list, online_language, offline_language = self.inspect_clone_voice()
        # Clear the existing message
        self.status_message.delete("1.0", customtkinter.END)
        if offline_status and len(offline_language) > 0:
            new_message = "Languages " + ", ".join(l for l in offline_language) + " can be used in offline mode."
            if len(online_language) > 0:
                new_message += "However, the following languages are not fully loaded for offline mode " + ", ".join(l for l in online_language)
        else:
            if self.user_offline_mode_val:
                new_message = "There are no models available for offline use. Download models manually, or run online for automatic downloading."
            else:
                new_message = "There will be automatically downloading models, but if you wish, you can download them manually by link."
            if len(model_list) > 0:
                new_message += "".join(f"\nDownload in {m[0]} model from link {m[1]}" for m in model_list)
            if len(online_language) > 0:
                new_message += "The following languages are not fully loaded " + ", ".join(l for l in online_language)
        self.status_message.insert(customtkinter.END, new_message)


    def speech_synthesis_command(self):
        offline_status = self.inspect_speech_synthesis()
        # Clear the existing message
        self.status_message.delete("1.0", customtkinter.END)
        if offline_status:
            new_message = "Can be used in offline mode."
        else:
            new_message = "To use offline default models, you must download at least one voice. More information in documentation https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki/How-manually-install-model-for-text-to-speech"
        self.status_message.insert(customtkinter.END, new_message)


    def enhancer_command(self):
        # Insert the new message
        offline_status, model_list = self.inspect_enhancer()
        self.update_message(offline_status, model_list)


    def google_command(self):
        # Clear the existing message
        self.status_message.delete("1.0", customtkinter.END)
        # Create new message
        new_message = "Google Translator is used for automatic translation in speech synthesis and voice cloning. Not available offline."
        self.status_message.insert(customtkinter.END, new_message)


    def start_command(self):
        # Save settings
        setting_file = os.path.join(SETTING_FOLDER, "settings.json")
        with open(setting_file, 'w') as f:
            self.user_settings["browser"] = self.user_browser_var
            self.user_settings["offline_mode"] = self.user_offline_mode_val
            self.user_settings["not_show_preload_window"] = self.not_show_preload_window
            json.dump(self.user_settings, f)
        # Quit from preload and start Wunjo AI
        self.root.quit()
        self.root.withdraw()

    def on_closing(self):
        # Close the window
        self.root.destroy()
        sys.exit()
        # import signal
        # os.kill(os.getpid(), signal.SIGKILL)


def main():
    # Get user settings
    user_settings = set_settings()
    if not user_settings.get("not_show_preload_window"):
        window = Window(root=customtkinter.CTk(), user_settings=user_settings)
        window.root.mainloop()
        del window
        user_settings = set_settings()
    if user_settings.get("offline_mode"):
        os.environ['WUNJO_OFFLINE_MODE'] = 'True'
    else:
        os.environ['WUNJO_OFFLINE_MODE'] = 'False'
