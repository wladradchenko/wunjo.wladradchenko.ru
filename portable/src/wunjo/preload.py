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
        win_height = 450

        # Create main Window
        with dpg.window(tag="_main", autosize=True, no_resize=True, no_background=True, no_title_bar=True, min_size=(win_width + 35, win_height), max_size=(win_width + 35, win_height)) as window:
            dpg.bind_item_theme(window, 'app_theme')  # Set style
            with dpg.group(horizontal=True):
                dpg.add_text("CPU / GPU" if torch.cuda.is_available() else "Only CPU")
                dpg.add_spacer(width=160)
                dpg.add_checkbox(label="Offline", callback=self._checkbox_offline, default_value=self.user_settings.get("offline_mode", False))
                dpg.add_checkbox(label="Not show", callback=_log, default_value=self.user_settings.get("not_show_preload_window", False))
            dpg.add_spacer(height=160)
            with dpg.texture_registry():
                image_width, image_height, image_channels, image_buffer = dpg.load_image("/home/user/Documents/CONDA/wunjo.wladradchenko.ru/portable/src/wunjo/static/basic/icon/gray/logo.png")
                texture = dpg.add_static_texture(image_width, image_height, image_buffer)
            dpg.add_image(texture_tag=texture, width=100, height=100, pos=(int(win_width/2.5), 70))
            dpg.add_text("Where is run app?")
            dpg.add_radio_button(items=("Default Browser", "WebGUI"), default_value=self._get_user_browser(), callback=self._set_user_browser, horizontal=True)
            dpg.add_text("WebGUI not found. Once clicked run, the files will be downloaded if not turn on offline mode. It can take some time.", wrap=win_width, color=(255, 79, 139))
            dpg.add_spacer(height=25)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Face Animation", tag="_collapsing_header_face_animation", width=int(win_width / 3))
                dpg.add_button(label="Mouth Animation", tag="_collapsing_header_mouth_animation", width=int(win_width / 3))
                dpg.add_button(label="Face Swap", tag="_collapsing_header_face_swap", width=int(win_width / 3))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Video Diffusion", tag="_collapsing_header_video_diffusion", width=int(win_width / 3))
                dpg.add_button(label="Remove Objects", tag="_collapsing_header_remove_objects", width=int(win_width / 3))
                dpg.add_button(label="Enhancer", tag="_collapsing_header_enhancer", width=int(win_width / 3))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Clone Voice", tag="_collapsing_header_clone_voice", width=int(win_width / 3))
                dpg.add_button(label="Speech Synthesis", tag="_collapsing_header_speech_synthesis", width=int(win_width / 3))
                dpg.add_button(label="Google Translate", tag="_collapsing_header_google_translate", width=int(win_width / 3))
            with dpg.popup(dpg.last_item(), modal=True, mousebutton=dpg.mvMouseButton_Left, tag="_popup_google_translate"):
                dpg.add_text("Google Translator is used for automatic translation in speech synthesis and voice cloning. Not available offline.", wrap=win_width - 30)
                dpg.add_separator()
                dpg.add_button(label="Close", width=75, callback=lambda: dpg.configure_item("_popup_google_translate", show=False))
            dpg.bind_item_theme("_popup_google_translate", 'app_theme')  # Set style
            dpg.add_spacer(height=25)
            btn_start = dpg.add_button(label="Save Settings & Run", callback=_log, width=win_width + 15)
            self._set_color_button_theme(btn_start)

        # Render
        dpg.create_viewport(decorated=True, title="Settings Wunjo AI", width=win_width + 30, height=win_height, resizable=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("_main", True)
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _get_user_browser(self):
        _user_browser_bone = {"default": "Default Browser", "webgui": "WebGUI"}
        user_browser = self.user_settings.get("browser", "default")
        return _user_browser_bone.get(user_browser)

    def _set_user_browser(self, s, a, u):
        _sys_browser_bone = {"Default Browser": "default", "WebGUI": "webgui"}
        sys_browser = _sys_browser_bone.get(a)
        self.user_settings["browser"] = sys_browser

    def _checkbox_offline(self, s, a, u):
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
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Header, colour)  # button bg
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, colour_hv)  # button hover
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, colour_ac)  # button active
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

    @staticmethod
    def load_fonts():
        # add fonts to font registry
        with dpg.font_registry():
            # add font (set as default for entire app)
            default_font = dpg.add_font(id='font1', file='fonts/open_sans/opensans_regular.ttf', size=20)
            dpg.bind_font(default_font)
            # add second font
            dpg.add_font(tag="font2", file='fonts/sedgwick_ave_display/sedgwickavedisplay-regular.ttf', size=36)
            # add third font
            dpg.add_font(tag="font3", file='fonts/pacifico/pacifico-regular.ttf', size=140)
            # add fourth font, used for logo 'music player' part of the logo.
            # Font size must match size specified in draw_text.
            dpg.add_font(tag="font4", file='fonts/reader_pro/readerpro_medium.ttf', size=21)


def main():
    App().create_app()
