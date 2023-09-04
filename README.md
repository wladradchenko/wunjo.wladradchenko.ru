[![Price](https://img.shields.io/badge/price-FREE-0098f7.svg)](https://github.com/wladradchenko/wunjo.wladradchenko.ru/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/librosa.svg)](https://badge.fury.io/py/librosa)
[![GitHub package version](https://img.shields.io/github/v/release/wladradchenko/wunjo.wladradchenko.ru?display_name=tag&sort=semver)](https://github.com/wladradchenko/wunjo.wladradchenko.ru)
[![License: MIT v1.0](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/wladradchenko/wunjo.wladradchenko.ru/blob/main/LICENSE)

<p align="right">(<a href="README_ru.md">RU</a>)</p>
<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru">
    <img src="example/robot.gif" alt="Logo" width="180" height="180">
  </a>

  <h3 align="center">Wunjo AI</h3>

  <p align="center">
    Project documentation
    <br/>
    <br/>
    <br/>
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues">Issue</a>
    ·
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki">Wikipedia</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About

Synthesis of speech from text, voice clone, speech to text and creation of deepfake animation face, lips, face swap by Wunjo AI neural networks

Application for speech synthesis from text, speech recognition to text, voice clone and create deepfake animation face, lips, face swap by one photo. One of the unique features of this application is the ability to create multi-dialogues with multiple voices, clone voices from audio files of real time record voice, and the number of characters used is not limited, unlike similar web applications. You can also speak text in real time and the app will recognize it from the audio to text. This feature is great for dictating text instead of manually typing it.

Also, deepfakes are another approach to use the application. You can animate face by photo and audio, use deepfake lips of user audio, and use face swap deepfake on video, gif animation and photo by one photo face. 

All in all, this neural network desktop application is a handy and powerful tool for anyone who needs speech synthesis, voice-to-text recognition and create deepfake animation. Best of all, the app is free, installs locally, and is easy to use! And you can use it in the voice acting of commercials, books, games, etc.

<!-- FEATURES -->
## Setup

Requirements 3.8 <= [Python](https://www.python.org/downloads/) <=3.10 and [ffmpeg](https://ffmpeg.org/download.html).

Create venv and activate ones:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Attention! The first time you run video synthesis, models will be downloaded in .wunja/talker/checkpoints and .wunja/talker/gfpgan in size 5GB. This may take a long time.

Go to portable folder
```
cd portable
```

Run:
```
briefcase dev
```

Additionally, you can create a build:
```
briefcase build
```

Run build
```
briefcase run
```

Create install packet for your OS:
```
briefcase package
```

Read more in the documentation [BeeWare](https://beeware.org/project/projects/tools/briefcase)

<!-- EXAMPLE -->
## Extensions

The functionality of the program can be supplemented by custom extensions. Extension example and format at [link](https://github.com/wladradchenko/advanced.wunjo.wladradchenko.ru/README.md)

Available list of extensions at [link](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/extensions.json).

<!-- DOWNLOAD -->
## Install packets

[Ubuntu / Debian Stable v1.3](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.3.1.deb)

[Ubuntu / Debian Beta v1.4](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.4.0.deb)

```
// Requirement to create animation is ffmpeg
sudo apt install ffmpeg

// Install app
sudo dpkg -i wunjo_{vesrion}.deb

// Attention! The first time you run video synthesis, models will be downloaded in .wunja/talker/checkpoints and .wunja/talker/gfpgan in size 5GB. This may take a long time.

// Remove app
sudo dpkg -r wunjo

// Remove cache
rm -rf ~/.wunjo
```

[MacOS Stable v1.3](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.3.1.zip)

[MacOS Beta v1.4](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.4.0.zip)

```
// Requirement to create animation is ffmpeg
brew install ffmpeg 

// Unzip app
unzip wunjo_macos_{vesrion}.zip

// Attention! The first time you run video synthesis, models will be downloaded in .wunja/talker/checkpoints and .wunja/talker/gfpgan in size 5GB. This may take a long time.

// Remove cache
rm -rf ~/.wunjo
```

[Windows Stable v1.3](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/windows/wunjo_1.3.2.msi)

[Windows Beta v1.4](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/windows/wunjo_1.4.0.msi)

```
// Requirement to create animation is ffmpeg, Install ffmpeg and add to Path env
setx PATH "%PATH%;C:\path\to\ffmpeg\bin"

// Install app
wunjo_{vesrion}.msi

// Important! How to set up deepfake for Windows. You need to give permission to read the neural network models in the gfpgan folder after the models are installed! Without this setting, the result of deepfake generation will be "Face not found".

icacls "%USERPROFILE%/.wunjo/deepfake/gfpgan/weights/*.pth" /grant:r "Users":(R,W)

// Attention! The first time you run video synthesis, models will be downloaded in .wunja/talker/checkpoints and .wunja/talker/gfpgan in size 5GB. This may take a long time.

//Remove cache
%USERPROFILE%/.wunjo
```

<!-- EXAMPLE -->
## Example

### Face animation from image src

<div align="center">
  <table>
  <tr>
    <th>Original</th>
    <th>Move face + Enhancer</th>
    <th>Fix face + Enhancer</th>
  </tr>
  <tr align="center">
    <td><img src="example/samurai_original.gif" alt="original" width="228" height="360"></td>
    <td><img src="example/samurai_move_enhancer.gif" alt="move_enhancer" width="228" height="360"></td>
    <td><img  src="example/samurai_static_enhancer.gif" alt="static_enhancer" width="228" height="360"></td>
  </tr>
</table>
</div>

### Mouth animation from video src

<div align="center">
  <table>
  <tr>
    <th>Original</th>
    <th>Mouth animation</th>
    <th>Mouth animation + Enhancer</th>
  </tr>
  <tr align="center">
    <td><img src="example/pirate_original.gif" alt="original" width="228" height="228"></td>
    <td><img src="example/pirate_mouth.gif" alt="move_enhancer" width="228" height="228"></td>
    <td><img  src="example/pirate_mouth_enhancer.gif" alt="static_enhancer" width="228" height="228"></td>
  </tr>
</table>
</div>

<!-- EXAMPLE -->

<!-- TRANSLATION -->
## Language

The application comes with built-in support for the following languages: English, Russian, Chinese, Portuguese, and Korean.

If you wish to add a new language:

Navigate to `.wunjo/settings/settings.json`.
Add your desired language in the format: `"default_language": {"name": "code"}`.
To find the appropriate code for your language, please refer to the [Google Cloud Translate Language Codes](https://cloud.google.com/translate/docs/languages).
<!-- TRANSLATION -->

<!-- UPDATE -->
Update 1.3.2

- [x] Added support for extensions (any developer can create extensions without opening the main code)
- [x] Fix bugs

Update 1.4.0
- [x] Add a deepfake to work with the source video, which synchronizes the movement of the lips under the audio
- [x] Add selection of video segment for deepfake depending on the length of the audio track
- [x] Add ability to change directory for .wunjo cache folder
- [x] Add application translation into different languages
- [x] Train and add TTS models for English speech
- [x] Add language of model identification on interface
- [x] Add the possibility to speak on english by russian model and speak on russian by english model
- [x] Add the possibility of using speech synthesis on TTS user models (ru, en)
- [x] Add the new update information
- [x] Integrate links to train video and wiki page

Update 1.5.0
- [x] Add voice translation with encoders english, russian
- [x] Add synthesys audio from text with voice clone from another audio
- [x] Create hub for voice cloning languages models (or download all default models)
- [x] Add face swap module for deepfake on video/photo from one photo face
- [ ] Add enchanter face or enchanter background on user video/photo without deepfake
- [x] Make a version that will include all extensions without extensions download
- [x] Improve indication and translation of current progress
- [x] Add check debug module with python console
- [x] Change real time translation on native translate
- [ ] Improve message about GPU unavailable for user
- [ ] Add module deepfake emotions
- [ ] Update guid in application 
- [x] Add AI retouch frames in video by user tool

:TODO check downloads tts models on windows, maybe is bug because of symbols
:TODO check dicts from tacotron2/data after build, will be errors because of torch lib logical

Update 1.6.0
- [ ] Add create deepfake video by text prompts
- [ ] Add in face animation and mouth approach to worked at the some time with a few heads and difference audio
- [ ] Indicate user how much space on drive for tmp and result folders
- [ ] Imitate emotions in voice
- [ ] Train voices for Chinese and add logical to work with chinese grammatical

<!-- VIDEO -->
## Video

### What is new

[![Watch the video](https://img.youtube.com/vi/oHQR1Zx6YOk/hqdefault.jpg)](https://youtu.be/oHQR1Zx6YOk)

### How install

[![Install tutorial on Windows](https://img.youtube.com/vi/2qIpJYhOL2U/hqdefault.jpg)](https://youtu.be/2qIpJYhOL2U)

<!-- CONTACT -->
## Контакт

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project: [https://github.com/wladradchenko/wunjo.wladradchenko.ru](https://github.com/wladradchenko/wunjo.wladradchenko.ru)

Web site: [wladradchenko.ru/wunjo](https://wladradchenko.ru/wunjo)

<!-- PREMISE -->
## Premise

Wunjo comes from the ancient runic alphabet and represents joy and contentment, which could tie into the idea of using the application to create engaging and expressive speech. Vunyo (ᚹ) is the eighth rune of the Elder and Anglo-Saxon Futhark. Prior to the introduction of the letter W into the Latin alphabet, the letter Ƿynn (Ƿƿ) was used instead in English, derived from this rune.

<!-- CREDITS -->
## Credits

* Tacatron 2 - https://github.com/NVIDIA/tacotron2
* Waveglow - https://github.com/NVIDIA/waveglow
* Flask UI - https://github.com/ClimenteA/flaskwebgui
* BeeWare - https://beeware.org/project/projects/tools/briefcase/
* Sad Talker - https://github.com/OpenTalker/SadTalker
* Wav2lip - https://github.com/Rudrabha/Wav2Lip
* Face Utils - https://github.com/xinntao/facexlib
* Face Enhancement - https://github.com/TencentARC/GFPGAN
* Image/Video Enhancement - https://github.com/xinntao/Real-ESRGAN

<p align="right">(<a href="#top">to top</a>)</p>
