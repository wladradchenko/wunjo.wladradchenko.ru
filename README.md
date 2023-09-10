[![Price](https://img.shields.io/badge/price-FREE-0098f7.svg)](https://github.com/wladradchenko/wunjo.wladradchenko.ru/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![GitHub package version](https://img.shields.io/github/v/release/wladradchenko/wunjo.wladradchenko.ru?display_name=tag&sort=semver)](https://github.com/wladradchenko/wunjo.wladradchenko.ru)
[![License: MIT v1.0](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/wladradchenko/wunjo.wladradchenko.ru/blob/main/LICENSE)

<p align="right">(<a href="README_ru.md">RU</a>)</p>
<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru">
    <img src="example/robot.gif" alt="Logo" width="180" height="180">
  </a>

  <h3 align="center">Wunjo AI: Advanced Speech & Deepfake Neural Network Tool</h3>

  <p align="center">
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki">Documentation</a>
    <br/>
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues">Issue</a>
    ·
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/discussions">Discussions</a>
    ·
    <a href="https://youtube.com/playlist?list=PLJG0sD6007zFJyV78mkU-KW2UxbirgTGr&feature=shared">Tutorial</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About

Harness the power of neural networks with Wunjo AI for an array of applications—from speech synthesis to deepfake animations.

**Key Features:**

- **Speech Synthesis:** Effortlessly convert text into human-like speech.
- **Voice Cloning:** Clone voices from provided audio files or directly record your voice within the app for real-time cloning.
- **Multilingual Support:** Currently supports English, Chinese for voice cloning (from any language audio) and English, Russian synthesis, with plans to extend voice cloning synthesis model for Russian.
- **Real-time Speech Recognition:** Dictate text and get instant transcriptions. An efficient tool for hands-free content creation.
- **Multidialogue Creation:** Craft multi-dialogues using unlimited characters with distinct voice profiles.
- **Deepfake Animation:**
  - Animate faces using just one photo combined with audio.
  - Achieve precise lip syncing with your audio using our deepfake lips feature.
  - Effortlessly swap faces in videos, GIFs, and photos using just a single photograph with our "Face Swap" feature.
  - Experimental feature. Change the emotions of a person in the video, with the help of a text description.
- **AI Retouch Tool:** Elevate your videos by removing unwanted objects or refining the quality of your deepfakes.

**Applications:**
From voiceovers in commercials to character voicing in games, from audiobook narrations to fun deepfake projects, Wunjo AI offers endless possibilities and all is free and local on your device.

**Why Choose Wunjo AI?:**

- **All-in-One:** A comprehensive tool catering to both your voice and visual AI needs.
- **User-friendly:** Designed for all, from beginners to professionals.
- **Privacy First:** Functions locally on your desktop, ensuring your data remains private.
- **Open-source & Free:** Benefit from community-driven enhancements and enjoy the app without any cost.

Step into the future of AI-powered creativity with Wunjo AI.

<!-- FEATURES -->
## Setup

Requirements [Python](https://www.python.org/downloads/) version 3.10 and [ffmpeg](https://ffmpeg.org/download.html).

Create venv and activate ones:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Go to portable folder

```
cd portable
```

Run:

```
briefcase dev
```

At the first start, automatic translation into the selected language will be performed, it may take some time. Additionally, you can create a build:

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

<!-- DOWNLOAD -->
## Install packets

[Ubuntu / Debian Stable v1.4 (GPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.4.0.deb)

[Ubuntu / Debian Beta v1.5 (GPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.5.1.deb)

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

// If switching to GPU is not available for you, see the documentation for how to install drivers CUDA
```

[MacOS Stable v1.4 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.4.0.zip)

[MacOS Beta v1.5 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.5.1.zip)

```
// Requirement to create animation is ffmpeg
brew install ffmpeg 

// Unzip app
unzip wunjo_macos_{vesrion}.zip

// Attention! The first time you run video synthesis, models will be downloaded in .wunja/talker/checkpoints and .wunja/talker/gfpgan in size 5GB. This may take a long time.

// Remove cache
rm -rf ~/.wunjo

// How to adjust the use of the GPU and increase the processing speed by several times, see the documentation.
```

[Windows Stable v1.4 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/windows/wunjo_1.4.0.msi)

[Windows Beta v1.5 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/windows/wunjo_1.5.1.msi)

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

// How to adjust the use of the GPU and increase the processing speed by several times, see the documentation.
```

<!-- EXAMPLE -->
## Example

### Speech synthesis and voice cloning

- [Russian synthesized voice from text](https://soundcloud.com/vladislav-radchenko-234338135/russian-voice-text-synthesis?si=ebfc8ea75d0f4c56a3012ca4fdfb6ab5&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
- [English voice cloned from previously synthesized Russian voice](https://soundcloud.com/vladislav-radchenko-234338135/english-voice-clone?si=057718ee0e714e79b2023ce2e37dfb39&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
- [Chinese voice cloned from a previously synthesized Russian voice](https://soundcloud.com/vladislav-radchenko-234338135/chinese-voice-clone?si=43d437bbdf4d4d9a80c7a4a0031189c0&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

### Face animation from image src

<div align="center">
  <table>
  <tr>
    <th>Original</th>
    <th>Move face + Enhancer</th>
    <th>Fix face + Enhancer</th>
  </tr>
  <tr align="center">
    <td><img src="example/animation_face/samurai_original.gif" alt="original" width="228" height="360"></td>
    <td><img src="example/animation_face/samurai_move_enhancer.gif" alt="move_enhancer" width="228" height="360"></td>
    <td><img  src="example/animation_face/samurai_static_enhancer.gif" alt="static_enhancer" width="228" height="360"></td>
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
    <td><img src="example/animate_mouth/pirate_original.gif" alt="original" width="228" height="228"></td>
    <td><img src="example/animate_mouth/pirate_mouth.gif" alt="move_enhancer" width="228" height="228"></td>
    <td><img  src="example/animate_mouth/pirate_mouth_enhancer.gif" alt="static_enhancer" width="228" height="228"></td>
  </tr>
</table>
</div>

### Face swap by one photo

<div align="center">
  <table>
  <tr>
    <th>Original photo</th>
    <th>Original video</th>
    <th>Face swap + Background enhancer</th>
  </tr>
  <tr align="center">
    <td><img src="example/face_swap/face_swap_photo_original.gif" alt="original" width="203" height="203"></td>
    <td><img src="example/face_swap/face_swap_original.gif" alt="original" width="360" height="203"></td>
    <td><img  src="example/face_swap/face_swap_smith.gif" alt="static_enhancer" width="360" height="203"></td>
  </tr>
</table>
</div>

### Remove object by Retouch AI

<div align="center">
  <table>
  <tr>
    <th>Original video</th>
    <th>Remove object</th>
  </tr>
  <tr align="center">
    <td><img src="example/retouch/remove_object_original.gif" alt="original" width="480" height="270"></td>
    <td><img src="example/retouch/remove_object_retouch.gif" alt="original" width="480" height="270"></td>
  </tr>
</table>
</div>

### Retouch AI to improve quality of deepfake

<div align="center">
  <table>
  <tr>
    <th>Defective lines on the chins after animation lip</th>
    <th>Retouch lines on the chins + Face swap</th>
  </tr>
  <tr align="center">
    <td><img src="example/retouch/speech_wav2lip.gif" alt="original" width="480" height="270"></td>
    <td><img src="example/retouch/speech_result.gif" alt="original" width="480" height="270"></td>
  </tr>
</table>
</div>

### Emotion deepfake [Experimental]

This is an experimental feature that is under development, but you can take a look at some of the work right now in Wunjo AI.

<div align="center">
  <table>
  <tr>
    <th>Original</th>
    <th>Happy</th>
    <th>Angry</th>
  </tr>
  <tr align="center">
    <td><img src="example/fake_emotion/original.gif" alt="original" width="213" height="213"></td>
    <td><img src="example/fake_emotion/happy.gif" alt="original" width="213" height="213"></td>
    <td><img src="example/fake_emotion/angry.gif" alt="original" width="213" height="213"></td>
  </tr>
  <tr>
    <th>Fear</th>
    <th>Sad</th>
    <th>Disgust</th>
  </tr>
  <tr align="center">
    <td><img src="example/fake_emotion/fear.gif" alt="original" width="213" height="213"></td>
    <td><img src="example/fake_emotion/sad.gif" alt="original" width="213" height="213"></td>
    <td><img src="example/fake_emotion/disgust.gif" alt="original" width="213" height="213"></td>
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

Update 1.5.1
- [x] Add voice translation with encoders, vocoder english, russian and synthesis english model
- [x] Add synthesys audio from text with voice clone from another audio
- [x] Create hub for voice cloning languages models (or download all default models)
- [x] Add face swap module for deepfake on video/photo from one photo face
- [x] Add enchanter face or enchanter background on user video/photo without deepfake
- [x] Make a version that will include all extensions without extensions download
- [x] Improve indication and translation of current progress
- [x] Add check debug module with python console
- [x] Change real time translation on native translate
- [x] Improve message about GPU unavailable for user
- [x] Add module deepfake emotions as experimental research
- [x] Update guid in application 
- [x] Add AI retouch frames in video by user tool
- [x] Add work with Chinese grammatical and train model to use voice clone on Chinese

Update 1.6.0
- [ ] Add create deepfake video by text prompts
- [ ] Indicate user how much space on drive for tmp and result folders
- [ ] Imitate emotions in voice
- [ ] Train Russian synthesis model voice clone 

<!-- VIDEO -->
## Video

<div align="center">
  <table>
  <tr>
    <th>What is new?</th>
    <th>How install on Windows?</th>
  </tr>
  <tr align="center">
    <td><a href="https://youtu.be/vdvf9NxrUC8"><img src="example/thumbnail/what_is_new.gif" alt="video" width="400"></a></td>
    <td><a href="https://youtu.be/2qIpJYhOL2U"><img src="example/thumbnail/how_install.gif" alt="video" width="400"></a></td>
  </tr>
</table>
</div>

<!-- DONAT -->
## Support the Project

You can support the author of the project in the development of his creative ideas, or just treat him to [a cup of coffee](https://www.buymeacoffee.com/wladradchenko) in USD or [a slice of pizza](https://wladradchenko.ru/donat) in RUB. There are other ways to support the development of the project, more details on [page](https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki/Support-the-Project).

<div align="center">
  <table>
  <tr>
    <th>Buy a cup of coffee in USD</th>
    <th>Buy a slice of pizza in RUB</th>
  </tr>
  <tr align="center">
    <td><img src="https://github.com/wladradchenko/wunjo.wladradchenko.ru/assets/56233697/bc6eefa2-705f-4307-89fd-85d96ec29917" alt="pizza" width="250"></td>
    <td><img src="https://github.com/wladradchenko/wunjo.wladradchenko.ru/assets/56233697/acc80acd-0e39-4476-88db-0a10f2098e25" alt="coffee" width="250"></td>
  </tr>
</table>
</div>

<!-- CONTACT -->
## Contact

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
* Real-Time Voice Cloning - https://github.com/CorentinJ/Real-Time-Voice-Cloning

<p align="right">(<a href="#top">to top</a>)</p>
