<p align="right">(<a href="README_en.md">RU</a>)</p>
<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru">
    <img src="icons/robot.gif" alt="Logo" width="180" height="180">
  </a>

  <h3 align="center">Voice AI</h3>

  <p align="center">
    Project documentation
    <br/>
    <br/>
    <br/>
    <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru/issues">Issue</a>
    ·
    <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru/issues">Features</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About

Application for speech synthesis from text and speech recognition to text. One of the unique features of this application is the ability to create multi-dialogues with multiple voices, and the number of characters used is not limited, unlike similar web applications. You can also speak text in real time and the app will recognize it from the audio. This feature is great for dictating text instead of manually typing it.

All in all, this neural network desktop application is a handy and powerful tool for anyone who needs speech synthesis and voice-to-text recognition. Best of all, the app is free, installs locally, and is easy to use! And you can use it in the voice acting of commercials, books, games, etc.

<!-- FEATURES -->
## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Download neural network models, configurations and dictionary:
```
wget invest.wladradchenko.ru/static/voiceai.wladradchenko.ru/download/voiceai.robot.zip 
mv DOWNLOAD_DIR/voiceai.robot.zip voiceai.wladradchenko.ru/portable/src/backend/voiceai.robot.zip
cd voiceai.wladradchenko.ru/portable/src/backend/
unzip voiceai.robot.zip
```

Unzip archive:
```
The folder data to voiceai.wladradchenko.ru/portable/src/backend
The file config.yaml to voiceai.wladradchenko.ru/portable/src/backend/config.yaml
The dictionary stress.dict to voiceai.wladradchenko.ru/portable/src/backend/tps/data
```

Add models for animation. Move to dir talker

```
cd voiceai.wladradchenko.ru/talker
```

Create folder checkpoints and download models by scripts/download_models.sh:

```
bash scripts/download_models.sh
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

<!-- DOWNLOAD -->
## Install packets
Ubuntu / Debian - [https://wladradchenko.ru/voice](https://invest.wladradchenko.ru/static/voiceai.wladradchenko.ru/download/linux/voiceai_1.0.0-1~ubuntu-jammy_amd64.deb)


```
// Install app
sudo dpkg -i voiceai.deb

// Remove app
sudo dpkg -r voiceai

// Remove cache
rm -rf ~/.voiceai
```


<!-- VIDEO -->
## Video

[![Watch the video](https://img.youtube.com/vi/aekVTaJHzqY/maxresdefault.jpg)](https://youtu.be/aekVTaJHzqY)

<!-- CONTACT -->
## Контакт

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project: [https://github.com/wladradchenko/voiceai.wladradchenko.ru](https://github.com/wladradchenko/voiceai.wladradchenko.ru)

Web site: [wladradchenko.ru/voice](https://wladradchenko.ru/voice)

<!-- CREDITS -->
## Credits

* Tacatron 2 - https://github.com/NVIDIA/tacotron2
* Waveglow - https://github.com/NVIDIA/waveglow
* Flask UI - https://github.com/ClimenteA/flaskwebgui
* BeeWare - https://beeware.org/project/projects/tools/briefcase/
* Sad Talker - https://github.com/OpenTalker/SadTalke
* Face Utils: https://github.com/xinntao/facexlib
* Face Enhancement: https://github.com/TencentARC/GFPGAN
* Image/Video Enhancement:https://github.com/xinntao/Real-ESRGAN

<p align="right">(<a href="#top">to top</a>)</p>
