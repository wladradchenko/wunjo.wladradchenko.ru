<p align="right">(<a href="README_en.md">RU</a>)</p>
<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru">
    <img src="icons/man.gif" alt="Logo" width="150" height="150">
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
wget появится позже
```

Unzip archive:
```
The folder data to voiceai.wladradchenko.ru/portable/src/backend
The file config.yaml to voiceai.wladradchenko.ru/portable/src/backend/config.yaml
The dictionary stress.dict to voiceai.wladradchenko.ru/portable/src/backend/tps/data
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
Ubuntu / Debian - https://wladradchenko.ru/voice/установка

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

<p align="right">(<a href="#top">to top</a>)</p>
