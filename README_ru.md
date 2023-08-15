[![Price](https://img.shields.io/badge/price-FREE-0098f7.svg)](https://github.com/wladradchenko/wunjo.wladradchenko.ru/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/librosa.svg)](https://badge.fury.io/py/librosa)
[![GitHub package version](https://img.shields.io/github/v/release/wladradchenko/wunjo.wladradchenko.ru?display_name=tag&sort=semver)](https://github.com/wladradchenko/wunjo.wladradchenko.ru)
[![License: MIT v1.0](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/wladradchenko/wunjo.wladradchenko.ru/blob/main/LICENSE)

<p align="right">(<a href="README.md">EN</a>)</p>
<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru">
    <img src="example/man.gif" alt="Logo" width="180" height="180">
  </a>

  <h3 align="center">Wunjo AI</h3>

  <p align="center">
    Документация о проекте
    <br/>
    <br/>
    <br/>
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues">Запросить функцию</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## О проекте

Приложение для синтеза речи из текста и распознования речи в текст, а также создания deepfake анимации из изображений. Одной из уникальных особенностей этого приложения является возможность создавать мультидиалоги с несколькими голосами, а количество используемых символов не ограничено, в отличие от аналогичных веб-приложений. Вы также можете проговаривать текст в режиме реального времени, и приложение распознает его по аудио. Эта функция отлично подходит для диктовки текста вместо того, чтобы набирать его вручную.

В целом, это настольное приложение с нейронными сетями представляет собой удобный и мощный инструмент для всех, кто нуждается в синтезе речи, распознавании голоса в текст и создании deepfake анимации. Лучше всего то, что приложение бесплатно, устанавливается локально и проста в использовании! А применить вы его можете, в озвучке роликов, книг, игр, итд.

<!-- FEATURES -->
## Запуск

Требуется 3.8 <= [Python](https://www.python.org/downloads/) <=3.10 и [ffmpeg](https://ffmpeg.org/download.html).

Создать виртуальную среду и активировать:

```
python -m venv venv
source venv/bin/activate
```

Установить зависимости:

```
pip install -r requirements.txt
```

Внимание! При первом запуске синтеза видео, будут скачаны модели в .wunjo/talker/checkpoints и .wunjo/talker/gfpgan в размере 5Гб. Это может занять длительное время.

Необходимо перейти в директорию portable, чтобы использовать briefcase:
```
cd portable
```

Запустить:
```
briefcase dev
```

Дополнительно, вы можете создать build
```
briefcase build
```

Запуск build
```
briefcase run
```

Для создания установщика:
```
briefcase package
```

Подробнее в документации [BeeWare](https://beeware.org/project/projects/tools/briefcase)

<!-- EXAMPLE -->
## Расширения

Функционал программы может дополняться пользовательскими расширениями. Пример расширения и формат по [ссылке](https://github.com/wladradchenko/advanced.wunjo.wladradchenko.ru/blob/main/README_ru.md)

Доступный список расширений по [ссылке](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/extensions.json). 

<!-- DOWNLOAD -->
## Готовые сборки

[Ubuntu / Debian Stable v1.2](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.2.0.deb)

[Ubuntu / Debian Extensions v1.3](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.3.1.deb)

```
// Для создания анимации понадобится установить ffmpeg
sudo apt install ffmpeg

// Установка приложения
sudo dpkg -i wunjo_{vesrion}.deb

// Внимание! При первом запуске синтеза видео, будут скачаны модели в .wunjo/talker/checkpoints и .wunjo/talker/gfpgan в размере 5Гб. Это может занять длительное время.

// Удаление приложения
sudo dpkg -r wunjo

// Удаление кеша
rm -rf ~/.wunjo
```

[MacOS Stable v1.2](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.2.0.zip)

[MacOS Extensions v1.3](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.3.1.zip)

```
// Для создания анимации понадобится установить ffmpeg
brew install ffmpeg 

// Разархивировать приложение
unzip wunjo_macos_{vesrion}.zip

// Внимание! При первом запуске синтеза видео, будут скачаны модели в .wunjo/talker/checkpoints и .wunjo/talker/gfpgan в размере 5Гб. Это может занять длительное время.

// Удаление кеша
rm -rf ~/.wunjo
```

[Windows Extensions v1.3](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/windows/wunjo_1.3.2.msi)

```
// Для создания анимации понадобится установить ffmpeg, после добавить путь в переменная среды
setx PATH "%PATH%;C:\path\to\ffmpeg\bin"

// Установщик
wunjo_{vesrion}.msi

// Внимание! При первом запуске синтеза видео, будут скачаны модели в .wunjo/talker/checkpoints и .wunjo/talker/gfpgan в размере 5Гб. Это может занять длительное время. 
// Если у вас стоит файрвол, он может заблокировать автоматическое скачивание моделей, вы их можете скачать самостоятельно из репозитория. 

// Важно! Как настроить deepfake для Windows. Необходимо дать права на чтение моделей нейронной сети в папке gfpgan, после того, как модели будут установлены! Без этой настройки, результат генерации deepfake будет "Лицо не найдено".

icacls "%USERPROFILE%/.wunjo/deepfake/gfpgan/weights/*.pth" /grant:r "Users":(R,W)

// Удаление кеша
%USERPROFILE%/.wunjo
```

<!-- EXAMPLE -->
## Пример

### Анимация лица из изображения

<div align="center">
  <table>
  <tr>
    <th>Оригинал</th>
    <th>Движение лица + Улучшение лица</th>
    <th>Фиксированное лицо + Улучшение лица</th>
  </tr>
  <tr align="center">
    <td><img src="example/samurai_original.gif" alt="original" width="228" height="360"></td>
    <td><img src="example/samurai_move_enhancer.gif" alt="move_enhancer" width="228" height="360"></td>
    <td><img  src="example/samurai_static_enhancer.gif" alt="static_enhancer" width="228" height="360"></td>
  </tr>
</table>
</div>

### Анимация рта из видео

<div align="center">
  <table>
  <tr>
    <th>Оригинал</th>
    <th>Анимация рта</th>
    <th>Анимация рта + Улучшение лица</th>
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
Приложение поставляется со встроенной поддержкой следующих языков: английский, русский, китайский, португальский и корейский.

Если вы хотите добавить новый язык:

Перейдите к `.wunjo/settings/settings.json`.
Добавьте желаемый язык в формате: `"default_language": {"name": "code"}`.
Чтобы найти соответствующий код для вашего языка, обратитесь к языковым кодам [Google Cloud Translate Language Codes](https://cloud.google.com/translate/docs/languages).
<!-- TRANSLATION -->

<!-- UPDATE -->
Обновление 1.2.0

- [x] Уменьшить размер приложения
- [x] Добавить функцию скачивания моделей на выбор
- [x] Добавить контроль поворота головы по оси Y (продвинутые опции для создания анимации)
- [x] Добавить контроль поворота головы по оси X (продвинутые опции для создания анимации)
- [x] Добавить контроль поворота головы по оси Z (продвинутые опции для создания анимации)
- [x] Добавить улучшения качества фона (продвинутые опции для создания анимации)
- [x] Добавить контроль мимики говорения (продвинутые опции для создания анимации)
- [x] Сделать билды

Обновление 1.3.2

- [x] Добавлена поддержка расширений (любой разработчик может создавать расширения без открытия основного кода)
- [x] Сделать билды
- [x] Исправлены ошибки с deepfake в Windows

Обновление 1.4.0
- [x] Добавить дипфейк для работы с исходным видео, который синхронизирует движение губ под аудио
- [x] Добавлен выбор видеофрагмента для дипфейка в зависимости от длины звуковой дорожки
- [x] Добавлена возможность смены каталога для папки кэша .wunjo.
- [x] Добавить перевод приложения на разные языки
- [x] Обучить и добавить модели TTS для английской речи
- [x] Добавить идентификацию языка языковой модели в интерфейсе
- [x] Добавить возможность говорить на английском по русской модели и говорить на русском по английской модели
- [x] Добавить возможность использования синтеза речи на пользовательских моделях TTS (ru, en)
- [x] Добавлено оповещение об обновлении
- [x] Интеграция ссылок на обучающее видео и вики-страницу

Обновление 1.5.0
- [ ] Разработать модуль для смены лица с исходного лица на целевое (Face swap).
- [ ] Интегрировать ссылки для обучения видео по функциям
- [ ] Добавлено создание дипфейкового видео с помощью текстовых подсказок

<!-- VIDEO -->
## Видео

[![Watch the video](https://img.youtube.com/vi/oHQR1Zx6YOk/hqdefault.jpg)](https://youtu.be/oHQR1Zx6YOk)

[![Install tutorial on Windows](https://img.youtube.com/vi/2qIpJYhOL2U/hqdefault.jpg)](https://youtu.be/2qIpJYhOL2U)

<!-- CONTACT -->
## Контакт

Автор: [Wladislav Radchenko](https://github.com/wladradchenko/)

Почта: [i@wladradchenko.ru](i@wladradchenko.ru)

Проект: [https://github.com/wladradchenko/wunjo.wladradchenko.ru](https://github.com/wladradchenko/wunjo.wladradchenko.ru)

Сайт приложения: [wladradchenko.ru/wunjo](https://wladradchenko.ru/wunjo)

<!-- PREMISE -->
## Предпосылки

Wunjo (Ву́ньо) происходит из древнего рунического алфавита и представляет радость и удовлетворение, что может быть связано с идеей использования приложения для создания увлекательной и выразительной речи. Вуньо (ᚹ) — восьмая руна старшего и англосаксонского футарка. До введения буквы W в латинский алфавит вместо неё в английском языке использовалась буква Ƿynn (Ƿƿ), происходящая от этой руны.

<!-- CREDITS -->
## Зависимости

* Tacatron 2 - https://github.com/NVIDIA/tacotron2
* Waveglow - https://github.com/NVIDIA/waveglow
* Flask UI - https://github.com/ClimenteA/flaskwebgui
* BeeWare - https://beeware.org/project/projects/tools/briefcase/
* Sad Talker - https://github.com/OpenTalker/SadTalker
* Wav2lip - https://github.com/Rudrabha/Wav2Lip
* Face Utils - https://github.com/xinntao/facexlib
* Face Enhancement - https://github.com/TencentARC/GFPGAN
* Image/Video Enhancement - https://github.com/xinntao/Real-ESRGAN


<p align="right">(<a href="#top">вернуться наверх</a>)</p>
