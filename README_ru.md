[![Price](https://img.shields.io/badge/price-FREE-0098f7.svg)](https://github.com/wladradchenko/wunjo.wladradchenko.ru/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
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
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki">Документация о проекте</a>
    <br/>
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/discussions">Предложить улучшения</a>
    ·
    <a href="https://youtube.com/playlist?list=PLJG0sD6007zFJyV78mkU-KW2UxbirgTGr&feature=shared">Видео курс</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## О проекте

Используйте возможности нейронных сетей с помощью Wunjo AI для множества задач — от синтеза речи и клонировании голоса до создания дипфейк анимации и удаления объектов с видео.

**Что умеет?**

- **Синтез речи:** Легко преобразуйте текст в человеческую речь.
- **Клонирование голоса:** Клонируйте голоса из предоставленных аудиофайлов или напрямую записывайте свой голос в приложении для клонирования в реальном времени.
- **Многоязычная поддержка:** В настоящее время поддерживается английский, китайский для клонирования голоса (с любого языка аудио) и синтеза речи на английском и русском языках - в ближайшее время планируется расширить приложение, добавить модель клонирования голоса на русский язык.
- **Распознавание речи в реальном времени:** Диктуйте текст и мгновенно получайте транскрипцию. Эффективный инструмент для создания контента без помощи рук.
- **Создание мультидиалогов:** Создавайте мультидиалоги, используя персонажей с разными голосовыми профилями вы можете создавать диалоги. __Если вам не хватит голосов, вы сможете в приложении обучить модель на своем голосе.__
- **Дипфейк анимация:**
   - Анимируйте лица, используя всего одну фотографию в сочетании со звуком.
   - Обеспечьте точную синхронизацию губ со звуком с помощью функции дипфейка губ.
   - С легкостью меняйте лица в видео, GIF-файлах и фотографиях, используя всего одну фотографию, с помощью функции «Замена лиц».
   - Экспериментальная функция. Измените эмоции человека на видео, при помощи текстового описания.
- **Инструмент AI Retouch:** Улучшите качество своих видео, удалив ненужные объекты или улучшив качество дипфейков.

**Приложения:**
От озвучки в рекламных роликах до озвучки персонажей в играх, от аудиокниг до забавных дипфейковых проектов — Wunjo AI предлагает безграничные возможности, и все это бесплатно и локально на вашем устройстве.

**Почему стоит выбрать Wunjo AI?:**

- **Все в одном:** Комплексный инструмент, отвечающий потребностям как голосового, так и визуального искусственного интеллекта.
- **Удобство использования**: Предназначено для всех, от новичков до профессионалов.
- **Конфиденциальность прежде всего:** Работает локально на вашем рабочем столе, обеспечивая конфиденциальность ваших данных.
- **Открытый исходный код и бесплатно:** Воспользуйтесь преимуществами улучшений, предложенных сообществом, и наслаждайтесь приложением без каких-либо затрат.

Шагните в будущее творчества на основе искусственного интеллекта с Wunjo AI.

<!-- FEATURES -->
## Запуск

Требуется [Python](https://www.python.org/downloads/) версии 3.10 и [ffmpeg](https://ffmpeg.org/download.html).

Создать виртуальную среду и активировать:

```
python -m venv venv
source venv/bin/activate
```

Установить зависимости:

```
pip install -r requirements.txt
```

Необходимо перейти в директорию portable, чтобы использовать briefcase:

```
cd portable
```

Запустить:

```
briefcase dev
```

При первом запуске будет выполнен автоматический перевод на выбранный язык, это может занять некоторое время. Дополнительно, вы можете создать build

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

<!-- DOWNLOAD -->
## Готовые сборки

[Ubuntu / Debian Stable v1.4 (GPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.4.0.deb)

[Ubuntu / Debian Beta v1.5 (GPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/linux/wunjo_1.5.1.deb)

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

// Если переключение на графический процессор для вас недоступно, см. документацию по установке драйверов CUDA
```

[MacOS Stable v1.4 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.4.0.zip)

[MacOS Beta v1.5 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/macos/wunjo_macos_1.5.1.zip)

```
// Для создания анимации понадобится установить ffmpeg
brew install ffmpeg 

// Разархивировать приложение
unzip wunjo_macos_{vesrion}.zip

// Внимание! При первом запуске синтеза видео, будут скачаны модели в .wunjo/talker/checkpoints и .wunjo/talker/gfpgan в размере 5Гб. Это может занять длительное время.

// Удаление кеша
rm -rf ~/.wunjo

// Сборка сделана на библиотеках с ЦПУ. Как настроить использование графического процессора и увеличить скорость обработки в несколько раз, смотрите в документации.
```

[Windows Stable v1.4 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/windows/wunjo_1.4.0.msi)

[Windows Beta v1.5 (CPU version)](https://wladradchenko.ru/static/wunjo.wladradchenko.ru/build/windows/wunjo_1.5.1.msi)

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

// Сборка сделана на библиотеках с ЦПУ. Как настроить использование графического процессора и увеличить скорость обработки в несколько раз, смотрите в документации.
```

<!-- EXAMPLE -->
## Пример

### Синтез речи и клонирование голоса

- [Русский синтезированный голос из текста](https://soundcloud.com/vladislav-radchenko-234338135/russian-voice-text-synthesis?si=ebfc8ea75d0f4c56a3012ca4fdfb6ab5&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
- [Английский голос клонированный из ранее синтезированного русского голоса](https://soundcloud.com/vladislav-radchenko-234338135/english-voice-clone?si=057718ee0e714e79b2023ce2e37dfb39&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
- [Китайский голос клонированный из ранее синтезированного русского голоса](https://soundcloud.com/vladislav-radchenko-234338135/chinese-voice-clone?si=43d437bbdf4d4d9a80c7a4a0031189c0&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

### Анимация лица из изображения

<div align="center">
  <table>
  <tr>
    <th>Оригинал</th>
    <th>Движение лица + Улучшение лица</th>
    <th>Фиксированное лицо + Улучшение лица</th>
  </tr>
  <tr align="center">
    <td><img src="example/animation_face/samurai_original.gif" alt="original" width="228" height="360"></td>
    <td><img src="example/animation_face/samurai_move_enhancer.gif" alt="move_enhancer" width="228" height="360"></td>
    <td><img  src="example/animation_face/samurai_static_enhancer.gif" alt="static_enhancer" width="228" height="360"></td>
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
    <td><img src="example/animate_mouth/pirate_original.gif" alt="original" width="228" height="228"></td>
    <td><img src="example/animate_mouth/pirate_mouth.gif" alt="move_enhancer" width="228" height="228"></td>
    <td><img  src="example/animate_mouth/pirate_mouth_enhancer.gif" alt="static_enhancer" width="228" height="228"></td>
  </tr>
</table>
</div>

### Замена лица по одному фото

<div align="center">
  <table>
  <tr>
    <th>Оригинал фото</th>
    <th>Оригинал видео</th>
    <th>Замена лица + Улучшение окружения</th>
  </tr>
  <tr align="center">
    <td><img src="example/face_swap/face_swap_photo_original.gif" alt="original" width="203" height="203"></td>
    <td><img src="example/face_swap/face_swap_original.gif" alt="original" width="360" height="203"></td>
    <td><img  src="example/face_swap/face_swap_smith.gif" alt="static_enhancer" width="360" height="203"></td>
  </tr>
</table>
</div>

### Удаление объектов с видео

<div align="center">
  <table>
  <tr>
    <th>Оригинал</th>
    <th>С удалением машины</th>
  </tr>
  <tr align="center">
    <td><img src="example/retouch/remove_object_original.gif" alt="original" width="480" height="270"></td>
    <td><img src="example/retouch/remove_object_retouch.gif" alt="original" width="480" height="270"></td>
  </tr>
</table>
</div>

### Улучшение качества дипфейка ретушью

<div align="center">
  <table>
  <tr>
    <th>Дефектные линии на подбородке после анимации лица</th>
    <th>Ретушь подбородка + Замена лица</th>
  </tr>
  <tr align="center">
    <td><img src="example/retouch/speech_wav2lip.gif" alt="original" width="480" height="270"></td>
    <td><img src="example/retouch/speech_result.gif" alt="original" width="480" height="270"></td>
  </tr>
</table>
</div>

### Emotion deepfake [Experimental]

Это экспериментальная функция, которая находится в стадии разработки, но вы можете прямо сейчас взглянуть на некоторый ее функционал в Wunjo AI.

<div align="center">
  <table>
  <tr>
    <th>Оригинал</th>
    <th>Радость</th>
    <th>Злость</th>
  </tr>
  <tr align="center">
    <td><img src="example/fake_emotion/original.gif" alt="original" width="213" height="213"></td>
    <td><img src="example/fake_emotion/happy.gif" alt="original" width="213" height="213"></td>
    <td><img src="example/fake_emotion/angry.gif" alt="original" width="213" height="213"></td>
  </tr>
  <tr>
    <th>Страх</th>
    <th>Грусть</th>
    <th>Отвращение</th>
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
## Языки

Приложение поставляется со встроенной поддержкой следующих языков: английский, русский, китайский, португальский и корейский.

Если вы хотите добавить новый язык:

Перейдите к `.wunjo/settings/settings.json`.
Добавьте желаемый язык в формате: `"default_language": {"name": "code"}`.
Чтобы найти соответствующий код для вашего языка, обратитесь к языковым кодам [Google Cloud Translate Language Codes](https://cloud.google.com/translate/docs/languages).
<!-- TRANSLATION -->

<!-- UPDATE -->

Обновление 1.5.1
- [x] Добавить голосовой перевод на английский, русский
- [x] Добавить клонирование голоса и автоматический перевод моделей синтеза на выбранный голос
- [x] Улучшить качество речи на русском языке
- [x] Создать хаб для моделей клонирования голоса на различные языки (английский, русский)
- [x] Добавить модуль замены лица для дипфейка на видео/фото с одного фото лица
- [x] Добавить улучшение лица или фона на видео/фото пользователя
- [x] Сделать версию, которая будет включать расширения без модуля загрузки расширений
- [x] Улучшение индикации и перевода текущего прогресса работы приложения
- [x] Добавлен модуль проверки отладки с консолью Python
- [x] Изменить текущий перевод на более быстрый
- [x] Улучшено сообщение о том, что графический процессор недоступен для пользователя
- [x] Добавить модуль изменений эмоций на видео в качестве экспериментальной функции
- [x] Обновление гида по приложению
- [x] Добавить AI-ретуши видео, которая позволяет удалять объекты с видео или улучшать качество дипфейка удаляя неровности
- [x] Добавлена работа с китайской грамматикой и модель обучена для клонирования голоса на китайском языке

Обновление 1.6.0
- [ ] Добавить создание дипфейкового видео с помощью текстовых подсказок (Посмотреть сколько место займет, и есть ли отдельные библиотеки, чтобы за предел 2 Гб не выйти)
- [ ] Показывать пользователю, сколько места на диске для папок tmp и результатов синтеза
- [ ] Имитация эмоций синтезированным голосом
- [ ] Обучить модель для клонирования голоса на русский язык
- [ ] Может генерацию музыки добавить? (Посмотреть сколько место займет, и есть ли отдельные библиотеки, чтобы за предел 2 Гб не выйти)
- [ ] Может DragGAN добавить? (Посмотреть сколько место займет, и есть ли отдельные библиотеки, чтобы за предел 2 Гб не выйти)

<!-- VIDEO -->
## Видео

| Что нового?                                                                                              | Как установить на Windows?                                                                                           |
|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [![Watch the video](https://img.youtube.com/vi/vdvf9NxrUC8/hqdefault.jpg)](https://youtu.be/vdvf9NxrUC8) | [![Install tutorial on Windows](https://img.youtube.com/vi/2qIpJYhOL2U/hqdefault.jpg)](https://youtu.be/2qIpJYhOL2U) |

<!-- DONAT -->
## Поддержка

Вы можете поддержать автора проекта на развитии его креативных идей, либо просто угостить [чашкой кофе](https://wladradchenko.ru/donat). Есть и другие способы поддержать развитие проекта, подробнее на [странице](https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki/Поддержите-проект).

<img src="https://github.com/wladradchenko/wunjo.wladradchenko.ru/assets/56233697/acc80acd-0e39-4476-88db-0a10f2098e25" alt="donat" width="250" height="250">

<!-- DONAT -->

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
* Real-Time Voice Cloning - https://github.com/CorentinJ/Real-Time-Voice-Cloning

<p align="right">(<a href="#top">вернуться наверх</a>)</p>
