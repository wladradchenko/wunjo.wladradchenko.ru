<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru">
    <img src="icons/logo.png" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">Voice AI</h3>

  <p align="center">
    Документация о проекте
    <br/>
    <br/>
    <br/>
    <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru/issues">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru/issues">Запросить функцию</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## О проекте

Приложение для синтеза речи из текста и распознования речи в текст. Одной из уникальных особенностей этого приложения является возможность создавать мультидиалоги с несколькими голосами, а количество используемых символов не ограничено, в отличие от аналогичных веб-приложений. Вы также можете проговаривать текст в режиме реального времени, и приложение распознает его по аудио. Эта функция отлично подходит для диктовки текста вместо того, чтобы набирать его вручную.

В целом, это настольное приложение представляет собой удобный и мощный инструмент для всех, кто нуждается в синтезе речи и распознавании голоса в текст. Лучше всего то, что приложение бесплатно, устанавливается локально и проста в использовании! А применить вы его можете, в озвучке роликов, книг, игр, итд.


Веб-сервис на Aiohttp созданный для работы с [Tinkoff Invest](https://github.com/Tinkoff/invest-python), при помощи AsyncClient.
Позволяет распределить указанный капитал на пирог выбранного индекса Московской биржи из открытого API. Работает сервис с акциями Т+: Акции и ДР - безадрес в RUB.

Если происходит работа со счетом Тинькофф, тогда распределение капитала учитывает бумаги, которые находятся в портфеле. Если они не входят в индекс, тогда учитываются к продаже.

Возможности открытой версии:
* Распределение капитала на пирог,
* Перераспределение акций на пирог (продажа / покупка),
* История сделок счёта,
* Создание пользовательского индекса,
* Фоновые задачи для получения исторических свечей и работы с ними (применение стратегии трейдинга/создание графиков/расчет индикаторов),
* Управление приложением через Telegram чат.


<!-- FEATURES -->
## Запуск

```
pip install -r requirements.txt
```

В conf.toml
```
database="sqlite.db"

[tinkoff]
readonly = "TOKEN"  # token Tinkoff readonly

[telegram]
token = "TOKEN"  # token telegram bot
chat_id = 0  # chat for messages from bot
```

Как получить Tinkoff [токен](https://tinkoff.github.io/investAPI/token/).
Как получить Telegram [токен](https://core.telegram.org/bots/api#authorizing-your-bot).
Как узнать [Chat ID](https://core.telegram.org/bots/api#getchatmember).

Запуск
```
python run.py
```

<!-- VIDEO -->
## Видео

[![Watch the video](https://img.youtube.com/vi/aekVTaJHzqY/maxresdefault.jpg)](https://youtu.be/aekVTaJHzqY)

<!-- CONTACT -->
## Контакт

Автор: [Wladislav Radchenko](https://github.com/wladradchenko/)

Почта: [i@wladradchenko.ru](i@wladradchenko.ru)

Проект: [https://github.com/wladradchenko/voiceai.wladradchenko.ru](https://github.com/wladradchenko/voiceai.wladradchenko.ru)

Сайт приложения: [wladradchenko.ru/voice](https://wladradchenko.ru/voice)

<p align="right">(<a href="#top">вернуться наверх / back to top</a>)</p>

<!-- CREDITS -->
## Зависимости

Tacatron 2 - https://github.com/NVIDIA/tacotron2
Waveglow - https://github.com/NVIDIA/waveglow
Flask UI - https://github.com/ClimenteA/flaskwebgui
BeeWare - https://beeware.org/project/projects/tools/briefcase/

# voiceai.wladradchenko.ru

написать откуда скачать модель и конфиг

написать про то, что нужно скачать voiceai.wladradchenko.ru/portable/src/backend/tps/data/stress.dict
