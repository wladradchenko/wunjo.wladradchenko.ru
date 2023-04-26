<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/voiceai.wladradchenko.ru">
    <img src="logo/main.png" alt="Logo" width="150" height="150">
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
## Видео / Video

[![Watch the video](https://img.youtube.com/vi/aekVTaJHzqY/maxresdefault.jpg)](https://youtu.be/aekVTaJHzqY)

<!-- CONTACT -->
## Контакт / Contact

Автор / Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Почта / Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Проект / Code: [https://github.com/wladradchenko/voiceai.wladradchenko.ru](https://github.com/wladradchenko/voiceai.wladradchenko.ru)

Сайт приложения / Project web-site: [wladradchenko.ru/voice](https://wladradchenko.ru/voice)

<p align="right">(<a href="#top">вернуться наверх / back to top</a>)</p>







# voiceai.wladradchenko.ru

написать откуда скачать модель и конфиг

написать про то, что нужно скачать voiceai.wladradchenko.ru/portable/src/backend/tps/data/stress.dict
