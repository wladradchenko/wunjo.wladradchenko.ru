function sendSpeechTrainParameters(param) {
    var speechTrainAudioPath = param.querySelector("#speech-train-audio-path")
    var speechTrainMarkPath = param.querySelector("#speech-train-mark-path")
    var speechTrainСheckpointPath = param.querySelector("#speech-train-checkpoint-path")
    var speechTrainСheckpointPathChecked = param.querySelector("#speech-train-checkpoint-path-field")
    if (!speechTrainСheckpointPathChecked.checked) {
        speechTrainСheckpointPath = "";
    }
    var speechTrainSplit = param.querySelector(".speech-train-split")
    var toggleDivLanguage= param.querySelector('.toggle-div-language');
    var toggleDivTypeSpeechTrain = param.querySelector('.toggle-div-type-speech-train');

    var valueDivLanguage;
    if (toggleDivLanguage.classList.contains('active')) {
        valueDivLanguage = "en";
    } else {
        valueDivLanguage = "ru";
    };

    var valueDivTypeSpeechTrain;
    var configTypeSpeechTrain;
    if (toggleDivTypeSpeechTrain.classList.contains('active')) {
        configTypeSpeechTrain = '/extensions/static/advanced/static/config/tacotron2/hparams.yaml'
        valueDivTypeSpeechTrain = "tacotron2";
    } else {
        configTypeSpeechTrain = '/extensions/static/advanced/static/config/waveglow/config.json'
        valueDivTypeSpeechTrain = "waveglow";
    };

    loadYamlFile(configTypeSpeechTrain)
      .then(data => {
        if (data) {
            var oneSpeechTrainSend = {
                                "checkpoint": speechTrainСheckpointPath.value,
                                "audio_path": speechTrainAudioPath.value,
                                "mark_path": speechTrainMarkPath.value,
                                "train_split": speechTrainSplit.value,
                                "language": valueDivLanguage,
                                "config": data
                            };

            // Get a reference to the #status-message element
            const statusMessage = document.getElementById('status-message');
            statusMessage.innerText = "Подождите... Происходит обработка";

            const closeIntroButton = document.querySelector('.introjs-skipbutton');
            closeIntroButton.click();

            fetch(`/${valueDivTypeSpeechTrain}/`, {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(oneSpeechTrainSend)
            })
        }
    });
};

function buttonRunClick(param) {
    sendSpeechTrainParameters(param);
};

function loadYamlFile(filepath) {
    return fetch(filepath)
        .then(response => response.text())
        .then(data => {
            return data;
        })
        .catch(error => console.error(error));
}

///TRAIN PANEL///
window.addEventListener("DOMContentLoaded", (event) => {
    const processor = document.getElementById('a-change-processor');
    processor.addEventListener('click', (event) => {
      event.preventDefault(); // prevent the link from following its href attribute
      fetch('/change_processor', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
          var processorMessage = document.getElementById('status-message');
          var linkSpeechTrain = document.getElementById('a-speech-train');
          currentProcessor(linkSpeechTrain);
          if (data.current_processor === 'cpu') {
            processor.style.color = 'red';
            processorMessage.innerText = 'Расчеты будут производиться на процессоре CPU';
          } else {
            processor.style.color = 'green';
            processorMessage.innerText = 'Расчеты будут производиться на графическом процессоре GPU';
          }
        })
        .catch(error => {
            console.log(error);
        });
    });

    const linkSpeechTrain = document.getElementById('a-speech-train');

    function currentProcessorSpeechTrain(elem = undefined) {
      fetch('/current_processor', { method: 'GET' })
        .then(response => response.json())
        .then(data => {
          var deviceStatus = data.current_processor;
          var deviceUpgrade = data.upgrade_gpu;

          if (processor) {
              if (!deviceUpgrade) {
                processor.style.display = "none";
              }

              if (deviceStatus == 'cuda') {
                processor.style.color = 'green';
              }
          };

          if (elem && deviceStatus == 'cpu') {
            elem.style.display = 'none';
          } else if (elem) {
            elem.style.display = 'block';
          };
        })
        .catch(error => {
          console.log(error);
        });
    };

    currentProcessorSpeechTrain(linkSpeechTrain);

    var speechTrain = introJs();
    speechTrain.setOptions({
        steps: [
            {
                element: '#a-speech-train',
                title: 'Панель обучения',
                position: 'right',
                intro: `
                        <div>
                            <div style="padding: 5pt;display: flex;flex-direction: column;">
                                <div style="margin-bottom:5pt;">
                                  <input onclick="document.getElementById('speech-train-checkpoint-path-field').style.display = this.checked ? 'block' : 'none';" type="checkbox" id="speech-train-checkpoint-info" name="speech-train-checkpoint">
                                  <label for="speech-train-checkpoint">Предобученная модель</label>
                                </div>
                                <div id="speech-train-checkpoint-path-field" style="display:none;margin-top:5pt;">
                                    <label for="speech-train-checkpoint-path">Путь до модели</label>
                                    <input type="text" id="speech-train-checkpoint-path" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                                </div>
                            </div>
                            <div style="padding: 5pt;display: flex;flex-direction: column;">
                                <label for="speech-train-audio-path">Путь до аудио файлов</label>
                                <input type="text" id="speech-train-audio-path" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                            </div>
                            <div style="padding: 5pt;display: flex;flex-direction: column;">
                                <label for="speech-train-mark-path">Путь до файла разметки</label>
                                <input type="text" id="speech-train-mark-path" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                            </div>
                            <div id="speech-train-card-toggle" style="display: flex;justify-content: space-around;margin-bottom: 20pt;">
                                <div style="margin-right: 15pt;" class="toggle-button toggle-button-language">Язык аудио
                                  <div style="margin-top: 10pt;" title="Флаг выбора языка датасета" class="button toggle-div-language">
                                    <div class="circle toggle-div-language">
                                        <i style="display: none;font-style: inherit;font-weight: bolder;" class="toggle-button-language-en toggle-div-language">en</i>
                                        <i class="toggle-button-language-ru toggle-div-language" style="font-style: inherit;font-weight: bolder;">ru</i>
                                    </div>
                                  </div>
                                </div>
                                <div style="margin-right: 15pt;" class="toggle-button toggle-button-type-speech-train">Модель
                                  <div style="margin-top: 10pt;" title="Флаг выбора типа модели" class="button toggle-div-type-speech-train">
                                    <div class="circle toggle-div-type-speech-train">
                                        <i style="display: none" class="fa-solid fa-pepper-hot toggle-button-tacotron2 toggle-div-type-speech-train"></i>
                                        <i class="fa-solid fa-wave-square toggle-button-waveglow toggle-div-type-speech-train"></i>
                                    </div>
                                  </div>
                                </div>
                            </div>
                            <label>Соотношение данных
                                <input style="width: 80pt;margin-top: 10pt;margin-bottom: 10pt;width: 100%;" class="range speech-train-split" type="range" min="0" max="100" step="1" value="80">
                                <div style="display: flex;justify-content: space-between;margin-bottom: 15pt;font-size: 10pt;color: #686868;">
                                    <div>Train</div>
                                    <div class="speech-train-split-value">80%</div>
                                    <div>Test</div>
                                </div>
                            </label>
                        </div>
                        <button value="waveglow" class="introjs-button" onclick="buttonRunClick(this.parentElement);" title="Начать обучение" id="button-run-speech-train" style="right: 0;left: 0;display: flex;justify-content: center;">Начать тренировать Waveglow</button>
                        `,
            },
        ],
          showButtons: false,
          showStepNumbers: false,
          showBullets: false,
          nextLabel: 'Продолжить',
          prevLabel: 'Вернуться',
          doneLabel: 'Закрыть'
    });

    linkSpeechTrain.addEventListener('click', (event) => {
        speechTrain.start();
        /// LISTEN CLICK ON TRAIN ELEMENTS ///
        const speechTrainCards = document.querySelector('#speech-train-card-toggle');
        const buttonSpeechTrain = document.querySelector('#button-run-speech-train');

        /// CHANGE TRAIN LANGUAGE TOGGLE ///
        function changeLanguageSpeechTrain(event) {
            if (event.target.classList.contains('toggle-div-language')) {
              var toggle = event.target.closest('.button');
              toggle.classList.toggle('active');
              var toggleIconEn = toggle.querySelector('.toggle-button-language-en');
              var toggleIconRu = toggle.querySelector('.toggle-button-language-ru');
              var isOn = toggle.classList.contains('active');
              if (isOn) {
                toggleIconEn.style.display = 'inline';
                toggleIconRu.style.display = 'none';
              } else {
                toggleIconEn.style.display = 'none';
                toggleIconRu.style.display = 'inline';
              }
            };
        };
        /// CHANGE TRAIN LANGUAGE TOGGLE ///

        /// CHANGE TRAIN TYPE MODEL TOGGLE ///
        function changeTypeModelSpeechTrain(event) {
            if (event.target.classList.contains('toggle-div-type-speech-train')) {
              var toggle = event.target.closest('.button');
              toggle.classList.toggle('active');
              var toggleIconTacotron2 = toggle.querySelector('.toggle-button-tacotron2');
              var toggleIconWaveglow = toggle.querySelector('.toggle-button-waveglow');
              var isOn = toggle.classList.contains('active');
              if (isOn) {
                toggleIconTacotron2.style.display = 'inline';
                toggleIconWaveglow.style.display = 'none';
                buttonSpeechTrain.innerText = 'Начать тренировать Tacotron2';
                buttonSpeechTrain.value = 'tacotron2';
              } else {
                toggleIconTacotron2.style.display = 'none';
                toggleIconWaveglow.style.display = 'inline';
                buttonSpeechTrain.innerText = 'Начать тренировать Waveglow';
                buttonSpeechTrain.value = 'waveglow';
              }
            };
        };
        /// CHANGE TRAIN TYPE MODEL TOGGLE ///

        /// LISTEN CLICK ON ELEMENTS ///
        function handleSpeechTrainButtonClick(event) {
          if (event.target.classList.contains('toggle-div-language')) {
            changeLanguageSpeechTrain(event);
          } else if (event.target.classList.contains('toggle-div-type-speech-train')) {
            changeTypeModelSpeechTrain(event);
          }
          // console.log(event.target.classList);
        }

        speechTrainCards.addEventListener('click', handleSpeechTrainButtonClick);
        /// LISTEN CLICK ON TRAIN ELEMENTS ///

        /// SPLIT TRAIN DATA ///
        var splitSpeechTrain = document.querySelector(".speech-train-split");
        var splitSpeechTrainValue = document.querySelector(".speech-train-split-value");
        splitSpeechTrain.addEventListener("input", function() {
          splitSpeechTrainValue.innerText = `${splitSpeechTrain.value} %`;
        });
        /// SPLIT TRAIN DATA ///
    });
});
///TRAIN PANEL///
