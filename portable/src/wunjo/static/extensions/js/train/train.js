///CREATE GENERAL TRAIN BUTTON///
window.onload = function () {
  const menubarGeneral = document.getElementById("menubar");

  // Create the 'a' element
  const trainButton = document.createElement("a");
  trainButton.id = "train-window";
  trainButton.style.color = "black";
  trainButton.style.width = "3.2vw";
  trainButton.style.height = "3.2vw";
  trainButton.style.fontSize = "1.5rem";
  trainButton.style.display = "none";
  trainButton.title = "Панель обучения";
  trainButton.addEventListener("click", (event) =>
    trainWindow(event.currentTarget)
  );

  // Create the 'i' element and append it to the 'a' element
  const icon = document.createElement("i");
  icon.className = "fa-solid fa-brain";
  trainButton.appendChild(icon);

  // Append the 'a' element to the 'menubarGeneral'
  menubarGeneral.appendChild(trainButton);
  document
    .getElementById("a-change-processor")
    .addEventListener("click", (event) => {
      availableFeaturesByCUDA(trainButton);
    });
  availableFeaturesByCUDA(trainButton);
};
///CREATE GENERAL TRAIN BUTTON///

function trainWindow(event) {
  var trainPanel = introJs();
  trainPanel.setOptions({
    steps: [
      {
        element: event,
        title: "Панель обучения голоса",
        position: "right",
        intro: `
                    <div style="padding: 5pt;display: flex;flex-direction: column;width: 370px;">
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
                    <div style="display: flex;justify-content: space-between;">
                        <div style="padding: 5pt;display: flex;flex-direction: column;width: 50%;">
                            <label for="speech-train-select-lang">Язык аудио файлов</label>
                            <select id="speech-train-select-lang" style="margin: 0;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                                <option value="en" selected>Английский</option>
                                <option value="ru">Русский</option>
                            </select>
                        </div>
                        <div style="padding: 5pt;display: flex;flex-direction: column;width: 50%;">
                            <label for="speech-train-batch-size">Batch size</label>
                            <select id="speech-train-batch-size" style="margin: 0;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                                <option value="8">8</option>
                                <option value="16">16</option>
                                <option value="32" selected>32</option>
                                <option value="64">64</option>
                            </select>
                        </div>
                    </div>
                    <i style="margin: 5pt;font-size: 10pt;"><b>Примечание:</b><a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki" target="_blank" rel="noopener noreferrer"> Подробнее об обучение в документации</a></i>
                    <fieldset style="margin: 5pt;padding: 10pt;">
                        <legend>Соотношение данных</legend>
                        <input style="width: 100%;" class="range speech-train-split" type="range" min="0" max="100" step="1" value="80">
                        <div style="display: flex;justify-content: space-between;font-size: 10pt;color: #686868;">
                            <div>Train</div>
                            <div class="speech-train-split-value">80%</div>
                            <div>Test</div>
                        </div>
                    </fieldset>
                </div>
                <p id="message-train-speech" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="sendDataToTrain(this.parentElement, 'tacotron2');">Начать обучение</button>
                `,
      },
    ],
    showButtons: false,
    showStepNumbers: false,
    showBullets: false,
    nextLabel: "Продолжить",
    prevLabel: "Вернуться",
    doneLabel: "Закрыть",
  });
  trainPanel.start();
}

function sendDataToTrain(elem, trainType) {
  // If process is free
  fetch("/synthesize_process/")
    .then((response) => response.json())
    .then((data) => {
      // Call the async function
      processAsyncTrainParam(data, elem, trainType)
        .then(() => {
          console.log("Start to fetch data to training");
        })
        .catch((error) => {
          console.log("Error to fetch data to training");
          console.log(error);
        });
    });
}

async function processAsyncTrainParam(data, elem, trainType) {
  if (data.status_code === 200) {
    const trainAudioPath = elem.querySelector("#speech-train-audio-path");
    const messageTrainSpeech = elem.querySelector("#message-train-speech");
    // Check if trainAudioPath has an input value or is empty
    if (!trainAudioPath.value) {
      var messageSetP = await translateWithGoogle(
        "Путь к аудио пуст.",
        "auto",
        targetLang
      );
      messageTrainSpeech.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
      return;
    }

    const trainMarkPath = elem.querySelector("#speech-train-mark-path");
    // Check if trainMarkPath has an input value or is empty
    if (!trainMarkPath.value) {
      var messageSetP = await translateWithGoogle(
        "Путь к файлу разметки пуст.",
        "auto",
        targetLang
      );
      messageTrainSpeech.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
      return;
    }

    const trainCheckpointPathField = elem.querySelector(
      "#speech-train-checkpoint-info"
    );
    let trainCheckpointPath = elem.querySelector(
      "#speech-train-checkpoint-path"
    );

    if (!trainCheckpointPathField.checked) {
      trainCheckpointPath.value = "";
    }

    const trainSplit = elem.querySelector(".speech-train-split");

    const trainSelectLang = elem.querySelector("#speech-train-select-lang");
    // Get value from the current option
    const selectedLang =
      trainSelectLang.options[trainSelectLang.selectedIndex].value;

    const trainSelectBatchSize = elem.querySelector("#speech-train-batch-size");
    // Get value from the current option
    const selectedBatchSize =
      trainSelectBatchSize.options[trainSelectBatchSize.selectedIndex].value;

    const trainParams = {
      checkpoint: trainCheckpointPath.value,
      audio_path: trainAudioPath.value,
      mark_path: trainMarkPath.value,
      train_split: trainSplit.value,
      language: selectedLang,
      batch_size: selectedBatchSize,
      train_type: trainType,
    };

    try {
      // Close window
      const closeIntroButton = document.querySelector(".introjs-skipbutton");
      closeIntroButton.click();
      const response = await fetch(`/training_voice`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(trainParams),
      });
    } catch (error) {
      console.error("Error during fetch:", error);
    }
  }
}
