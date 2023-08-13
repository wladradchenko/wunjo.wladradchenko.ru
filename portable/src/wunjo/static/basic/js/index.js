
var voiceCards = document.querySelector('#voice-cards');
const voiceCardContainer = document.querySelector('.voice-card-container');
const voiceCardTemplate = document.getElementById("voice-card-template");

/// CREATE FIRST AVATAR CARD ///
function firstVoiceCard() {
    var newVoiceCardContainer = voiceCardTemplate.content.cloneNode(true);
    voiceCards.appendChild(newVoiceCardContainer);
}

firstVoiceCard();
/// CREATE FIRST AVATAR CARD ///

/// APPEND NEW AVATAR CARD FROM TEMPLATE ///
function addVoiceCard(event) {
  if (event.target.classList.contains('voice-card-container-plus')) {
    // stop listen file
    if (audioFile && !audioFile.paused) {
        audioFile.pause();
        pauseAllAudioFile();
    }

    var voiceCardContainers = document.querySelectorAll('.voice-card-container');
    var voiceCardContainer = event.target.closest('.voice-card-container');
    var currentVoiceMultiSelect = voiceCardContainer.querySelector('.model-checkboxes');
    currentVoiceMultiSelect.style.display = "none";

    // Set all microphone buttons on off
    for (var i = 0; i < voiceCardContainers.length; i++) {
        var buttonMicrophoneOn = voiceCardContainers[i].querySelector(".fa-microphone");
        var buttonMicrophoneOff = voiceCardContainers[i].querySelector(".fa-microphone-slash");
        buttonMicrophoneOn.style.display = 'inline';
        buttonMicrophoneOff.style.display = 'none';
    }

    // Copy element to append
    // var newVoiceCardContainer = voiceCardContainer.cloneNode(true);
    var newVoiceCardContainer = voiceCardTemplate.content.cloneNode(true);

    // Determine the index of the new container in the list of containers
    var newIndex = Array.from(voiceCardContainers).indexOf(voiceCardContainer) + 1;

    // Set the order of the avatar and textarea based on the index
    var swapped = newIndex % 2 !== 0;
    var voiceCardAvatar = newVoiceCardContainer.querySelector('.voice-card-avatar');
    var voiceCardTextarea = newVoiceCardContainer.querySelector('.voice-card-textarea');
    voiceCardAvatar.style.order = swapped ? 2 : 1;
    voiceCardTextarea.style.order = swapped ? 1 : 2;

    // Set the bubble position based on the index
    var bubble = newVoiceCardContainer.querySelector('.bubble');
    bubble.classList.toggle('bubble-bottom-right', !swapped);
    bubble.classList.toggle('bubble-bottom-left', swapped);
    if (swapped) {
      bubble.style.left = 'auto';
      bubble.style.right = '-70%';
    } else {
      bubble.style.right = 'auto';
      bubble.style.left = '-20%';
    }

    // Insert the new container at the correct index
    voiceCards.insertBefore(newVoiceCardContainer, voiceCardContainers[newIndex]);

    // Remove recognition voice
    recognition.stop();
    isRecording = false;
  };
};
/// APPEND NEW AVATAR CARD FROM TEMPLATE ///

/// REMOVE AVATAR CARD ///
function removeVoiceCard(event) {
  if (event.target.classList.contains('voice-card-container-remove')) {
    // stop listen file
    if (audioFile && !audioFile.paused) {
        audioFile.pause();
        pauseAllAudioFile();
    }

    var voiceCardContainers = document.querySelectorAll('.voice-card-container');
    if (voiceCardContainers.length > 1) {
        var voiceCardContainer = event.target.closest('.voice-card-container');
        voiceCards.removeChild(voiceCardContainer);
    }

    // Remove recognition voice
    recognition.stop();
    isRecording = false;

    // Set all microphone buttons on off
    for (var i = 0; i < voiceCardContainers.length; i++) {
        var buttonMicrophoneOn = voiceCardContainers[i].querySelector(".fa-microphone");
        var buttonMicrophoneOff = voiceCardContainers[i].querySelector(".fa-microphone-slash");
        buttonMicrophoneOn.style.display = 'inline';
        buttonMicrophoneOff.style.display = 'none';
    }
  };
};
/// REMOVE AVATAR CARD ///

/// CHANGE VOLUME TOGGLE ///
function changeVolume(event) {
    if (event.target.classList.contains('toggle-div-voice')) {
      var toggle = event.target.closest('.button');
      toggle.classList.toggle('active');
      var toggleIconVoiceOn = toggle.querySelector('.toggle-button-voice-on');
      var toggleIconVoiceOff = toggle.querySelector('.toggle-button-voice-off');
      var isOn = toggle.classList.contains('active');
      if (isOn) {
        toggleIconVoiceOn.style.display = 'inline';
        toggleIconVoiceOff.style.display = 'none';
        // add code to turn on text-to-voice
      } else {
        toggleIconVoiceOn.style.display = 'none';
        toggleIconVoiceOff.style.display = 'inline';
        // add code to turn off text-to-voice
      }

      // check disable and enable all buttons
      checkToggleVoiceAll();
    };
};
/// CHANGE VOLUME TOGGLE ///

/// USING MICROPHONE TO RECOGNITION VOICE ///
function microphoneRecognition(event) {
  if (event.target.classList.contains("microphone")) {
    // stop listen file
    if (audioFile && !audioFile.paused) {
        audioFile.pause();
        pauseAllAudioFile();
        recognition.stop();
        isRecording = false;
    }

    // add event listeners for the new textarea
    var voiceCardContainer = event.target.closest(".voice-card-container");
    var textareaText = voiceCardContainer.querySelector(".text-input");

    var buttonMicrophoneOn = voiceCardContainer.querySelector(".fa-microphone");
    var buttonMicrophoneOff = voiceCardContainer.querySelector(".fa-microphone-slash");

    function recognitionStart() {
        recognition.start();
        isRecording = true;
    };

    if (!isRecording) {
        if (activeTextarea !== textareaText) {
          // stop recording on previously active textarea
          if (activeTextarea) {
            recognition.stop();
            isRecording = false;
          }

          // set new active textarea and start recording
          activeTextarea = textareaText;
          setTimeout(recognitionStart, 3500);

          buttonMicrophoneOn.style.display = 'none';
          buttonMicrophoneOff.style.display = 'inline';
        } else {
          // toggle recording
          if (recognition.recording) {
            recognition.stop();
            isRecording = false;

            buttonMicrophoneOn.style.display = 'inline';
            buttonMicrophoneOff.style.display = 'none';
          } else {
            recognition.stop();
            isRecording = false;

            setTimeout(recognitionStart, 3500);

            buttonMicrophoneOn.style.display = 'none';
            buttonMicrophoneOff.style.display = 'inline';
          }
        }
    } else {
      recognition.stop();
      isRecording = false;

      buttonMicrophoneOn.style.display = 'inline';
      buttonMicrophoneOff.style.display = 'none';
    }
  }
};
/// USING MICROPHONE TO RECOGNITION VOICE ///

/// CREATE DYNAMIC CHANGE AVATAR ///
function changeAvatarSelect(event) {
    if (event.target.classList.contains("model-checkbox-value")) {
        var checkbox = event.target.closest('.model-checkbox-value');
        var voiceCardContainer =  event.target.closest('.voice-card-container');
        var avatar = voiceCardContainer.querySelector('.img-avatar');
        // clear set interval to change avatars
        stopChangeAvatarSrc(avatar);

        let arr;

        if (checkbox.checked) {
          console.log('Checkbox is checked!');
          arr = JSON.parse(avatar.name);
          arr.push(checkbox.name);
          avatar.name = JSON.stringify(arr);
          avatar.src = checkbox.name;  // important keep to update in moment img

          fetch("/voice_status/")
            .then((response) => {
                if (!response.ok) throw response;
                return response.json();
            })
            .then((response) => {
                voice_dict = response;
                // Get the value from the dictionary using the key
                const value = voice_dict[checkbox.value];
                // Extract the values of checkpoint and waveglow
                const checkpoint = value.checkpoint;
                const waveglow = value.waveglow;
                // Show or not show message
                if (!checkpoint || !waveglow) {
                    avatarInfoPop(avatar, checkbox.value);
                }
                console.log(checkpoint, waveglow); // Output: true true
                })
            .catch((err) => {
                console.log(err);
            });
        } else {
          console.log('Checkbox is not checked!');
          // remove value from list
          arr = JSON.parse(avatar.name);
          arr = arr.filter((val) => val !== checkbox.name);
          avatar.name = JSON.stringify(arr);
        };

        function changeAvatarSrc(avatar) {
          arr = JSON.parse(avatar.name);
          let currentIndex = 0;

          const timerId = setInterval(() => {
            avatar.src = arr[currentIndex];

            currentIndex++;
            if (currentIndex === arr.length) {
              currentIndex = 0;
            }
          }, 2000);

          avatar.dataset.timerId = timerId;
        }

        if (arr.length == 0) {
            avatar.src = "media/avatar/Unknown.png";
        } else {
            // add set interval to change avatars
            changeAvatarSrc(avatar)
        }
    };
};

function avatarInfoPop(avatar, name) {
    var introAvatarStatus = introJs();
    introAvatarStatus.setOptions({
        steps: [
            {
                element: avatar,
                title: 'Сообщение',
                position: 'right',
                intro: `<div style="width: 200pt">
                            <p style="font-weight: 600;">Выбранный голос еще не загружен на устройство</p>
                            <p style="margin-top: 5pt;">Для синтеза необходима модель checkpoint и waveglow</p>
                            <p style="margin-top: 5pt;margin-bottom: 5pt;">При запуске синтеза речи, необходимые модели будут скачаны автоматически</p>
                            <p>Либо вы можете скачать самостоятельно модели из репозитория по <a href="https://wladradchenko.ru/static/wunjo.wladradchenko.ru/voice.json" target="_blank" rel="noopener noreferrer" >ссылке</a> и добавить модели в директорию <button style="background: none;border: none;color: blue;font-size: 12pt;cursor: pointer;" onclick="document.getElementById('a-link-open-folder').click();">.wunjo/voice/${name}</button></p>
                        </div>
                `,
            }
        ],
          showButtons: false,
          showStepNumbers: false,
          showBullets: false,
          nextLabel: 'Продолжить',
          prevLabel: 'Вернуться',
          doneLabel: 'Закрыть'
    });
    introAvatarStatus.start();
}
/// CREATE DYNAMIC CHANGE AVATAR ///

/// LISTEN CLICK ON ELEMENTS ///
function handleButtonClick(event) {
  if (event.target.classList.contains('voice-card-container-plus')) {
    addVoiceCard(event);
  } else if (event.target.classList.contains('voice-card-container-remove')) {
    removeVoiceCard(event);
  } else if (event.target.classList.contains('toggle-div-voice')) {
    changeVolume(event);
  } else if (event.target.classList.contains('microphone')) {
    microphoneRecognition(event);
  } else if (event.target.classList.contains('model-checkbox-value')) {
    changeAvatarSelect(event);
  }
  // console.log(event.target.classList);
}

voiceCards.addEventListener('click', handleButtonClick);
/// LISTEN CLICK ON ELEMENTS ///

/// CLEAR AVATAR SET INTERVAL BEFORE UPDATE NEW AVATAR ///
function stopChangeAvatarSrc(avatar) {
  const timerId = avatar.dataset.timerId;
  clearInterval(timerId);
  delete avatar.dataset.timerId;
}
/// CLEAR AVATAR SET INTERVAL BEFORE UPDATE NEW AVATAR ///

/// MULTI CHECKBOX SELECT OF VOICES ///
function multiVoiceSelect(element) {
  if (!element || !(element instanceof Element)) {
    return;
  }

  var checkboxes = element.querySelector(".model-checkboxes");
  checkboxes.style.display = "block";
  document.addEventListener("click", function(e) {
    if (!e.target.closest(".model-multiselect")) {
      checkboxes.style.display = "none";
    }
  });
}
/// MULTI CHECKBOX SELECT OF VOICES ///

// create a new SpeechRecognition object
const recognition = new window.webkitSpeechRecognition();

// set properties
recognition.continuous = true;
recognition.interimResults = true;

// set the recognition parameters
recognition.lang = "ru-RU";
recognition.interimResults = false;
recognition.maxAlternatives = 1;
recognition.onresult = handleRecognitionResult;
recognition.onerror = handleRecognitionError;
recognition.onend = handleRecognitionEnd;

// variable to keep track of currently active textarea
let activeTextarea;
let isRecording = false;

/// GET RECOGNITION RESULT ///
function handleRecognitionResult(event) {
  const lastResult = event.results[event.results.length - 1];
  const lastTranscript = lastResult[0].transcript;
  activeTextarea.value += lastTranscript;
}
/// GET RECOGNITION RESULT ///

/// GET RECOGNITION ERROR ///
function handleRecognitionError(event) {
  console.log(`Speech recognition error occurred: ${event.error}`);
}
/// GET RECOGNITION ERROR ///

/// UPDATE RECORD VOICE IF USER SILENCE ///
function handleRecognitionEnd() {
  // if still recording, restart recognition
  if (recognition.recording) {
    recognition.start();
  }
}
/// UPDATE RECORD VOICE IF USER SILENCE ///

/// SET AUDIO FILE FOR RECOGNITION ///
// add event listener to load audio button
let audioFile;

function recognitionAudioFile(elem) {
  var textareaText = elem.querySelector('.text-input');
  var file = elem.querySelector('.audio-file').files[0];

  if (!file) {
    console.error("No audio file selected");
    return;
  }

  activeTextarea = textareaText;
  isRecording = false;
  recognition.stop();

  const playBtn = elem.querySelector('.load-audio-button');
  const pauseBtn = elem.querySelector('.pause-audio-button');
  playBtn.style.display = "none";
  pauseBtn.style.display = "inline";

  const reader = new FileReader();
  reader.readAsDataURL(file);

  const audioEndedListener = function() {
    recognition.stop();
    isRecording = false;
    playBtn.style.display = "inline";
    pauseBtn.style.display = "none";
    audioFile.removeEventListener('ended', audioEndedListener);
  };

  reader.onload = function() {
    const audioUrl = reader.result;
    audioFile = new Audio(audioUrl);
    audioFile.addEventListener('loadedmetadata', () => {
      // set recognition duration limit to audio duration
      recognition.maxDuration = audioFile.duration;

      // transcribe audio
      recognition.start();
      isRecording = true;
      audioFile.play();
    });

    audioFile.addEventListener('ended', audioEndedListener);
  }
};
/// SET AUDIO FILE FOR RECOGNITION ///

/// PAUSE AUDIO FOR RECOGNITION ///
function pauseAllAudioFile() {
    const playBtnAll = document.querySelectorAll('.load-audio-button');
    const pauseBtnAll = document.querySelectorAll('.pause-audio-button');
    if (audioFile) {
        audioFile.pause();
        recognition.stop();
        isRecording = false;
    };
    playBtnAll.forEach(function(playBtn) {
        playBtn.style.display = "inline";
    });
    pauseBtnAll.forEach(function(pauseBtn) {
        pauseBtn.style.display = "none";
    });
};
/// PAUSE AUDIO FOR RECOGNITION ///

const buttonEnableAllButton = document.getElementById("button-enable-all");
const buttonDisableAllButton = document.getElementById("button-disable-all");

/// ENABLE ALL TOGGLE ///
buttonEnableAllButton.addEventListener("click", function () {
var toggleVoiceAll = document.querySelectorAll('.button.toggle-div-voice');
toggleVoiceAll.forEach(function(toggleVoice) {
  if (!toggleVoice.classList.contains('active')) {
    toggleVoice.classList.add('active');
    const toggleButtonOn = toggleVoice.querySelector('.toggle-button-voice-on');
    if (toggleButtonOn) {
      toggleButtonOn.style.display = 'inline';
    }
    const toggleButtonOff = toggleVoice.querySelector('.toggle-button-voice-off')
    if (toggleButtonOff) {
      toggleButtonOff.style.display = 'none';
    }
  }
});

buttonEnableAllButton.style.display = 'none';
buttonDisableAllButton.style.display = 'inline';
});
/// ENABLE ALL TOGGLE ///

/// DISABLE ALL TOGGLE ///
buttonDisableAllButton.addEventListener("click", function () {
var toggleVoiceAll = document.querySelectorAll('.button.toggle-div-voice');
toggleVoiceAll.forEach(function(toggleVoice) {
  if (toggleVoice.classList.contains('active')) {
    toggleVoice.classList.remove('active');
    const toggleButtonOn = toggleVoice.querySelector('.toggle-button-voice-on');
    if (toggleButtonOn) {
      toggleButtonOn.style.display = 'none';
    }
    const toggleButtonOff = toggleVoice.querySelector('.toggle-button-voice-off')
    if (toggleButtonOff) {
      toggleButtonOff.style.display = 'inline';
    }
  }
});

buttonEnableAllButton.style.display = 'inline';
buttonDisableAllButton.style.display = 'none';
});
/// DISABLE ALL TOGGLE ///

/// LOGICAL IF USER CHECKED ALL TOGGLE ///
function checkToggleVoiceAll() {
  var toggleVoiceAll = document.querySelectorAll('.button.toggle-div-voice');
  if (Array.from(toggleVoiceAll).every(toggleVoiceElem => toggleVoiceElem.classList.contains('active'))) {
    buttonEnableAllButton.style.display = 'none';
    buttonDisableAllButton.style.display = 'inline';
  } else if (Array.from(toggleVoiceAll).every(toggleVoiceElem => !toggleVoiceElem.classList.contains('active'))) {
    buttonEnableAllButton.style.display = 'inline';
    buttonDisableAllButton.style.display = 'none';
  } else {
    buttonEnableAllButton.style.display = 'inline';
    buttonDisableAllButton.style.display = 'none';
  }
};
/// LOGICAL IF USER CHECKED ALL TOGGLE ///

/// OPEN FOLDER ///
const link = document.getElementById('a-link-open-folder');
link.addEventListener('click', (event) => {
  event.preventDefault(); // prevent the link from following its href attribute
  fetch('/open_folder', { method: 'POST' })
    .then(response => {
      if (response.ok) {
        // handle the successful response from the server
      } else {
        throw new Error('Network response was not ok.');
      }
    })
    .catch(error => {
      console.error('There was a problem with the fetch operation:', error);
    });
});
/// OPEN FOLDER ///

/// EXTENSIONS ///
const extensions = document.getElementById('a-extensions');
extensions.addEventListener('click', (event) => {

  // Fetch the data from the URL
  fetch("/list_extensions/", {
    method: "GET"
  })
    .then(response => response.json())
    .then(data => {
      // Access the data and populate the select element
      const selectElement = document.getElementById("extensions-user-select");
      const descriptionElement = document.getElementById("extensions-user-select-description");

      // Iterate over the data and create an option for each entry
      for (const name in data) {
        const extension = data[name];
        const option = document.createElement("option");
        option.value = extension.url;
        option.text = name;
        option.dataset.key = name;  // Store the original key.
        selectElement.appendChild(option);
      }

      // Set initial description
      const firstOption = selectElement.querySelector("option");
      const selectedExtension = data[firstOption.text];
      descriptionElement.innerText = selectedExtension.description;

      // Add an event listener to the select element
      selectElement.addEventListener("change", async (event) => {
        const selectedOption = event.target.value;
        const selectedKey = event.target.options[event.target.selectedIndex].dataset.key;
        const selectedExtension = data[selectedKey];
        const translatedDescription = await translateWithGoogle(selectedExtension.description, 'auto', targetLang);
        descriptionElement.innerText = translatedDescription;
      });
    })
    .catch(error => {
      console.error("Not internet connection:", error);
    });

   var extensionsPanel = introJs();
   extensionsPanel.setOptions({
     steps: [
        {
            element: extensions,
            title: 'Расширения',
            position: 'right',
            intro: `<div style="width: 250pt;">
                    <div style="display: flex;flex-direction: column;">
                        <div style="margin-bottom:5pt;">
                          <input onclick="document.getElementById('extensions-url-field').style.display = this.checked ? 'block' : 'none';document.getElementById('extensions-select-field').style.display = this.checked ? 'none' : 'block';" type="checkbox" id="extensions-url-check-info" name="extensions-url-check">
                          <label for="extensions-url-check">Скачать по ссылке</label>
                        </div>
                        <div id="extensions-url-field" style="display:none;margin-top:5pt;">
                            <label for="extensions-url">Ссылка на скачивания расширения</label>
                            <input type="text" id="extensions-user-url" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                        </div>
                        <div id="extensions-select-field" style="display:block;margin-top:5pt;">
                            <label for="extensions-select">Выбор расширения для скачивания</label>
                            <select id="extensions-user-select" style="margin-left: 0;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                            </select>
                            <div id="extensions-user-select-description"></div>
                        </div>
                    </div>
                    <p style="margin-top: 10pt; margin-bottom: 10pt;">Угостите автора i@wladradchenko.ru <a href="https://wladradchenko.ru/donat" target="_blank" rel="noopener noreferrer">чашкой кофе</a> за труды и развития проекта</p>
                    <button class="introjs-button" style="margin-top: 20pt;right: 0;left: 0;display: flex;justify-content: center;width: 100%;padding-left: 0;padding-right: 0;" onclick="sendExtensionsKey(this.parentElement);">Загрузить</button>
                    <p style="margin-top: 15pt;font-size: 10pt;">Примечание: Расширения можно скачать самостоятельно в директорию .wunjo/extensions/{folder}. Подробнее о формате расширения по <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru" target="_blank" rel="noopener noreferrer">ссылке</a></p>
                </div>`,
        },
    ],
      showButtons: false,
      showStepNumbers: false,
      showBullets: false,
      nextLabel: 'Продолжить',
      prevLabel: 'Вернуться',
      doneLabel: 'Закрыть'
   });
   extensionsPanel.start();
});


function currentProcessor(elem = undefined) {
  fetch('/current_processor', { method: 'GET' })
    .then(response => response.json())
    .then(data => {
      var deviceStatus = data.current_processor;
      var deviceUpgrade = data.upgrade_gpu;

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
currentProcessor();

function sendExtensionsKey(elem) {
    const extensionsCheckbox = document.getElementById('extensions-url-check-info');
    const extensionsUrlInput = document.getElementById('extensions-user-url');
    const extensionsUrlSelect = document.getElementById('extensions-user-select');
    let extensionSend;

    if (extensionsCheckbox.checked) {
      const extensionUrlValue = extensionsUrlInput.value;
      extensionSend = { "extension_name": null, "extension_url": extensionUrlValue };
    } else {
      const selectedOption = extensionsUrlSelect.options[extensionsUrlSelect.selectedIndex];
      const extensionName = selectedOption.text;
      const extensionUrlValue = selectedOption.value;
      extensionSend = { "extension_name": extensionName, "extension_url": extensionUrlValue };
    }
    console.log(extensionSend)

    fetch("/get_extensions/", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(extensionSend)
    })

    const closeIntroButton = document.querySelector('.introjs-skipbutton');
    closeIntroButton.click();
};
/// EXTENSIONS ///


///SUPPORT USER///
if (document.cookie.indexOf('introCompleted=true') !== -1) {
   // Not show intro
} else {
    var intro = introJs();
    intro.setOptions({
        steps: [
            {
                element: '.text-input',
                intro: 'Поле для ввода текста',
            },
            {
                element: '.a-button.microphone',
                intro: 'Переключатель микрофона для ввода текста голосом',
            },
            {
                element: '.audio-load',
                intro: 'Загрузить аудио файл',
            },
            {
                element: '.load-audio-button',
                intro: 'Распознать текст из аудио файла. Распознавание закончится после того, как аудио файл будет озвучен полностью',
            },
            {
                element: '.model-over-select',
                intro: 'Набор голосов для озвучивания',
            },
            {
                element: '.voice-card-container-plus',
                intro: 'Вы можете создать диалог добавив новое поле',
            },
            {
                element: '.voice-card-container-remove',
                intro: 'Удалите ненужные поля для диалогов',
            },
            {
                element: '.button.toggle-div-voice',
                intro: 'Если вы хотите озвучить выбранное поле текста необходимо активировать флаг',
            },
            {
                element: '#button-enable-all',
                intro: 'Вы также можете активировать все поля текста для озвучки или убрать флаг со всех полей',
            },
            {
                element: '#button-run-synthesis',
                intro: 'Отправить текст нейронной сети для озвучивания. Это займет некоторое время',
            },
            {
                element: '#button-show-voice-window',
                intro: 'Кнопка переключения между результатом синтеза речи и видео',
            },
            {
                element: '#button-run-deepfake-synthesis',
                intro: 'Окно создания анимации. Вы можете загрузить как входной файл изображение, видео или гиф анимацию. <br><br><text style="color:red;">Внимание! При первом запуске скачается 5 Гб данных - этой займет некоторое время.</text>',
            },
            {
                element: '#a-link-open-folder',
                intro: 'Ненужные файлы можно удалить из директории',
            },
            {
                element: '#a-extensions',
                intro: 'Вы можете скачать доступные расширения',
            },
            {
                element: '#a-link-open-author',
                intro: 'Узнать о других проектах автора вы можете на сайте <a target="_blank" rel="noopener noreferrer" href="https://wladradchenko.ru">wladradchenko.ru</a>. Приятного пользования!',
            }
        ],
          showButtons: true,
          showStepNumbers: false,
          nextLabel: 'Продолжить',
          prevLabel: 'Вернуться',
          doneLabel: 'Закрыть'
    });
    intro.start();
}
document.cookie = "introCompleted=true; expires=Fri, 31 Dec 9999 23:59:59 GMT";
///SUPPORT USER///