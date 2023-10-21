var voiceCards = document.querySelector("#voice-cards");
const voiceCardContainer = document.querySelector(".voice-card-container");
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
  if (event.target.classList.contains("voice-card-container-plus")) {
    var voiceCardContainers = document.querySelectorAll(
      ".voice-card-container"
    );
    var voiceCardContainer = event.target.closest(".voice-card-container");
    var currentVoiceMultiSelect =
      voiceCardContainer.querySelector(".model-checkboxes");
    currentVoiceMultiSelect.style.display = "none";

    // Copy element to append
    // var newVoiceCardContainer = voiceCardContainer.cloneNode(true);
    var newVoiceCardContainer = voiceCardTemplate.content.cloneNode(true);

    // Determine the index of the new container in the list of containers
    var newIndex =
      Array.from(voiceCardContainers).indexOf(voiceCardContainer) + 1;

    // Set the order of the avatar and textarea based on the index
    var swapped = newIndex % 2 !== 0;
    var voiceCardAvatar =
      newVoiceCardContainer.querySelector(".voice-card-avatar");
    var voiceCardTextarea = newVoiceCardContainer.querySelector(
      ".voice-card-textarea"
    );
    voiceCardAvatar.style.order = swapped ? 2 : 1;
    voiceCardTextarea.style.order = swapped ? 1 : 2;

    // Set the bubble position based on the index
    var bubble = newVoiceCardContainer.querySelector(".bubble");
    bubble.classList.toggle("bubble-bottom-right", !swapped);
    bubble.classList.toggle("bubble-bottom-left", swapped);
    if (swapped) {
      bubble.style.left = "auto";
      bubble.style.right = "-70%";
    } else {
      bubble.style.right = "auto";
      bubble.style.left = "-20%";
    }

    // Insert the new container at the correct index
    voiceCards.insertBefore(
      newVoiceCardContainer,
      voiceCardContainers[newIndex]
    );
  }
}
/// APPEND NEW AVATAR CARD FROM TEMPLATE ///

/// REMOVE AVATAR CARD ///
function removeVoiceCard(event) {
  if (event.target.classList.contains("voice-card-container-remove")) {
    var voiceCardContainers = document.querySelectorAll(
      ".voice-card-container"
    );
    if (voiceCardContainers.length > 1) {
      var voiceCardContainer = event.target.closest(".voice-card-container");
      voiceCards.removeChild(voiceCardContainer);
    }

    // Remove recognition voice
    recognition.stop();
  }
}
/// REMOVE AVATAR CARD ///

/// CHANGE VOLUME TOGGLE ///
function changeVolume(event) {
  if (event.target.classList.contains("toggle-div-voice")) {
    var toggle = event.target.closest(".button");
    toggle.classList.toggle("active");
    var toggleIconVoiceOn = toggle.querySelector(".toggle-button-voice-on");
    var toggleIconVoiceOff = toggle.querySelector(".toggle-button-voice-off");
    var isOn = toggle.classList.contains("active");
    if (isOn) {
      toggleIconVoiceOn.style.display = "inline";
      toggleIconVoiceOff.style.display = "none";
      // add code to turn on text-to-voice
    } else {
      toggleIconVoiceOn.style.display = "none";
      toggleIconVoiceOff.style.display = "inline";
      // add code to turn off text-to-voice
    }

    // check disable and enable all buttons
    checkToggleVoiceAll();
  }
}
/// CHANGE VOLUME TOGGLE ///

/// CREATE DYNAMIC CHANGE AVATAR ///
function changeAvatarSelect(event) {
  if (event.target.classList.contains("model-checkbox-value")) {
    var checkbox = event.target.closest(".model-checkbox-value");
    var voiceCardContainer = event.target.closest(".voice-card-container");
    var avatar = voiceCardContainer.querySelector(".img-avatar");
    // clear set interval to change avatars
    stopChangeAvatarSrc(avatar);

    let arr;

    if (checkbox.checked) {
      console.log("Checkbox is checked!");
      arr = JSON.parse(avatar.name);
      arr.push(checkbox.name);
      avatar.name = JSON.stringify(arr);
      avatar.src = checkbox.name; // important keep to update in moment img

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
          // console.log(checkpoint, waveglow); // Output: true true
        })
        .catch((err) => {
          console.log(err);
        });
    } else {
      console.log("Checkbox is not checked!");
      // remove value from list
      arr = JSON.parse(avatar.name);
      arr = arr.filter((val) => val !== checkbox.name);
      avatar.name = JSON.stringify(arr);
    }

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
      changeAvatarSrc(avatar);
    }
  }
}

function avatarInfoPop(avatar, name) {
  var introAvatarStatus = introJs();
  introAvatarStatus.setOptions({
    steps: [
      {
        element: avatar,
        title: "Сообщение",
        position: "right",
        intro: `<div style="width: 200pt">
                            <p style="font-weight: 600;">Выбранный голос еще не загружен на устройство</p>
                            <p style="margin-top: 5pt;">Для синтеза необходима модель checkpoint и waveglow</p>
                            <p style="margin-top: 5pt;">При запуске синтеза речи, необходимые модели будут скачаны автоматически</p>
                            <p style="margin-top: 5pt;margin-bottom: 5pt;">Либо вы можете скачать самостоятельно модели из репозитория по <a style="text-transform: lowercase;" href="https://wladradchenko.ru/static/wunjo.wladradchenko.ru/voice_multi.json" target="_blank" rel="noopener noreferrer" >ссылке</a><text style="text-transform: lowercase;"> и добавить модели в директорию </text><button class="notranslate" style="background: none;border: none;color: blue;font-size: 12pt;cursor: pointer;" onclick="document.getElementById('a-link-open-folder').click();">.wunjo/voice/name/</button></p>
                            <p><b>Как установить модели ручным способом</b><text style="text-transform: lowercase;"> для синтеза речи, вы найдете в </text><a style="text-transform: lowercase;" href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki" target="_blank" rel="noopener noreferrer">документации проекта</a></p>
                        </div>
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
  introAvatarStatus.start();
  translateReload(targetLang); // Idea how translate open elements!
}
/// CREATE DYNAMIC CHANGE AVATAR ///

/// LISTEN CLICK ON ELEMENTS ///
function handleButtonClick(event) {
  if (event.target.classList.contains("voice-card-container-plus")) {
    addVoiceCard(event);
  } else if (event.target.classList.contains("voice-card-container-remove")) {
    removeVoiceCard(event);
  } else if (event.target.classList.contains("toggle-div-voice")) {
    changeVolume(event);
  } else if (event.target.classList.contains("model-checkbox-value")) {
    changeAvatarSelect(event);
  }
  // console.log(event.target.classList);
}

voiceCards.addEventListener("click", handleButtonClick);
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
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".model-multiselect")) {
      checkboxes.style.display = "none";
    }
  });
}
/// MULTI CHECKBOX SELECT OF VOICES ///

// create a new SpeechRecognition object
const recognition = new window.webkitSpeechRecognition();

// Set properties
recognition.continuous = true;
recognition.interimResults = true;

// Set the recognition parameters
recognition.lang = toDialectCode(targetLang); // init by lang

// SETTINGS FOR TEXT RECOGNITION AND TRANSLATION OF TTS //
function toDialectCode(langCode = "en") {
  const langToDialect = {
    af: "af-ZA",
    sq: "sq-AL",
    am: "am-ET",
    ar: "ar-SA",
    hy: "hy-AM",
    az: "az-AZ",
    eu: "eu-ES",
    be: "be-BY",
    bn: "bn-BD",
    bs: "bs-BA",
    bg: "bg-BG",
    ca: "ca-ES",
    ceb: "ceb-PH",
    ny: "ny-MW",
    zh: "zh-CN",
    co: "co-FR",
    hr: "hr-HR",
    cs: "cs-CZ",
    da: "da-DK",
    nl: "nl-NL",
    en: "en-US",
    eo: "eo-EO",
    et: "et-EE",
    tl: "tl-PH",
    fi: "fi-FI",
    fr: "fr-FR",
    fy: "fy-NL",
    gl: "gl-ES",
    ka: "ka-GE",
    de: "de-DE",
    el: "el-GR",
    gu: "gu-IN",
    ht: "ht-HT",
    ha: "ha-NG",
    haw: "haw-US",
    iw: "iw-IL",
    hi: "hi-IN",
    hmn: "hmn-CN",
    hu: "hu-HU",
    is: "is-IS",
    ig: "ig-NG",
    id: "id-ID",
    ga: "ga-IE",
    it: "it-IT",
    ja: "ja-JP",
    jw: "jw-ID",
    kn: "kn-IN",
    kk: "kk-KZ",
    km: "km-KH",
    ko: "ko-KR",
    ku: "ku-IQ",
    ky: "ky-KG",
    lo: "lo-LA",
    lv: "lv-LV",
    lt: "lt-LT",
    lb: "lb-LU",
    mk: "mk-MK",
    mg: "mg-MG",
    ms: "ms-MY",
    ml: "ml-IN",
    mt: "mt-MT",
    mi: "mi-NZ",
    mr: "mr-IN",
    mn: "mn-MN",
    my: "my-MM",
    ne: "ne-NP",
    no: "no-NO",
    or: "or-IN",
    ps: "ps-AF",
    fa: "fa-IR",
    pl: "pl-PL",
    pt: "pt-PT",
    pa: "pa-PK",
    ro: "ro-RO",
    ru: "ru-RU",
    sm: "sm-WS",
    gd: "gd-GB",
    sr: "sr-RS",
    st: "st-ZA",
    sn: "sn-ZW",
    sd: "sd-PK",
    si: "si-LK",
    sk: "sk-SK",
    sl: "sl-SI",
    so: "so-SO",
    es: "es-ES",
    su: "su-ID",
    sw: "sw-TZ",
    sv: "sv-SE",
    tg: "tg-TJ",
    ta: "ta-IN",
    te: "te-IN",
    th: "th-TH",
    tr: "tr-TR",
    uk: "uk-UA",
    ur: "ur-PK",
    uz: "uz-UZ",
    vi: "vi-VN",
    cy: "cy-GB",
    xh: "xh-ZA",
    yi: "yi-YI",
    zu: "zu-ZA",
  };

  return langToDialect[langCode] || langCode;
}

function audioDragAndDropSTT(event, elem) {
  const file = URL.createObjectURL(event.target.files[0]);
  const audio = document.getElementById("audioSTTSrc");

  // Update the audio src
  audio.src = file;

  // Set attributes for voice on setting button to send after in Synthesis
  elem.setAttribute("blob-audio-src", file);

  // Show the play button if hidden
  const playBtn = document.getElementById("audioSTTPlay");
  // Show the recognition button and message
  document.getElementById("cloneVoiceMessage").style.display = "inline";
  document.getElementById("recognitionSTTAudio").style.display = "inline";

  playBtn.style.display = "inline";

  audio.pause();
  playBtn.children[0].style.display = "inline";
  playBtn.children[1].style.display = "none";
}

async function submittedSTT(elem, activeTextarea) {
  const audioElement = elem.querySelector("#audioSTTSrc");
  const audioSrc = audioElement.src;

  if (!audioSrc) {
    console.error("No audio file selected");
    return;
  }

  elem.querySelector("#recognitionSTTAudio").innerText =
    await translateWithGoogle(
      "Распознавание... Не выключайте",
      "auto",
      targetLang
    );

  const stream = audioElement.captureStream();
  // const recognition = new window.webkitSpeechRecognition();

  recognition.onresult = (event) => {
    const result = event.results[0][0].transcript;
    activeTextarea.value = result;
  };

  recognition.onerror = (event) => {
    console.error(event.error);
  };

  recognition.onend = () => {
    console.log("Recognition ended");
  };

  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.start();

  const audioTrack = stream.getAudioTracks()[0];
  audioElement.play();

  // Stop recognition when the audio ends
  audioElement.addEventListener("ended", () => {
    recognition.stop();
    audioTrack.stop();
    const closeIntroButton = document.querySelector(".introjs-skipbutton");
    closeIntroButton.click();
  });
}

function settingTextToSpeech(elem, languages) {
  var sectionTextTTS =
    elem.parentElement.parentElement.parentElement.parentElement;
  var textareaTTS = sectionTextTTS.querySelector(".text-input");
  var curLang = elem.getAttribute("value-translate");

  var introSettingTextToSpeech = introJs();
  introSettingTextToSpeech.setOptions({
    steps: [
      {
        element: elem,
        title: "Настройки",
        position: "right",
        intro: `<div style="min-height: 180pt;">
                    <div style="display: flex;flex-direction: row;">
                        <div style="width: 250pt;margin: 10pt;">
                            <div id="setting-tts-lang-select-field" style="display:block;margin-top:5pt;">
                                <label for="setting-tts-lang-select">Выбрать язык</label>
                                <select id="setting-tts-user-lang-select" style="margin-left: 0;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                                </select>
                            </div>
                            <div>
                                <div class="uploadSTTAudio" style="margin-top: 10pt;margin-bottom: 10pt;display: flex;">
                                    <button class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onChange="audioRecordVoiceSTT(event)"  id="recordSTTAudio">Записать голос</button>
                                    <label id="uploadSTTAudioLabel" for="uploadSTTAudio" class="introjs-button" style="margin-left: 5pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Загрузить аудио</label>
                                    <input style="width: 0;" accept="audio/*" type="file" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadSTTAudio"  />
                                    <div id="previewSTTAudio">
                                      <button id="audioSTTPlay" class="introjs-button" style="display:none;margin-left: 5pt;">
                                        <i class="fa fa-play"></i>
                                        <i style="display: none;" class="fa fa-pause"></i>
                                      </button>
                                      <audio id="audioSTTSrc" style="display:none;" controls preload="none">
                                        Your browser does not support audio.
                                      </audio>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <button class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important; display: none;" id="recognitionSTTAudio">Распознать</button>
                            </div>
                            <i style="margin-top: 5pt;margin-bottom: 15pt;font-size: 10pt;display: flow;">
                                <b>Примечание:</b> Как добавить любой язык в приложение?
                                <a style="color: blue;" onclick="document.querySelector('.setting-tts-translation-info').style.display = (document.querySelector('.setting-tts-translation-info').style.display === 'block' ? 'none' : 'block');">Открыть инструкцию.</a>
                            </i>
                            <i class="setting-tts-translation-info" style="margin-top: 0pt;margin-bottom: 15pt;font-size: 10pt;display: none;padding:10pt;background-color: rgb(235 240 243 / 0%);border-color: rgb(230, 231, 238);box-shadow: rgb(184, 185, 190) 2px 2px 5px inset, rgb(255, 255, 255) -3px -3px 7px inset;">
                                Перейдите в <b class="notranslate">.wunjo/settings/settings.json</b>.
                                Добавьте желаемый язык в формате: <b class="notranslate">"default_language": {"name": "code"}</b>.
                                Чтобы найти соответствующий код для вашего языка, обратитесь к языковым кодам <a class="notranslate" target="_blank" rel="noopener noreferrer" href="https://cloud.google.com/translate/docs/languages">Google Cloud Translate Language Codes</a>.
                            </i>
                        </div>
                        <div style="width: 250pt;margin: 10pt;">
                            <div style="display: none;" id="cloneVoiceMessage">
                                <div style="margin-bottom:5pt;margin-top: 15pt;">
                                  <input onclick="" type="checkbox" id="setting-rtvc-check-info" name="setting-rtvc-check">
                                  <label for="setting-rtvc-check">Клонировать голос</label>
                                </div>
                                <i style="margin-top: 5pt;font-size: 10pt;margin-bottom: 15pt;"><b>Примечание:</b> При клонировании голоса, аудио будет синтезировано на основе текста и прикрепленного или записанного аудио. В этом случая можно не выбирать голос для озвучки. Поддерживается английский, русский и китайский.</i>
                                <hr style="margin-top: 15pt;">
                            </div>
                            <div style="margin-bottom:5pt;margin-top: 15pt">
                              <input onclick="" type="checkbox" id="setting-tts-translation-check-info" name="setting-tts-translation-check">
                              <label for="setting-tts-translation-check">Автоматический перевод</label>
                            </div>
                            <i style="margin-top: 5pt;font-size: 10pt;"><b>Примечание:</b> Автоматический перевод означает, что текст будет озвучен на выбранном языке с клонированием голоса модели на любой язык, даже если текст введен на другом языке или нейронная сеть имеет голос на другом языке, перевод будет совершен на выбранный язык.</i>
                        </div>
                    </div>
                </div>`,
      },
    ],
    showButtons: false,
    showStepNumbers: false,
    showBullets: false,
    nextLabel: "Продолжить",
    prevLabel: "Вернуться",
    doneLabel: "Закрыть",
  });
  introSettingTextToSpeech.start();

  // Get checkbox voice clone
  const checkboxCloneAudioVoice = document.getElementById(
    "setting-rtvc-check-info"
  );
  const attrValueCloneAudioVoice = elem.getAttribute("voice-audio-clone");
  checkboxCloneAudioVoice.checked = attrValueCloneAudioVoice === "true";
  checkboxCloneAudioVoice.addEventListener("change", function () {
    if (this.checked) {
      elem.setAttribute("voice-audio-clone", true);
    } else {
      elem.setAttribute("voice-audio-clone", false);
    }
  });

  // Get checkbox auto translate
  const checkboxAutomationTranslate = document.getElementById(
    "setting-tts-translation-check-info"
  );
  const attrValueAutomationTranslate = elem.getAttribute("automatic-translate");
  checkboxAutomationTranslate.checked = attrValueAutomationTranslate === "true";
  checkboxAutomationTranslate.addEventListener("change", function () {
    if (this.checked) {
      elem.setAttribute("automatic-translate", true);
    } else {
      elem.setAttribute("automatic-translate", false);
    }
  });

  // Set audio upload onChange
  const uploadSTTAudio = document.getElementById("uploadSTTAudio");
  uploadSTTAudio.addEventListener("change", function () {
    audioDragAndDropSTT(event, elem);
  });

  // Set recognition button
  const recognitionSTTAudio = document.getElementById("recognitionSTTAudio");

  recognitionSTTAudio.addEventListener("click", function () {
    submittedSTT(this.parentElement.parentElement, textareaTTS);
  });

  // Get the select element
  const selectElementLanguageTTS = document.getElementById(
    "setting-tts-user-lang-select"
  );

  // Populate the select element with options
  for (const [key, value] of Object.entries(languages)) {
    const option = document.createElement("option");
    option.text = key;
    option.value = value;
    if (curLang === value) {
      option.selected = true;
    }
    option.classList.add("notranslate");
    selectElementLanguageTTS.add(option);
  }

  // Add event listener for the change event
  selectElementLanguageTTS.addEventListener("change", function () {
    const selectedValueLanguageTTS = this.value; // Get the value of the selected option
    const selectedTextLanguageTTS = this.options[this.selectedIndex].text; // Get the text of the selected option
    // this value will use for automatic translate to send in backend
    elem.setAttribute("value-translate", selectedValueLanguageTTS);

    // Do something with the selected value or text
    recognition.lang = toDialectCode(selectedValueLanguageTTS);

    // Support languages for voice cloning
    if (["en", "ru", "zh"].includes(selectedValueLanguageTTS)) {
      // If the selected value is one of "en", "ru", or "zh", enable the checkbox
      checkboxCloneAudioVoice.disabled = false;
      checkboxAutomationTranslate.disabled = false;
    } else {
      // If the selected value is not one of "en", "ru", or "zh", uncheck and disable the checkbox
      checkboxCloneAudioVoice.checked = false;
      checkboxCloneAudioVoice.disabled = true;
      checkboxAutomationTranslate.checked = false;
      checkboxAutomationTranslate.disabled = true;
    }
  });

  // record voice button
  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;
  let audioStream; // To keep track of the audio stream

  const audio = document.getElementById("audioSTTSrc");
  // Show the play button if hidden
  const playBtn = document.getElementById("audioSTTPlay");

  // Existing play/pause logic
  playBtn.addEventListener("click", function () {
    if (audio.paused) {
      audio.play();
      playBtn.children[0].style.display = "none";
      playBtn.children[1].style.display = "inline";
    } else {
      audio.pause();
      playBtn.children[0].style.display = "inline";
      playBtn.children[1].style.display = "none";
    }
  });

  // Existing audio ended logic
  audio.addEventListener("ended", function () {
    playBtn.children[0].style.display = "inline";
    playBtn.children[1].style.display = "none";
  });

  document
    .getElementById("recordSTTAudio")
    .addEventListener("click", async function () {
      if (!isRecording) {
        isRecording = true;
        this.textContent = await translateWithGoogle(
          "Стоп",
          "auto",
          targetLang
        );

        // Initialize MediaRecorder
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        audioStream = stream; // Store the stream for later use
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(audioBlob);
          document.getElementById("audioSTTSrc").src = audioUrl;

          // Set attributes for voice on setting button to send after in Synthesis
          elem.setAttribute("blob-audio-src", audioUrl);

          // Show the play button
          document.getElementById("audioSTTPlay").style.display = "inline";

          // Show the recognition button and message
          document.getElementById("cloneVoiceMessage").style.display = "inline";
          recognitionSTTAudio.style.display = "inline";

          // Clear old audio chunks
          audioChunks = [];
        };

        mediaRecorder.start();
      } else {
        isRecording = false;
        this.textContent = await translateWithGoogle(
          "Записать голос",
          "auto",
          targetLang
        );

        // Stop the MediaRecorder and close the stream
        mediaRecorder.stop();

        // Reset play button if playing
        audio.pause();
        playBtn.children[0].style.display = "inline";
        playBtn.children[1].style.display = "none";

        // Close the audio stream to completely stop recording
        audioStream.getTracks().forEach((track) => track.stop());
      }
    });

  // If blob already is, when show play button?
  const voiceCloneBlobUrl = elem.getAttribute("blob-audio-src");
  if (voiceCloneBlobUrl !== "") {
    // Show the play button
    document.getElementById("audioSTTPlay").style.display = "inline";

    // Show the recognition button and message
    document.getElementById("cloneVoiceMessage").style.display = "inline";
    recognitionSTTAudio.style.display = "inline";

    // Set prev blob
    document.getElementById("audioSTTSrc").src = voiceCloneBlobUrl;
  }
}
// SETTINGS FOR TEXT RECOGNITION AND TRANSLATION OF TTS //

const buttonEnableAllButton = document.getElementById("button-enable-all");
const buttonDisableAllButton = document.getElementById("button-disable-all");

/// ENABLE ALL TOGGLE ///
buttonEnableAllButton.addEventListener("click", function () {
  var toggleVoiceAll = document.querySelectorAll(".button.toggle-div-voice");
  toggleVoiceAll.forEach(function (toggleVoice) {
    if (!toggleVoice.classList.contains("active")) {
      toggleVoice.classList.add("active");
      const toggleButtonOn = toggleVoice.querySelector(
        ".toggle-button-voice-on"
      );
      if (toggleButtonOn) {
        toggleButtonOn.style.display = "inline";
      }
      const toggleButtonOff = toggleVoice.querySelector(
        ".toggle-button-voice-off"
      );
      if (toggleButtonOff) {
        toggleButtonOff.style.display = "none";
      }
    }
  });

  buttonEnableAllButton.style.display = "none";
  buttonDisableAllButton.style.display = "inline";
});
/// ENABLE ALL TOGGLE ///

/// DISABLE ALL TOGGLE ///
buttonDisableAllButton.addEventListener("click", function () {
  var toggleVoiceAll = document.querySelectorAll(".button.toggle-div-voice");
  toggleVoiceAll.forEach(function (toggleVoice) {
    if (toggleVoice.classList.contains("active")) {
      toggleVoice.classList.remove("active");
      const toggleButtonOn = toggleVoice.querySelector(
        ".toggle-button-voice-on"
      );
      if (toggleButtonOn) {
        toggleButtonOn.style.display = "none";
      }
      const toggleButtonOff = toggleVoice.querySelector(
        ".toggle-button-voice-off"
      );
      if (toggleButtonOff) {
        toggleButtonOff.style.display = "inline";
      }
    }
  });

  buttonEnableAllButton.style.display = "inline";
  buttonDisableAllButton.style.display = "none";
});
/// DISABLE ALL TOGGLE ///

/// LOGICAL IF USER CHECKED ALL TOGGLE ///
function checkToggleVoiceAll() {
  var toggleVoiceAll = document.querySelectorAll(".button.toggle-div-voice");
  if (
    Array.from(toggleVoiceAll).every((toggleVoiceElem) =>
      toggleVoiceElem.classList.contains("active")
    )
  ) {
    buttonEnableAllButton.style.display = "none";
    buttonDisableAllButton.style.display = "inline";
  } else if (
    Array.from(toggleVoiceAll).every(
      (toggleVoiceElem) => !toggleVoiceElem.classList.contains("active")
    )
  ) {
    buttonEnableAllButton.style.display = "inline";
    buttonDisableAllButton.style.display = "none";
  } else {
    buttonEnableAllButton.style.display = "inline";
    buttonDisableAllButton.style.display = "none";
  }
}
/// LOGICAL IF USER CHECKED ALL TOGGLE ///

/// OPEN FOLDER ///
const link = document.getElementById("a-link-open-folder");
link.addEventListener("click", (event) => {
  event.preventDefault(); // prevent the link from following its href attribute
  fetch("/open_folder", { method: "POST" })
    .then((response) => {
      if (response.ok) {
        // handle the successful response from the server
      } else {
        throw new Error("Network response was not ok.");
      }
    })
    .catch((error) => {
      console.error("There was a problem with the fetch operation:", error);
    });
});
/// OPEN FOLDER ///

/// CURRENT PROCESSOR ///
const processor = document.getElementById("a-change-processor");
processor.addEventListener("click", (event) => {
  event.preventDefault(); // prevent the link from following its href attribute
  fetch("/change_processor", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      currentProcessor();
    })
    .catch((error) => {
      console.log(error);
    });
});

function availableFeaturesByCUDA(elem = undefined) {
  // inspect what can be use torch cuda is available
  fetch("/current_processor", { method: "GET" })
    .then((response) => response.json())
    .then((data) => {
      var deviceStatus = data.current_processor;
      var deviceUpgrade = data.upgrade_gpu;
      if (elem && deviceStatus == "cuda") {
        elem.style.display = "block";
      } else if (elem) {
        elem.style.display = "none";
      }
    })
    .catch((error) => {
      console.log(error);
    });
}

function currentProcessor() {
  fetch("/current_processor", { method: "GET" })
    .then((response) => response.json())
    .then((data) => {
      var deviceStatus = data.current_processor;
      var deviceUpgrade = data.upgrade_gpu;
      if (deviceStatus == "cuda") {
        processor.style.color = "green";
      } else {
        processor.style.color = "red";
      }
    })
    .catch((error) => {
      console.log(error);
    });
}

currentProcessor();
/// CURRENT PROCESSOR ///

///SUPPORT USER///
if (document.cookie.indexOf("introCompleted=true") !== -1) {
  // Not show intro
} else {
  var introWelcome = introJs();
  introWelcome.setOptions({
    steps: [
      {
        element: "#a-link-open-tutorial",
        position: "right",
        title: "Welcome to Wunjo AI!",
        intro: `
                <div style="width: 300pt;padding: 5pt;">
                    <p style="margin-bottom: 10pt;"><b>Explore the capabilities of neural networks:</b></p>
                    <ul style="display: flex;flex-direction: column;margin-bottom: 10pt;">
                       <text>🎥 Animating faces, lips, emotions and craft deepfakes by face swap using one photo.</text>
                        <text>✨ Enhance videos with AI retouch tool.</text>
                        <text>🗣️ Convert text into lifelike speech.</text>
                        <text>🎙️ Clone voices in real-time or from audio.</text>
                        <text>🌍 Multi-language support for voice cloning and synthesis.</text>
                        <text>📝 Speech-to-text transcription.</text>
                        <text>🎭 Create dynamic dialogues with distinct voices.</text>
                    </ul>
                    <p style="margin-bottom: 10pt;">From voiceovers to game character voicing, from animate face or lips to entertaining deepfakes by face swap, Wunjo AI empowers creativity, securely and freely on your device.</p>
                    <p style="margin-bottom: 10pt;"><b>If you wanna start tour about</b> Wunjo AI, you can tap on this button with interrogative point.</p>
                    <label style="margin-top: 15pt;"><b>Select Your Language:</b></label>
                    <select id="languageWelcomeGuide" style="margin-left: 0;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                    </select>
                </div>
                `,
      },
    ],
    showButtons: false,
    showStepNumbers: false,
    nextLabel: "Next",
    prevLabel: "Back",
    doneLabel: "Close",
  });
  introWelcome.start();
  // Get the select element
  const selectWelcomeHelp = document.getElementById("languageWelcomeGuide");
  const languageGeneral = document.getElementById("language-dropdown");

  // Populate the select element with options
  for (const [key, value] of Object.entries(listExistingLang)) {
    const option = document.createElement("option");
    option.text = key;
    option.value = value;
    if (targetLang === value) {
      option.selected = true;
    }
    option.setAttribute("name", key);
    option.classList.add("notranslate");
    selectWelcomeHelp.add(option);
  }

  // Add event listener for the change event
  selectWelcomeHelp.addEventListener("change", function () {
    const selectValueWelcomeHelp = this.value; // Get the value of the selected option
    let selectedOption = this.options[this.selectedIndex];
    let selectedLangName = selectedOption.getAttribute("name");

    const option = Array.from(languageGeneral.options).find(
      (opt) => opt.value === selectValueWelcomeHelp
    );
    if (option) {
      languageGeneral.selectedIndex = option.index;
    }

    translateNewUserLang(selectValueWelcomeHelp);
    updateLangSetting(selectValueWelcomeHelp, selectedLangName);
  });
}
document.cookie = "introCompleted=true; expires=Fri, 31 Dec 9999 23:59:59 GMT";

async function supportUser() {
  // Main
  const btnNext = await translateWithGoogle("Next", "auto", targetLang);
  const btnBack = await translateWithGoogle("Back", "auto", targetLang);
  const btnClose = await translateWithGoogle("Close", "auto", targetLang);
  // Page 1
  const welcomeSupportOne = await translateWithGoogle(
    "Welcome to the Wunjo AI guide.",
    "auto",
    targetLang
  );
  const welcomeSupportTwo = await translateWithGoogle(
    "For detailed tutorials, check out on",
    "auto",
    targetLang
  );
  const welcomeSupportThree = await translateWithGoogle(
    "For documentation, visit",
    "auto",
    targetLang
  );
  const welcomeSupportFour = await translateWithGoogle(
    "If you encounter any issues or have suggestions, please report them on",
    "auto",
    targetLang
  );
  // Page 2
  const translateApplicationBtn = await translateWithGoogle(
    "Application language selection.",
    "auto",
    targetLang
  );
  // Page 3
  const textInputSupport = await translateWithGoogle(
    "Enter the text for speech synthesis here.",
    "auto",
    targetLang
  );
  // Page 4
  const aButtonSettingTtsSupportOne = await translateWithGoogle(
    "Speech Synthesis Settings.",
    "auto",
    targetLang
  );
  const aButtonSettingTtsSupportTwo = await translateWithGoogle(
    "Clone your voice in real-time or from audio, recognize text from voice and audio, and automatically translate synthesized speech.",
    "auto",
    targetLang
  );
  // Page 5
  const modalOverSelectSupportOne = await translateWithGoogle(
    "Voice Sets for Different Languages.",
    "auto",
    targetLang
  );
  const modalOverSelectSupportTwo = await translateWithGoogle(
    "Use auto-translation to employ chosen models in languages other than their default. ",
    "auto",
    targetLang
  );
  const modalOverSelectSupportThree = await translateWithGoogle(
    "Note: Initial launch will prompt a download for voice synthesis models. This may take a while.",
    "auto",
    targetLang
  );
  // Page 6
  const voiceCardContainerPlusSupport = await translateWithGoogle(
    "Create a dialogue by adding a new field.",
    "auto",
    targetLang
  );
  // Page 7
  const voiceCardContainerRemoveSupport = await translateWithGoogle(
    "Remove unnecessary dialogue fields.",
    "auto",
    targetLang
  );
  // Page 8
  const buttonToggleDivVoiceSupport = await translateWithGoogle(
    "Activate this to voice the selected text field.",
    "auto",
    targetLang
  );
  // Page 9
  const buttonEnableAllSupport = await translateWithGoogle(
    "Activate all text fields for voicing or deselect them all at once.",
    "auto",
    targetLang
  );
  // Page 10
  const buttonRunSynthesisSupport = await translateWithGoogle(
    "Submit the text to the neural network for voicing. This might take a moment.",
    "auto",
    targetLang
  );
  // Page 11
  const buttonShowVoiceWindowSupport = await translateWithGoogle(
    "Switch between synthesized voice results and video.",
    "auto",
    targetLang
  );
  // Page 12
  const buttonRunDeepfakeSynthesisSupportOne = await translateWithGoogle(
    "Face & Lip Animation Creator.",
    "auto",
    targetLang
  );
  const buttonRunDeepfakeSynthesisSupportTwo = await translateWithGoogle(
    "Upload an image, video, or GIF as input.",
    "auto",
    targetLang
  );
  const buttonRunDeepfakeSynthesisSupportThree = await translateWithGoogle(
    "Note: The initial launch will download 5GB of data. This may take a while.",
    "auto",
    targetLang
  );
  // Page 13
  const buttonRunFaceSwapSupportOne = await translateWithGoogle(
    "Face Swap with One Photo.",
    "auto",
    targetLang
  );
  const buttonRunFaceSwapSupportTwo = await translateWithGoogle(
    "For target and face source, upload an image, video, or GIF. ",
    "auto",
    targetLang
  );
  const buttonRunFaceSwapSupportThree = await translateWithGoogle(
    "Note: The initial launch will download 1GB of data. Please be patient.",
    "auto",
    targetLang
  );
  // Page 14
  const buttonRunRetouchOne = await translateWithGoogle(
    "Face Retouch & Object Removal.",
    "auto",
    targetLang
  );
  const buttonRunRetouchTwo = await translateWithGoogle(
    "Enhance deepfake results by smoothing out imperfections.",
    "auto",
    targetLang
  );
  // Page 15
  const buttonRunEditorVideoOne = await translateWithGoogle(
    "Video Editor.",
    "auto",
    targetLang
  );
  const buttonRunEditorVideoTwo = await translateWithGoogle(
    "Improve face or background quality, split videos into frames, extract audio, and recompile frames back into videos.",
    "auto",
    targetLang
  );
  // Page 16
  const aChangeProcessorSupportOne = await translateWithGoogle(
    "Switch from CPU to GPU for faster processing.",
    "auto",
    targetLang
  );
  const aChangeProcessorSupportTwo = await translateWithGoogle(
    "Check the documentation for drivers installation.",
    "auto",
    targetLang
  );
  const aChangeProcessorSupportThree = await translateWithGoogle(
    "After switching to GPU, voice training on your dataset becomes available.",
    "auto",
    targetLang
  );
  // Page 17
  const aLinkOpenfFolderSupport = await translateWithGoogle(
    "Remove unnecessary files from this directory.",
    "auto",
    targetLang
  );
  // Page 17
  const consoleResultSupport = await translateWithGoogle(
    "In this panel, you can monitor the process of the program.",
    "auto",
    targetLang
  );
  // Page 18
  const aLinkOpenAuthorSupportOne = await translateWithGoogle(
    "Discover other projects by the author at",
    "auto",
    targetLang
  );
  const aLinkOpenAuthorSupportTwo = await translateWithGoogle(
    "Enjoy your experience application!",
    "auto",
    targetLang
  );

  var intro = introJs();
  intro.setOptions({
    steps: [
      {
        element: "#a-link-open-tutorial",
        position: "right",
        intro: `
                <div style="width: 200pt;">
                    <text>${welcomeSupportOne}</text><br><br><text>${welcomeSupportTwo} </text><a class="notranslate" href="https://youtube.com/playlist?list=PLJG0sD6007zFJyV78mkU-KW2UxbirgTGr" target="_blank" rel="noopener noreferrer">YouTube</a>.<br><br><text>${welcomeSupportThree} </text><a class="notranslate" href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki" target="_blank" rel="noopener noreferrer">GitHub Wiki</a>.<br><br><text>${welcomeSupportFour} </text><a class="notranslate" href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki" target="_blank" rel="noopener noreferrer">GitHub Issue</a>.
                </div>
                `,
      },
      {
        element: "#translate-application-btn",
        intro: `${translateApplicationBtn}`,
      },
      {
        element: ".text-input",
        intro: `${textInputSupport}`,
      },
      {
        element: ".a-button.setting-tts",
        intro: `<div style="width: 200pt;"><b>${aButtonSettingTtsSupportOne}</b> ${aButtonSettingTtsSupportTwo}</div>`,
      },
      {
        element: ".model-over-select",
        position: "right",
        intro: `<div style="width: 200pt;"><b>${modalOverSelectSupportOne}</b> ${modalOverSelectSupportTwo} <br><br><span style="color:red;">${modalOverSelectSupportThree}</span></div>`,
      },
      {
        element: ".voice-card-container-plus",
        intro: `${voiceCardContainerPlusSupport}`,
      },
      {
        element: ".voice-card-container-remove",
        intro: `${voiceCardContainerRemoveSupport}`,
      },
      {
        element: ".button.toggle-div-voice",
        intro: `${buttonToggleDivVoiceSupport}`,
      },
      {
        element: "#button-enable-all",
        intro: `${buttonEnableAllSupport}`,
      },
      {
        element: "#button-run-synthesis",
        intro: `${buttonRunSynthesisSupport}`,
      },
      {
        element: "#button-show-voice-window",
        intro: `${buttonShowVoiceWindowSupport}`,
      },
      {
        element: "#button-run-deepfake-synthesis",
        intro: `<div style="width: 200pt;"><b>${buttonRunDeepfakeSynthesisSupportOne}</b> ${buttonRunDeepfakeSynthesisSupportTwo} <br><br><span style="color:red;">${buttonRunDeepfakeSynthesisSupportThree}</span></div>`,
      },
      {
        element: "#button-run-face-swap",
        position: "right",
        intro: `<div style="width: 200pt;"><b>${buttonRunFaceSwapSupportOne}</b> ${buttonRunFaceSwapSupportTwo} <br><br><span style="color:red;">${buttonRunFaceSwapSupportThree}</span></div>`,
      },
      {
        element: "#button-run-retouch",
        position: "right",
        intro: `<b>${buttonRunRetouchOne}</b> ${buttonRunRetouchTwo}`,
      },
      {
        element: "#button-run-editor-video",
        position: "right",
        intro: `<b>${buttonRunEditorVideoOne}</b> ${buttonRunEditorVideoTwo}`,
      },
      {
        element: "#a-change-processor",
        position: "right",
        intro: `<div style="width:250pt;">${aChangeProcessorSupportOne} <a href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki" target="_blank" rel="noopener noreferrer"> ${aChangeProcessorSupportTwo}</a><br><br>${aChangeProcessorSupportThree}</div>`,
      },
      {
        element: "#a-link-open-folder",
        position: "right",
        intro: `${aLinkOpenfFolderSupport}`,
      },
      {
        element: "#console-result",
        position: "left",
        intro: `${consoleResultSupport}`,
      },
      {
        element: "#a-link-open-author",
        position: "right",
        intro: `${aLinkOpenAuthorSupportOne} <a target="_blank" rel="noopener noreferrer" href="https://wladradchenko.ru">wladradchenko.ru</a>. <b>${aLinkOpenAuthorSupportTwo}</b>`,
      },
    ],
    showButtons: true,
    showStepNumbers: false,
    nextLabel: `${btnNext}`,
    prevLabel: `${btnBack}`,
    doneLabel: `${btnClose}`,
  });
  intro.start();
}
///SUPPORT USER///

///TRANSLATE///
function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

let translationsCache = {}; // an object to store translations

// Assume fetchJSON() fetches and returns a JSON from a local file
async function loadTranslations() {
  try {
    const response = await fetch("/media/setting/localization.json");
    if (!response.ok) {
      // file not create, create file after first init
      console.log(
        "HTTP error " +
          response.status +
          ". File not created yet!. Now file is created"
      );
    }
    translationsCache = await response.json();
  } catch (err) {
    console.log("Failed to load translations", err);
  }
}

loadTranslations();

async function translateWithGoogle(text, sourceLang, targetLang) {
  // Check if the translation is cached
  if (translationsCache[text] && translationsCache[text][targetLang]) {
    return translationsCache[text][targetLang];
  } else {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sourceLang}&tl=${targetLang}&dt=t&q=${encodeURI(
      text
    )}`;

    try {
      const res = await fetch(url);
      const data = await res.json();
      if (data && data[0] && data[0][0] && data[0][0][0]) {
        const translatedText = capitalizeFirstLetter(data[0][0][0]);

        // Initialize if not already an object
        if (!translationsCache[text]) {
          translationsCache[text] = {};
        }

        // Update cache
        translationsCache[text][targetLang] = translatedText;

        // Send updated translations to the server
        await fetch("/update_translation", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(translationsCache),
        });

        return translatedText;
      }
      throw new Error("Failed to parse Google Translate response.");
    } catch (err) {
      console.error(err);
      return text; // default to returning original text
    }
  }
}

async function translateTitleAttributes(targetLang) {
  let elementsWithTitle = document.querySelectorAll("[title]");
  for (const elem of elementsWithTitle) {
    if (elem.getAttribute("lang-title") !== targetLang) {
      let originalTitle = elem.title.trim();
      if (originalTitle) {
        let translatedTitle = await translateWithGoogle(
          originalTitle,
          "auto",
          targetLang
        );
        elem.title = translatedTitle;
      }
      elem.setAttribute("lang-title", targetLang);
    }
  }
}

async function translateReload(newTargetLang = "ru") {
  targetLang = newTargetLang;

  async function translateTextNode(node, sourceLang, targetLang) {
    if (node.nodeType === Node.TEXT_NODE) {
      const trimmedText = node.nodeValue.trim();
      if (trimmedText) {
        const translatedText = await translateWithGoogle(
          trimmedText,
          sourceLang,
          targetLang
        );
        node.nodeValue = node.nodeValue.replace(trimmedText, translatedText);
      }
    } else if (
      node.nodeType === Node.ELEMENT_NODE &&
      !node.classList.contains("notranslate") &&
      node.getAttribute("lang") !== targetLang
    ) {
      for (let child of node.childNodes) {
        await translateTextNode(child, sourceLang, targetLang);
      }
      node.setAttribute("lang", targetLang);
    }
  }

  async function translatePage() {
    let allTextAreaElements = document.querySelectorAll("textarea");

    allTextAreaElements.forEach(async (elem) => {
      if (
        !elem.classList.contains("notranslate") &&
        elem.getAttribute("lang") !== targetLang
      ) {
        let originalTextArea = elem.placeholder.trim();
        if (originalTextArea) {
          let translatedTextArea = await translateWithGoogle(
            originalTextArea,
            "auto",
            targetLang
          );
          elem.placeholder = translatedTextArea;
          elem.setAttribute("lang", targetLang);
        }
      }
    });

    let allTextElements = document.querySelectorAll(
      "div, p, h1, h2, h3, h4, h5, h6, a, span, li, td, th, option, legend, label, text, button"
    );

    for (const elem of allTextElements) {
      await translateTextNode(elem, "auto", targetLang);
    }

    await translateTitleAttributes(targetLang);
  }

  // Initially translate the page
  translatePage();
}

document.addEventListener("DOMContentLoaded", function () {
  translateReload(targetLang);
});

document.body.addEventListener("click", (event) => {
  if (event.target.tagName === "BUTTON" || event.target.tagName === "I") {
    translateReload(targetLang);
  }
});

async function translateNewUserLang(lang) {
  await translateReload(lang);
}

document
  .getElementById("translate-application-btn")
  .addEventListener("click", function () {
    const dropdown = document.getElementById("language-dropdown");
    dropdown.style.display = "block";
  });

document
  .getElementById("language-dropdown")
  .addEventListener("change", function () {
    let selectedLangCode = this.value;
    let selectedOption = this.options[this.selectedIndex];
    let selectedLangName = selectedOption.getAttribute("name");

    translateNewUserLang(selectedLangCode);
    updateLangSetting(selectedLangCode, selectedLangName);
  });

function updateLangSetting(lang_code, lang_name) {
  const settingsURL = "/record_settings";
  const data = {
    code: lang_code,
    name: lang_name,
  };

  fetch(settingsURL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.response_code === 0) {
        console.log(data.response);
      } else {
        console.error("Failed to update language setting");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// translate html element from string
// translation
async function translateHtmlString(htmlString, targetLang) {
    // Convert string to DOM elements
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlString, 'text/html');

    // Recursive function to traverse and translate nodes
    const translateNodes = async (node) => {
        if (node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== "") {
            node.textContent = await translateWithGoogle(node.textContent.trim(), "auto", targetLang);
        } else {
            for (let child of node.childNodes) {
                await translateNodes(child);
            }
        }
    };

    await translateNodes(doc.body);

    // Convert translated DOM back to a string
    return doc.body.innerHTML;
}
///TRANSLATE///

///UPDATE VERSION///
// Convert version string to array for easy comparison
function versionUpdateToArray(version) {
  return version.split(".").map(Number);
}

function updateVersion(serverVersionData) {
  // Get the version passed from Jinja
  let serverVersionDataJSON = JSON.parse(serverVersionData);
  if (
    Object.keys(serverVersionDataJSON).length !== 0 &&
    serverVersionDataJSON.hasOwnProperty("version")
  ) {
    let serverVersion = serverVersionDataJSON.version;

    // Get the current version displayed in the HTML
    let currentVersion = document
      .getElementById("version")
      .getAttribute("vers");

    // Check if the version from the server is newer
    if (serverVersion !== currentVersion) {
      // Get history witch upper current version
      let allVersionHistory = serverVersionDataJSON.history;

      let currentVersionArray = versionUpdateToArray(currentVersion);

      // Filter versions that are less than or equal to the current version
      let filteredVersions = Object.keys(allVersionHistory)
        .filter((version) => {
          let versionArray = versionUpdateToArray(version);
          for (let i = 0; i < 3; i++) {
            if (versionArray[i] < currentVersionArray[i]) return false;
            if (versionArray[i] > currentVersionArray[i]) return true;
          }
          return false; // if all parts are equal
        })
        .sort(
          (a, b) =>
            versionUpdateToArray(b).join(".") -
            versionUpdateToArray(a).join(".")
        ); // Sort versions in descending order

      // Generate HTML
      let htmlUpdateInfo = "";

      filteredVersions.forEach((version) => {
        htmlUpdateInfo += `<h3>${version}</h3><ul>`;
        let items = allVersionHistory[version].split("\n");
        items.forEach((item) => {
          htmlUpdateInfo += `<li>${item}</li>`;
        });
        htmlUpdateInfo += `<br></ul>`;
      });

      // Update the content of the paragraph
      document.getElementById("version").innerHTML =
        "Доступно обновление " +
        serverVersion +
        `. <button style="text-decoration: underline;background: transparent;border: none;font-size: 8pt;color: blue;" id="version-history-info" onclick="(() => openUpdateHistory(this, '${htmlUpdateInfo}'))()">Что нового?</button>`;
    }
  }
}

function openUpdateHistory(elem, info) {
  var introUpdateVersion = introJs();
  introUpdateVersion.setOptions({
    steps: [
      {
        element: elem,
        title: "Что нового",
        position: "left",
        intro: `
                <div style="max-width: 400pt;max-height: 70vh;padding-left: 20pt;padding-right: 20pt;">${info}</div>
                <a class="introjs-button" href="https://wladradchenko.ru/wunjo" target="_blank" rel="noopener noreferrer" style="margin-top: 20pt;right: 0;left: 0;display: flex;justify-content: center;width: 100%;padding-left: 0;padding-right: 0;">Загрузить обновление</a>
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
  introUpdateVersion.start();
}

updateVersion(serverVersionData);
///UPDATE VERSION///

///GENERATE RANDOM NAME///
function getRandomString(length) {
  const characters =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let randomName = "";
  for (let i = 0; i < length; i++) {
    randomName += characters.charAt(
      Math.floor(Math.random() * characters.length)
    );
  }
  return randomName;
}
///GENERATE RANDOM NAME///

///GET DISK SPACE USED///
function getDriveSpace() {
  fetch("/disk_space_used/")
    .then((response) => response.json())
    .then((data) => {
      // Assuming the sizes are in bytes, convert them to MB and round to 1 decimal place
      const audioSizeMB = data.audio.toFixed(1);
      const videoSizeMB = data.video.toFixed(1);

      // Set the content for the HTML elements
      document.getElementById(
        "drive-space-wavs-used"
      ).textContent = `.wunjo/waves ${audioSizeMB} Mb,`;
      document.getElementById(
        "drive-space-video-used"
      ).textContent = `.wunjo/video ${videoSizeMB} Mb`;
    })
    .catch((error) => {
      console.error("Error fetching disk space used:", error);
    });
}
// Init message about disk space used
getDriveSpace();
// TUpdate information about disk space used each 10 seconds
setInterval(getDriveSpace, 10000);
///GET DISK SPACE USED///
