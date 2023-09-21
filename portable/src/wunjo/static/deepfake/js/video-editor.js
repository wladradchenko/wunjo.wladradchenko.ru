function editorVideo(button, audio_url = undefined, audio_name = undefined) {
  var audioInputField = `
                          <div class="uploadOuterDeepfakeAudio" style="margin-top: 10pt;margin-bottom: 10pt;display: flex;">
                            <label id="uploadAudioDeepfakeLabel" for="uploadAudioDeepfake" class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Загрузить аудио</label>
                            <input style="width: 0;" accept="audio/*" type="file" onChange="dragDropAudio(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadAudioDeepfake"  />
                            <div id="previewAudio"></div>
                          </div>
                         `;

  if (audio_url) {
    var request = new XMLHttpRequest();
    request.open("GET", audio_url, true);
    request.responseType = "blob";
    request.onload = function () {
      var audioInputLabel = document.getElementById("uploadAudioDeepfakeLabel");
      audioInputLabel.textContent =
        audio_name.length > 20 ? audio_name.slice(0, 20) + "..." : audio_name;

      var audioInputButton = document.getElementById("uploadAudioDeepfake");
      audioInputButton.disabled = true;

      var audioBlobMedia = URL.createObjectURL(request.response);
      var audioPreview = document.getElementById("previewAudio");
      audioPreview.innerHTML = `
          <button id="audioDeepfakePlay" class="introjs-button" style="display:inline;margin-left: 5pt;">
            <i class="fa fa-play"></i>
            <i style="display: none;" class="fa fa-pause"></i>
          </button>
          <audio id="audioDeepfakeSrc" style="display:none;" controls="" preload="none">
            <source src="${audioBlobMedia}">
            Your browser does not support audio.
          </audio>
        `;
      var playBtn = document.getElementById("audioDeepfakePlay");
      var audio = document.getElementById("audioDeepfakeSrc");
      // Set audio length on the text element
      // Wait for metadata to be loaded
      audio.onloadedmetadata = function () {
        // Set audio length on the text element
        var audioLength = document.getElementById("audio-length");
        audioLength.innerText = audio.duration.toFixed(1); // rounded to 2 decimal places
      };

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

      audio.addEventListener("ended", function () {
        playBtn.children[0].style.display = "inline";
        playBtn.children[1].style.display = "none";
      });
    };
    request.send();
  }

  var introEditorVideo = introJs();
  introEditorVideo.setOptions({
    steps: [
      {
        element: button,
        title: "Панель обработки видео",
        position: "right",
        intro: `
                <div style="width: 400pt;width: 400pt;display: flex;justify-content: space-around;flex-direction: inherit;">
                <div style="width: 200pt;">
                    <div class="uploadTargetFile">
                        <div style="flex-direction: row;display: none;margin-bottom: 10pt;justify-content: space-between;">
                            <button class="introjs-button" style="display: none;margin-right: 5pt;" id="clearButtonTarget">Очистить</button>
                            <button style="width: 100%;display: none;" class="introjs-button" id="drawButtonTarget" data-controlval="get-face">Выделить лицо для замены</button>
                        </div>
                        <span id="previewEditorVideo" class="dragBox" style="height: 200pt;justify-content: center;">
                          Загрузить целевое изображение или видео
                        <input accept="image/*,video/*" type="file" onChange="dragDropImgOrVideo(event, 'previewEditorVideo', 'canvasTarget', this, 'clearButtonTarget', 'drawButtonTarget');"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadTarget"  />
                        </span>
                    </div>
                </div>
                <div style="display: flex;flex-direction: column;justify-content: space-between;">
                    <fieldset style="padding: 5pt;">
                        <legend>Выбор препроцессинга</legend>
                        <div>
                          <input type="radio" id="enhancer-face" name="preprocessing_editor" value="face">
                          <label for="enhancer-face">Улучшить лицо</label>
                        </div>
                        <div>
                          <input type="radio" id="enhancer-background" name="preprocessing_editor" value="background">
                          <label for="enhancer-background">Улучшить окружение</label>
                        </div>
                        <div>
                          <input type="radio" id="get-frames" name="preprocessing_editor" value="frames" checked>
                          <label for="get-frames">Получить кадры</label>
                        </div>
                    </fieldset>
                    <div>
                        <p id="message-editor-video" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                        <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="sendDataToVideoEditor(this.parentElement.parentElement.parentElement);">Начать обрабатывать</button>
                    </div>
                </div>
                </div>
                <script>
                    availableFeaturesByCUDA(document.getElementById("enhancer-background"));
                </script>
                `,
      },
      {
        element: button,
        title: "Панель обработки видео",
        position: "right",
        intro: `
                <div style="width: 280pt;display: flex;flex-direction: column;align-items: center;">
                    <div style="padding: 5pt;display: flex;flex-direction: column;">
                       <div style="display: flex;flex-direction: row;align-items: center;justify-content: space-between;">
                            <div>
                            <label for="frames-path">Путь до кадров</label>
                            <input type="text" id="frames-path" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;margin-top: 5pt;">
                            </div>
                            <div>
                            <label for="video-fps">FPS </label>
                            <input type="number" title="Введите число" id="video-fps" name="video-fps" min="1" max="30" step="1" value="30" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;margin-top: 5pt;width: 60pt;">
                            </div>
                        </div>
                        <i style="margin-top: 5pt;font-size: 10pt;"><b>Примечание:</b> В этом окне вы можете указав путь до директории с кадрами, объединить их в видео. Кадры должны назваться с 1.png по 99...9.png.</i>
                    </div>
                    <div style="padding: 5pt;display: flex;flex-direction: column;width: 370px;">
                        <div style="margin-bottom:5pt;">
                          <input onclick="document.getElementById('editor-video-audio').style.display = this.checked ? 'block' : 'none';" type="checkbox" id="editor-video-audio-checkpoint-info" name="editor-video-audio-checkpoint">
                          <label for="editor-video-audio-checkpoint">Добавить аудио</label>
                        </div>
                        <div id="editor-video-audio" style="display:none;margin-top:5pt;">
                            ${audioInputField}
                        </div>
                        <div>
                            <p id="message-editor-video" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                            <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="sendDataToVideoMerge(this.parentElement.parentElement.parentElement);">Начать обрабатывать</button>
                        </div>
                    </div>
                </div>
                `,
      },
    ],
    showButtons: true,
    showStepNumbers: false,
    showBullets: true,
    nextLabel: "Продолжить",
    prevLabel: "Вернуться",
    doneLabel: "Закрыть",
  });
  introEditorVideo.start();
}

function dragDropAudio(event) {
  if (event.target.files.length === 0) {
    console.warn("No files selected");
    return; // Exit the function if no files were selected
  }

  var file = URL.createObjectURL(event.target.files[0]);
  // Get audio length
  var audioElement = new Audio(file);
  var reader = new FileReader();
  var preview = document.getElementById("previewAudio");
  preview.innerHTML = `<button id="audioDeepfakePlay" class="introjs-button" style="display:inline;margin-left: 5pt;">
                          <i class="fa fa-play"></i>
                          <i style="display: none;" class="fa fa-pause"></i>
                      </button>
                      <audio id="audioDeepfakeSrc" style="display:none;" controls preload="none">
                        <source src="${file}">
                        Your browser does not support audio.
                      </audio>`;

  var playBtn = document.getElementById("audioDeepfakePlay");
  var audio = document.getElementById("audioDeepfakeSrc");

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

  audio.addEventListener("ended", function () {
    playBtn.children[0].style.display = "inline";
    playBtn.children[1].style.display = "none";
  });
}

function sendDataToVideoEditor(elem) {
  // If process is free
  fetch("/synthesize_process/")
    .then((response) => response.json())
    .then((data) => {
      // Call the async function
      processAsyncVideoEditor(data, elem)
        .then(() => {
          console.log("Start to fetch msg for deepfake");
        })
        .catch((error) => {
          console.log("Error to fetch msg for deepfake");
          console.log(error);
        });
    });
}

async function processAsyncVideoEditor(data, elem) {
  if (data.status_code === 200) {
    var synthesisDeepfakeTable = document.getElementById(
      "table_body_deepfake_result"
    );

    var messageVideoEditor = elem.querySelector("#message-editor-video");
    messageVideoEditor.innerHTML = "";

    // Get target content
    var previewVideoEditor = elem.querySelector("#previewEditorVideo");

    var imgVideoEditor = previewVideoEditor.querySelector("img");
    var videoVideoEditor = previewVideoEditor.querySelector("video");
    var mediaNameTarget = "";
    var mediaBlobUrlTarget = "";
    var typeFileTarget = "";

    if (imgVideoEditor) {
      var messageSetP = await translateWithGoogle(
        "Загрузите видео.",
        "auto",
        targetLang
      );
      messageEditorVideo.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
      return;
    } else if (videoVideoEditor) {
      typeFileTarget = "video";
      mediaBlobUrlTarget = videoVideoEditor.src;
      mediaNameTarget = "video_target_" + Date.now() + "_" + getRandomString(5);
    } else {
      var messageSetP = await translateWithGoogle(
        "Вы не загрузили видео. Нажмите на окно загрузки.",
        "auto",
        targetLang
      );
      messageFaceSwap.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
    }
    if (mediaBlobUrlTarget) {
      fetch(mediaBlobUrlTarget)
        .then((res) => res.blob())
        .then((blob) => {
          var file = new File([blob], mediaNameTarget);
          uploadFile(file);
        });
    }

    // Get preprocessing
    var enhancerFace = elem.querySelector("#enhancer-face");
    var enhancerBackground = elem.querySelector("#enhancer-background");
    var frames = elem.querySelector("#get-frames");
    if (enhancerFace.checked || enhancerBackground.checked) {
      enhancerFace = "gfpgan";
    } else {
      enhancerFace = false;
    }

    if (mediaNameTarget) {
      const buttonAnimationWindows = document.querySelector(
        "#button-show-voice-window"
      );
      buttonAnimationWindows.click();

      var predictParametersFaceSwap = {
        source: mediaNameTarget,
        enhancer: enhancerFace,
        enhancer_background: enhancerBackground.checked,
        get_frames: frames.checked,
      };

      synthesisDeepfakeTable.innerHTML = "";

      fetch("/synthesize_video_editor/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(predictParametersFaceSwap),
      });

      const closeIntroButton = document.querySelector(".introjs-skipbutton");
      closeIntroButton.click();
    }
  } else {
    var synthesisDeepfakeTable = document.getElementById(
      "table_body_deepfake_result"
    );

    var messageEditorVideo = elem.querySelector("#message-editor-video");
    var messageSetP = await translateWithGoogle(
      "Процесс занят. Дождитесь его окончания.",
      "auto",
      targetLang
    );
    messageEditorVideo.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
  }
}

function sendDataToVideoMerge(elem) {
  // If process is free
  fetch("/synthesize_process/")
    .then((response) => response.json())
    .then((data) => {
      // Call the async function
      processAsyncVideoMerge(data, elem)
        .then(() => {
          console.log("Start to fetch msg for deepfake");
        })
        .catch((error) => {
          console.log("Error to fetch msg for deepfake");
          console.log(error);
        });
    });
}

async function processAsyncVideoMerge(data, elem) {
  if (data.status_code === 200) {
    var synthesisDeepfakeTable = document.getElementById(
      "table_body_deepfake_result"
    );

    var messageVideoEditor = elem.querySelector("#message-editor-video");
    messageVideoEditor.innerHTML = "";

    const framesPath = elem.querySelector("#frames-path");

    if (!framesPath) {
      var messageSetP = await translateWithGoogle(
        "Вы не указали абсолютный путь до изображений.",
        "auto",
        targetLang
      );
      messageVideoEditor.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
    }

    var audioMergeSrc = elem.querySelector("#audioDeepfakeSrc");
    var audioName = "";
    if (audioMergeSrc) {
      var audioBlobUrl = audioMergeSrc.querySelector("source").src;
      audioName = "audio_" + Date.now();
      fetch(audioBlobUrl)
        .then((res) => res.blob())
        .then((blob) => {
          var file = new File([blob], audioName);
          uploadFile(file);
        });
    } else {
      audioName = null;
    }

    const videoFps = elem.querySelector("#video-fps");

    const buttonAnimationWindows = document.querySelector(
      "#button-show-voice-window"
    );
    buttonAnimationWindows.click();

    var predictParametersFaceSwap = {
      source_folder: framesPath.value,
      audio_name: audioName,
      fps: videoFps.value,
    };

    synthesisDeepfakeTable.innerHTML = "";

    fetch("/synthesize_video_merge/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(predictParametersFaceSwap),
    });

    const closeIntroButton = document.querySelector(".introjs-skipbutton");
    closeIntroButton.click();
  } else {
    var synthesisDeepfakeTable = document.getElementById(
      "table_body_deepfake_result"
    );

    var messageEditorVideo = elem.querySelector("#message-editor-video");
    var messageSetP = await translateWithGoogle(
      "Процесс занят. Дождитесь его окончания.",
      "auto",
      targetLang
    );
    messageEditorVideo.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
  }
}
