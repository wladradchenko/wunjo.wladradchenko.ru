function initializeVideoEditor(button, audioURL = undefined, audioName = undefined) {
    function setupAudioPreview(url, name) {
        const audioInputLabel = document.getElementById("upload-audio-for-merge-label");
        audioInputLabel.textContent = name.length > 20 ? name.slice(0, 20) + "..." : name;

        const audioInputButton = document.getElementById("upload-audio-for-merge");
        audioInputButton.disabled = true;

        const audioBlob = URL.createObjectURL(url);
        const audioPreview = document.getElementById("preview-audio-for-merge");
        audioPreview.innerHTML = `
            <button id="audio-play-button-for-merge" class="introjs-button" style="display:inline;margin-left: 5pt;">
                <i class="fa fa-play"></i>
                <i style="display: none;" class="fa fa-pause"></i>
            </button>
            <audio id="audio-play-for-merge" style="display:none;" controls preload="none">
                <source src="${audioBlob}">
                Your browser does not support audio.
            </audio>
        `;

        const playBtn = document.getElementById("audio-play-button-for-merge");
        const audioElement = document.getElementById("audio-play-for-merge");

        playBtn.addEventListener("click", function () {
            if (audioElement.paused) {
                audioElement.play();
                playBtn.children[0].style.display = "none";
                playBtn.children[1].style.display = "inline";
            } else {
                audioElement.pause();
                playBtn.children[0].style.display = "inline";
                playBtn.children[1].style.display = "none";
            }
        });

        audioElement.addEventListener("ended", function () {
            playBtn.children[0].style.display = "inline";
            playBtn.children[1].style.display = "none";
        });
    }

    const audioInputField = `
        <div style="margin-top: 10pt;margin-bottom: 10pt;display: flex;">
            <label id="upload-audio-for-merge-label" for="upload-audio-for-merge" class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Загрузить аудио</label>
            <input style="width: 0;" accept="audio/*" type="file" onChange="dragDropAudioVideoMerge(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="upload-audio-for-merge"  />
            <div id="preview-audio-for-merge"></div>
        </div>
    `;

    if (audioURL) {
        const request = new XMLHttpRequest();
        request.open("GET", audioURL, true);
        request.responseType = "blob";
        request.onload = function () {
            setupAudioPreview(request.response, audioName);
        };
        request.send();
    }

    const introEditorVideo = introJs();
    introEditorVideo.setOptions({
        steps: [
          {
            title: "Панель обработки видео и изображений",
            position: "right",
            intro: `
                    <div style="width: 60vw; max-width: 70vw; height: 70vh; max-height: 80vh;align-items: inherit;display: flex;flex-direction: column;justify-content: space-between">
                    <div></div>
                    <div>
                        <div style="display: flex;flex-direction: column;justify-content: center;align-items: center;">
                            <span class="dragBox" style="margin-bottom: 15px;width: 60vw;display: flex;text-align: center;flex-direction: column;position: relative;justify-content: center;height: 45vh;">
                                  Загрузите изображение или видео
                                <input accept="image/*,video/*" type="file" onChange="handleEditorVideo(event, document.getElementById('preview-media'), this.parentElement)" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" />
                            </span>
                            <p id="message-about-status" style="text-align: center;color: #393939;height: 30px;display: none;justify-content: center;align-items: center;padding: 5px;margin-bottom: 15px;"></p>
                            <div id="preview-media" style="position: relative;max-width: 60vw; max-height:60vh;display: flex;flex-direction: column;align-items: center;">
                            </div>
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
                            <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="triggerVideoEditorProcess(this.parentElement.parentElement.parentElement);">Начать обрабатывать</button>
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
                                <input type="text" id="frames-path" style="border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;margin-top: 5pt;">
                                </div>
                                <div>
                                <label for="video-fps">FPS </label>
                                <input type="number" title="Введите число" id="video-fps" name="video-fps" min="1" max="30" step="1" value="30" style="border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;margin-top: 5pt;width: 60pt;">
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
                                <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="triggerVideoMerge(this.parentElement.parentElement.parentElement);">Начать обрабатывать</button>
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

function dragDropAudioVideoMerge(event) {
    console.log(event)
    if (event.target.files.length === 0) {
        console.warn("No files selected");
        return;
    }

    const fileURL = URL.createObjectURL(event.target.files[0]);
    const audioPreview = document.getElementById("preview-audio-for-merge");
    audioPreview.innerHTML = `
        <button id="audio-play-button-for-merge" class="introjs-button" style="display:inline;margin-left: 5pt;">
            <i class="fa fa-play"></i>
            <i style="display: none;" class="fa fa-pause"></i>
        </button>
        <audio id="audio-play-for-merge" style="display:none;" controls preload="none">
            <source src="${fileURL}">
            Your browser does not support audio.
        </audio>
    `;

    const playBtn = document.getElementById("audio-play-button-for-merge");
    const audio = document.getElementById("audio-play-for-merge");

    playBtn.addEventListener("click", () => {
        if (audio.paused) {
            audio.play();
            playBtn.querySelector(".fa-play").style.display = "none";
            playBtn.querySelector(".fa-pause").style.display = "inline";
        } else {
            audio.pause();
            playBtn.querySelector(".fa-play").style.display = "inline";
            playBtn.querySelector(".fa-pause").style.display = "none";
        }
    });

    audio.addEventListener("ended", () => {
        playBtn.querySelector(".fa-play").style.display = "inline";
        playBtn.querySelector(".fa-pause").style.display = "none";
    });
}


async function handleEditorVideo(event, previewElement, parentElement) {
    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        const fileUrl = window.URL.createObjectURL(file);
        const fileType = file.type.split('/')[0];
        parentElement.style.height = "30px";
        previewElement.innerHTML = "";
        const messageElement = document.getElementById("message-about-status");
        let messageAboutStatusText;

        let canvas;
        if (fileType === 'image') {
            messageElement.style.display = "flex";
            messageElement.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Choose a face to animate by tool","auto",targetLang);
            messageElement.innerHTML = `${messageAboutStatusText} <i class="fa-solid fa-draw-polygon" style="margin-left: 10px;"></i>`;
            canvas = await setupImageCanvas(previewElement, fileUrl, "35vh", "80vw");
            document.getElementById("enhancer-face").checked = true;
            document.getElementById("get-frames").disabled = true;
        } else if (fileType === 'video') {
            messageElement.style.display = "flex";
            messageElement.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Video is loading...","auto",targetLang);
            messageElement.innerHTML = `${messageAboutStatusText}`;
            canvas = await setupVideoTimeline(previewElement, fileUrl, "35vh", "80vw");
            messageElement.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Video was loaded and you can cut the video","auto",targetLang);
            messageElement.innerHTML = `${messageAboutStatusText}`;
            document.getElementById("get-frames").disabled = false;
        }
    }
}

function triggerVideoEditorProcess(elem) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => handleVideoEditorProcess(data, elem))
        .catch(error => {
            console.error("Error fetching message for deepfake:", error);
        });
}

async function handleVideoEditorProcess(data, elem) {
    const messageElement = elem.querySelector("#message-about-status");
    clearMessage(messageElement);

    if (data.status_code !== 200) {
        displayMessage(messageElement, "The process is busy. Wait for the previous process to finish");
        return;
    }

    const mediaDetails = retrieveMediaDetails(elem.querySelector("#preview-media"));
    if (!mediaDetails.mediaName) {
        displayMessage(messageElement, "Input video or image");
        return;
    }

    executeVideoEditorProcess(mediaDetails, elem);
}

function clearMessage(element) {
    element.innerHTML = "";
    element.style.display = "none";
}

function executeVideoEditorProcess(mediaDetails, elem) {
    const buttonAnimationWindows = document.querySelector("#button-show-voice-window");
    buttonAnimationWindows.click();

    const parameters = {
        source: mediaDetails.mediaName,
        enhancer: elem.querySelector("#enhancer-face").checked ? "gfpgan" : false,
        enhancer_background: elem.querySelector("#enhancer-background").checked,
        get_frames: elem.querySelector("#get-frames").checked,
        media_start: mediaDetails.mediaStart,
        media_end: mediaDetails.mediaEnd,
    };

    document.getElementById("table_body_deepfake_result").innerHTML = "";

    fetch("/synthesize_video_editor/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(parameters),
    });

    // This open display result for deepfake videos
    const tutorialButton = document.querySelector("#button-show-voice-window");
    tutorialButton.click();
    closeTutorial();
}


function triggerVideoMerge(elem) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => handleVideoMergeProcess(data, elem))
        .catch(error => {
            console.error("Error fetching message for deepfake:", error);
        });
}

async function handleVideoMergeProcess(data, elem) {
    function clearMessage(element) {
        element.innerHTML = "";
    }

    async function displayTranslatedMessage(element, message) {
        const translatedMessage = await translateWithGoogle(message, "auto", targetLang);
        element.innerHTML = `<p style='margin-top: 5pt;'>${translatedMessage}</p>`;
    }

    async function processAudioData(audioElement) {
        if (!audioElement) return null;

        const audioBlobUrl = audioElement.querySelector("source").src;
        const audioName = "audio_" + Date.now();

        const blob = await fetch(audioBlobUrl).then(res => res.blob());
        const file = new File([blob], audioName);
        uploadFile(file);

        return audioName;
    }

    const synthesisTable = document.getElementById("table_body_deepfake_result");
    const messageElement = elem.querySelector("#message-editor-video");
    clearMessage(messageElement);

    if (data.status_code !== 200) {
        displayTranslatedMessage(messageElement, "Процесс занят. Дождитесь его окончания.");
        return;
    }

    const framesPath = elem.querySelector("#frames-path").value;
    if (!framesPath) {
        displayTranslatedMessage(messageElement, "Вы не указали абсолютный путь до изображений.");
        return;
    }

    const audioName = await processAudioData(elem.querySelector("#audio-play-for-merge"));

    const videoFps = elem.querySelector("#video-fps").value;
    const mergeParameters = {
        source_folder: framesPath,
        audio_name: audioName,
        fps: videoFps
    };

    synthesisTable.innerHTML = "";

    fetch("/synthesize_video_merge/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(mergeParameters),
    });

    // This open display result for deepfake videos
    const tutorialButton = document.querySelector("#button-show-voice-window");
    tutorialButton.click();
    closeTutorial();
}
