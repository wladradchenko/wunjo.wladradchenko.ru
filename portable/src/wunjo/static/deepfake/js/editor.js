async function initializeMediaEditor(button, audioURL = undefined, audioName = undefined) {
    const audioInputField = `
        <div style="margin-top: 10pt;margin-bottom: 10pt;display: flex;">
            <label id="upload-audio-for-merge-label" for="upload-audio-for-merge" class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Load audio</label>
            <input style="width: 0;" accept=".mp3,.wav,.ogg,.flac" type="file" onChange="dragDropAudioVideoMerge(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="upload-audio-for-merge"  />
            <div id="preview-audio-for-merge"></div>
        </div>
    `;

    // translated media edit
    const introMediaEdit = `
                    <div style="width: 60vw; max-width: 70vw; height: 70vh; max-height: 80vh;align-items: inherit;display: flex;flex-direction: column;justify-content: space-between">
                    <div></div>
                    <div>
                        <div style="display: flex;flex-direction: column;justify-content: center;align-items: center;">
                            <span class="dragBox" style="margin-bottom: 15px;width: 60vw;display: flex;text-align: center;flex-direction: column;position: relative;justify-content: center;height: 45vh;">
                                  Load image, video or audio
                                <input accept="image/*,video/*,audio/*" type="file" onChange="handleEditorVideo(event, document.getElementById('preview-media'), this.parentElement)" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" />
                            </span>
                            <p id="message-about-status" style="text-align: center;color: #393939;height: 30px;display: none;justify-content: center;align-items: center;padding: 5px;margin-bottom: 15px;"></p>
                            <div id="preview-media" style="position: relative;max-width: 60vw; max-height:60vh;display: flex;flex-direction: column;align-items: center;">
                            </div>
                        </div>
                    </div>
                    <div style="display: flex;flex-direction: column;justify-content: space-between;">
                        <fieldset style="padding: 5pt;">
                            <legend>Processing mode</legend>
                            <div style="display: flex;flex-direction: row;">
                                <div>
                                    <div>
                                      <input type="radio" id="gfpgan" name="preprocessing" value="gfpgan" checked>
                                      <label for="gfpgan">Improve face quality</label>
                                    </div>
                                    <div id="realesrganDiv">
                                      <input type="radio" id="realesrgan" name="preprocessing" value="realesrgan"  onclick="radioSetMessage(document.getElementById('message-about-status'), 'Sides under 640px will be adjusted to 640px, keeping the aspect ratio');">
                                      <label for="realesrgan">Improve visual quality</label>
                                    </div>
                                    <div id="animesganDiv">
                                      <input type="radio" id="animesgan" name="preprocessing" value="animesgan" onclick="radioSetMessage(document.getElementById('message-about-status'), 'Sides under 640px will be adjusted to 640px, keeping the aspect ratio');">
                                      <label for="animesgan">Improve hand-drawn quality</label>
                                    </div>
                                    <div>
                                      <input type="radio" id="getFrames" name="preprocessing" value="frames">
                                      <label for="getFrames">Extract frames</label>
                                    </div>
                                </div>
                                <div style="margin-left:5vw;">
                                    <div>
                                      <input type="radio" id="getVoicefixer" name="preprocessing" value="voicefixer">
                                      <label for="getVoicefixer">Speech enhancement</label>
                                    </div>
                                    <div>
                                      <input type="radio" id="getVocals" name="preprocessing" value="vocals">
                                      <label for="getVoice">Extract vocals</label>
                                    </div>
                                    <div>
                                      <input type="radio" id="getResidual" name="preprocessing" value="residual">
                                      <label for="getBackground">Extract accompaniment</label>
                                    </div>
                                </div>
                            </div>
                        </fieldset>
                        <div>
                            <p id="message-editor-video" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                            <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="triggerMediaEditorProcess(this.parentElement.parentElement.parentElement);">Start processing</button>
                        </div>
                    </div>
                    </div>
                    `;

    const introTranslatedMediaEdit = await translateHtmlString(introMediaEdit, targetLang);
    const introTranslatedTitleMediaEdit = await translateWithGoogle("Panel media content editor","auto",targetLang);

    // translate video merge
    const introMediaMerge = `
                    <div style="width: 450px;display: flex;flex-direction: column;align-items: center;">
                        <div style="padding: 5pt;display: flex;flex-direction: column;">
                           <div style="display: flex;flex-direction: row;align-items: center;justify-content: space-between;">
                                <div>
                                <label for="frames-path">Path to frames</label>
                                <input type="text" id="frames-path" style="border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;margin-top: 5pt;">
                                </div>
                                <div>
                                <label for="video-fps">FPS </label>
                                <input type="number" title="Input number" id="video-fps" name="video-fps" min="1" max="30" step="1" value="30" style="border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;margin-top: 5pt;width: 60pt;">
                                </div>
                            </div>
                            <i style="margin-top: 5pt;font-size: 10pt;"><b>Note:</b> Specify the directory path containing frames to create a video and frames should be named from 1.png up to 99...9.png</i>
                        </div>
                        <div style="padding: 5pt;display: flex;flex-direction: column;width: 450px;">
                            <div style="margin-bottom:5pt;">
                              <input onclick="document.getElementById('editor-video-audio').style.display = this.checked ? 'block' : 'none';" type="checkbox" id="editor-video-audio-checkpoint-info" name="editor-video-audio-checkpoint">
                              <label for="editor-video-audio-checkpoint">Merge to audio</label>
                            </div>
                            <div id="editor-video-audio" style="display:none;margin-top:5pt;">
                                ${audioInputField}
                            </div>
                            <div>
                                <p id="message-editor-video" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                                <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="triggerVideoMerge(this.parentElement.parentElement.parentElement);">Start processing</button>
                            </div>
                        </div>
                    </div>
                    `;

    const introTranslatedMediaMerge = await translateHtmlString(introMediaMerge, targetLang);
    const introTranslatedTitleMediaMerge = await translateWithGoogle("Panel merge images to video","auto",targetLang);

    // translated buttons
    nextTranslatedLabel = await translateWithGoogle("Next","auto",targetLang);
    prevTranslatedLabel = await translateWithGoogle("Back","auto",targetLang);
    doneTranslatedLabel = await translateWithGoogle("Close","auto",targetLang);

    const introEditorVideo = introJs();
    introEditorVideo.setOptions({
        steps: [
          {
            title: introTranslatedTitleMediaEdit,
            position: "right",
            intro: introTranslatedMediaEdit,
          },
          {
            element: button,
            title: introTranslatedTitleMediaMerge,
            position: "right",
            intro: introTranslatedMediaMerge,
          },
        ],
        showButtons: true,
        showStepNumbers: false,
        showBullets: false,
        nextLabel: nextTranslatedLabel,
        prevLabel: prevTranslatedLabel,
        doneLabel: doneTranslatedLabel,
    });
    introEditorVideo.setOption('keyboardNavigation', false).start();
    availableFeaturesByCUDA(document.getElementById("realesrganDiv"));
    availableFeaturesByCUDA(document.getElementById("animesganDiv"));
}

async function radioSetMessage(msgElement, msg) {
    msgElement.style.display = "flex";
    msgElement.style.background = getRandomColor();
    messageAboutStatusText = await translateWithGoogle(msg,"auto",targetLang);
    msgElement.innerHTML = `${messageAboutStatusText}`;
}

function dragDropAudioVideoMerge(event) {
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
            messageAboutStatusText = await translateWithGoogle("Choose a preprocessing mode","auto",targetLang);
            messageElement.innerHTML = `${messageAboutStatusText}`;
            canvas = await setupImageCanvas(previewElement, fileUrl, "35vh", "80vw");
            document.getElementById("gfpgan").checked = true;
            document.getElementById("getFrames").disabled = true;
            document.getElementById("getVocals").disabled = true;
            document.getElementById("getResidual").disabled = true;
        } else if (fileType === 'video') {
            messageElement.style.display = "flex";
            messageElement.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Video is loading...","auto",targetLang);
            messageElement.innerHTML = `${messageAboutStatusText}`;
            canvas = await setupVideoTimeline(previewElement, fileUrl, "35vh", "80vw");
            messageElement.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Video was loaded and you can cut the video and choose preprocessing mode","auto",targetLang);
            messageElement.innerHTML = `${messageAboutStatusText}`;
            let getFrames = document.getElementById("getFrames");
            if (getFrames) {
               getFrames.disabled = false;
            };
            let getVocals = document.getElementById("getVocals");
            if (getVocals) {
               getVocals.disabled = false;
            };
            let getResidual = document.getElementById("getResidual");
            if (getResidual) {
               getResidual.disabled = false;
            };
        } else if (fileType === 'audio') {
            document.getElementById("gfpgan").disabled = true;
            document.getElementById("realesrgan").disabled = true;
            document.getElementById("animesgan").disabled = true;
            document.getElementById("getFrames").disabled = true;

            document.getElementById("getVocals").checked = true;
            document.getElementById("getVocals").disabled = false;
            document.getElementById("getResidual").disabled = false;

            previewElement.innerHTML = `
            <div style="display: flex;align-items: center;justify-content: space-around;flex-direction: row;max-height: 60vh;width: 60vw;height: 33px;">
                <button id="audio-play-button-separator" class="introjs-button" style="display:inline;">
                    <i class="fa fa-play"></i>
                    <i style="display: none;" class="fa fa-pause"></i>
                </button>
                <div id="audio-progress-container" style="height:100%;width:100%;display: inline-block;padding: 3px;border: 1px solid #0000005e;margin-left: 10px;">
                    <div id="audio-progress" style="width: 0%; height: 100%; background-color: #f7db4d;"></div>
                </div>
                <audio id="audio-play-separator" style="display:none;" controls preload="none">
                    <source src="${fileUrl}" class="audioMedia">
                    Your browser does not support audio.
                </audio>
            </div>
            `;

            const playBtn = document.getElementById("audio-play-button-separator");
            const audio = document.getElementById("audio-play-separator");
            const progressBar = document.getElementById("audio-progress");

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
                progressBar.style.width = "0%";
            });

            audio.addEventListener("timeupdate", () => {
                let percentage = (audio.currentTime / audio.duration) * 100;
                progressBar.style.width = `${percentage}%`;
            });

            messageElement.style.display = "flex";
            messageElement.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Audio was loaded","auto",targetLang);
            messageElement.innerHTML = `${messageAboutStatusText}`;
        };
    }
}

function triggerMediaEditorProcess(elem) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => handleMediaEditorProcess(data, elem))
        .catch(error => {
            console.error("Error fetching message for deepfake:", error);
        });
}

async function handleMediaEditorProcess(data, elem) {
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

    executeMediaEditorProcess(mediaDetails, elem);
}

function clearMessage(element) {
    element.innerHTML = "";
    element.style.display = "none";
}

function executeMediaEditorProcess(mediaDetails, elem) {
    const parameters = {
        source: mediaDetails.mediaName,
        gfpgan: elem.querySelector("#gfpgan").checked ? "gfpgan" : false,
        animesgan: elem.querySelector("#animesgan").checked ? "animesgan" : false,
        realesrgan: elem.querySelector("#realesrgan").checked ? "realesrgan" : false,
        get_frames: elem.querySelector("#getFrames").checked,
        vocals: elem.querySelector("#getVocals").checked ? "vocals" : false,
        residual: elem.querySelector("#getResidual").checked ? "residual" : false,
        voicefixer: elem.querySelector("#getVoicefixer").checked ? true : false,
        media_start: mediaDetails.mediaStart,
        media_end: mediaDetails.mediaEnd,
        media_type: mediaDetails.mediaType,
    };

    fetch("/synthesize_media_editor/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(parameters),
    });

    // This open display result for deepfake videos
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

    const messageElement = elem.querySelector("#message-editor-video");
    clearMessage(messageElement);

    if (data.status_code !== 200) {
        displayTranslatedMessage(messageElement, "The process is busy. Wait for it to finish.");
        return;
    }

    const framesPath = elem.querySelector("#frames-path").value;
    if (!framesPath) {
        displayTranslatedMessage(messageElement, "You did not specify the absolute path to the images.");
        return;
    }

    const audioName = await processAudioData(elem.querySelector("#audio-play-for-merge"));

    const videoFps = elem.querySelector("#video-fps").value;
    const mergeParameters = {
        source_folder: framesPath,
        audio_name: audioName,
        fps: videoFps
    };

    fetch("/synthesize_video_merge/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(mergeParameters),
    });

    // This open display result for deepfake videos
    closeTutorial();
}
