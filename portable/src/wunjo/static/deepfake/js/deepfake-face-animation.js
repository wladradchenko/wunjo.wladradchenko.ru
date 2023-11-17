// SEND DATA IN BACKEND //
function triggerFaceAndMouthSynthesis(elem) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => initiateFaceAndMouthProcess(data, elem))
        .catch(error => {
            console.error("Error fetching the synthesis process status:", error);
        });
}

async function initiateFaceAndMouthProcess(data, elem) {
    if (data.status_code !== 200) {
        displayStatusMessage(elem, "The process is busy. Wait for it to finish");
        return null;
    }

    const mediaPreview = elem.querySelector("#preview-media");
    const selectedFaceData = retrieveSelectedFaceData(mediaPreview);

    const {
        mediaType: mediaType,
        mediaName: mediaName,
        mediaBlobUrl: mediaBlobUrl,
        mediaStart: mediaStart,
        mediaEnd: mediaEnd
    } = retrieveMediaDetails(mediaPreview);

    const audioDetails = retrieveAudioDetailsFaceAndMouthAnimation(elem);

    const synthesisSettings = gatherSynthesisSettings(elem, mediaType, mediaName, mediaStart, mediaEnd, audioDetails.audioName, selectedFaceData);

    if (synthesisSettings) {
        triggerSynthesisAPI(synthesisSettings);
        closeTutorial();
    }

    // FUNCTIONS TO GET PARAMETERS //
    function retrieveAudioDetailsFaceAndMouthAnimation(elem) {
        const audioElement = elem.querySelector("#audioDeepfakeSrc");
        const audioBlobUrl = audioElement ? audioElement.querySelector("source").src : null;
        const audioName = audioBlobUrl ? `audio_${Date.now()}` : "";

        if (audioBlobUrl) {
            fetch(audioBlobUrl)
            .then((res) => res.blob())
            .then((blob) => {
              var file = new File([blob], audioName);
              uploadFile(file, audioName);
            });
        }

        return { audioName, audioBlobUrl };
    }

    function gatherSynthesisSettings(elem, mediaType, mediaName, mediaStart, mediaEnd, audioName, selectedFaceData) {
        if (!mediaName) {
            displayStatusMessage(elem, "Ensure media was loaded");
            return null;
        }
        if (!audioName) {
            displayStatusMessage(elem, "Ensure audio was loaded");
            return null;
        }
        if (!selectedFaceData) {
            displayStatusMessage(elem, "Ensure face selection is set");
            return null;
        }

        const preprocessingType = getSelectedPreprocessingType(elem);
        const advancedSettings = getAdvancedSettings(elem);

        return {
            face_fields: selectedFaceData,
            source_media: mediaName,
            driven_audio: audioName,
            preprocess: preprocessingType,
            still: elem.querySelector("#still-deepfake").checked,
            type_file: mediaType,
            media_start: mediaStart,
            media_end: mediaEnd,
            ...advancedSettings
        };
    }

    function getSelectedPreprocessingType(elem) {
        if (elem.querySelector("#resize-deepfake").checked) return "resize";
        return "full";
    }

    function getAdvancedSettings(elem) {
        return {
            expression_scale: elem.querySelector("#expression-scale-deepfake").value,
            input_yaw: elem.querySelector("#input-yaw-deepfake").value,
            input_pitch: elem.querySelector("#input-pitch-deepfake").value,
            input_roll: elem.querySelector("#input-roll-deepfake").value,
            emotion_label: getSelectedEmotionLabel(elem),
            similar_coeff: elem.querySelector("#similar-coeff-face").value
        };
    }

    function getSelectedEmotionLabel(elem) {
        const emotionSelect = elem.querySelector("#emotion-fake");
        const selectedValue = emotionSelect.options[emotionSelect.selectedIndex].value;
        return selectedValue === "null" ? null : selectedValue;
    }

    async function displayStatusMessage(elem, message) {
        const statusMessage = elem.querySelector("#message-about-status");
        statusMessage.innerText =  await translateWithGoogle(message,"auto",targetLang);
        statusMessage.style.display = "flex";
        statusMessage.style.background = getRandomColor();
    }

    function triggerSynthesisAPI(settings) {
        fetch("/synthesize_animation_talk/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings)
        });
    }
}

// ANIMATE WINDOWS //
function initiateFaceAndMouthPop(button, audio_url = undefined, audio_name = undefined) {
  var introFaceAndMouth = introJs();
  introFaceAndMouth.setOptions({
    steps: [
      {
        title: "Face and lip animation panel",
        position: "left",
        intro: `
                    <div id="main-windows-face-and-mouth-animation" style="width: 80vw; max-width: 90vw; height: 80vh; max-height: 90vh; columns: 2;display: flex;flex-direction: row;justify-content: space-around;">
                    <div id="previewDiv" style="display: flex;flex-direction: column;overflow: auto;justify-content: center;width: 100%;">
                        <div id="divGeneralPreviewMediaAndAudio" style="height: 100%;">
                            <span id="spanLoadMediaElement" class="dragBox" style="width: 100%;display: flex;text-align: center;margin-bottom: 15px;flex-direction: column;position: relative;height: 100%;justify-content: center;">
                                  Load image or video
                                <input accept="image/*,video/*" type="file" onChange="handleFaceAndMouthAnimation(event, document.getElementById('preview-media')); document.getElementById('divGeneralPreviewMediaAndAudio').style.height = '';" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" />
                            </span>
                            <p id="message-about-status" style="text-align: center;color: #393939;height: 30px;display: none;justify-content: center;align-items: center;padding: 5px;margin-bottom: 15px;"></p>
                            <div id="preview-media" style="position: relative;max-width: 60vw; max-height:60vh;display: flex;flex-direction: column;align-items: center;">
                            </div>
                        </div>

                        <div style="margin-top: 10pt;margin-bottom: 10pt;display: flex;">
                            <label id="uploadAudioDeepfakeLabel" for="uploadAudioDeepfake" class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Load audio</label>
                            <input style="width: 0;" accept=".mp3,.wav,.ogg,.flac" type="file" onChange="dragDropAudioDeepfakeFaceAnimation(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadAudioDeepfake"  />
                            <div id="previewDeepfakeAudio"></div>
                        </div>
                    </div>
                    <div id="face-animation-parameters-windows" style="display: none;align-items: stretch;flex-direction: column;justify-content: center;width: 25vw;">
                        <fieldset id="fieldset-preprocessing" style="padding: 5pt;">
                            <legend>Processing mode</legend>
                            <div>
                              <input type="radio" id="resize-deepfake" name="preprocessing_deepfake" value="resize">
                              <label for="resize-deepfake">Resize</label>
                            </div>
                            <div>
                              <input type="radio" id="full-deepfake" name="preprocessing_deepfake" value="full" checked>
                              <label for="full-deepfake">Without changes</label>
                            </div>
                        </fieldset>
                        <div id="still-deepfake-div" style="padding: 5pt;margin-top:5pt;">
                          <input type="checkbox" id="still-deepfake" name="still">
                          <label for="still-deepfake">Disable head movement</label>
                        </div>
                        <div id="similar-coeff-face-div" style="justify-content: space-between;padding: 5pt; display: flex;">
                          <label for="similar-coeff-face">Coefficient facial similarity</label>
                          <input type="number" id="similar-coeff-face" name="similar-coeff" min="0.1" max="3" step="0.1" value="1.2" style="border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 60pt;">
                        </div>
                        <fieldset style="margin-top:10pt;padding: 5pt;border-color: rgb(255 255 255 / 0%);box-shadow: none;">
                          <legend>
                            <button
                                style="background: none; border: none; font-size: 12pt; cursor: pointer; text-decoration: none;"
                                onclick="
                                    var advancedSettings = document.getElementById('advanced-settings');
                                    var parentLegend = this.parentElement.parentElement;
                                    advancedSettings.style.display = (advancedSettings.style.display === 'none') ? 'block' : 'none';
                                    parentLegend.style.borderColor = (parentLegend.style.borderColor === 'rgb(192, 192, 192)') ? 'rgb(255, 255, 255, 0)' : 'rgb(192, 192, 192)';
                                    parentLegend.style.boxShadow = (parentLegend.style.boxShadow === 'none' || parentLegend.style.boxShadow === '') ? 'rgba(0, 0, 0, 0.4) 0px 2px 4px, rgba(0, 0, 0, 0.3) 0px 7px 13px -3px, rgba(0, 0, 0, 0.2) 0px -3px 0px inset' : 'none';
                                ">
                                Advanced settings
                            </button>
                          </legend>
                          <div id="advanced-settings" style="display:none;">
                            <div id="expression-scale-deepfake-div" style="justify-content: space-between;padding: 5pt; display: flex;">
                              <label for="expression-scale-deepfake">Facial expressiveness</label>
                              <input type="number" id="expression-scale-deepfake" name="expression-scale" min="0.5" max="1.5" step="0.05" value="1.0" style="border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 30pt;">
                            </div>
                            <div id="input-yaw-deepfake-div" style="padding: 5pt;">
                              <label for="input-yaw-deepfake">Rotation angle XY</label>
                              <input type="text" pattern="[0-9,]+" oninput="this.value = this.value.replace(/[^0-9,-]/g, '');" title="Enter numbers separated by commas" id="input-yaw-deepfake" name="input-yaw" style="width: 100%;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;">
                            </div>
                            <div id="input-pitch-deepfake-div" style="padding: 5pt;">
                              <label for="input-pitch-deepfake">Rotation angle YZ</label>
                              <input type="text" pattern="[0-9,]+" oninput="this.value = this.value.replace(/[^0-9,-]/g, '');" title="Enter numbers separated by commas" id="input-pitch-deepfake" name="input-pitch" style="width: 100%;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;">
                            </div>
                            <div id="input-roll-deepfake-div" style="padding: 5pt;">
                              <label for="input-roll-deepfake">Rotation angle ZX</label>
                              <input type="text" pattern="[0-9,]+" oninput="this.value = this.value.replace(/[^0-9,-]/g, '');" title="Enter numbers separated by commas" id="input-roll-deepfake" name="input-roll" style="width: 100%;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;">
                            </div>
                            <div style="padding: 5pt;" id="use-experimental-functions-message">
                                <input onclick="document.getElementById('deepfake-emotion').style.display = this.checked ? 'block' : 'none';" type="checkbox" id="use-experimental-functions" name="experimental-functions">
                                <label for="use-experimental-functions">Experimental feature</label>
                                <div id="deepfake-emotion" style="margin-top: 10pt; margin-bottom: 10pt;display: none;">
                                    <label for="emotion-fake">Select emotion</label>
                                    <select id="emotion-fake" style="margin-left: 0;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 100%;margin-top: 5pt;">
                                        <option value="null" selected>Not use</option>
                                        <option value="0">Angry</option>
                                        <option value="1">Disgust</option>
                                        <option value="2">Fear</option>
                                        <option value="3">Happy</option>
                                        <option value="4">Neutral</option>
                                        <option value="5">Sad</option>
                                    </select>
                                    <p style="font-size: 10pt;"><b>Note:</b> <a>This is under research and is provided for display purposes only.</a></p>
                                </div>
                            </div>
                            <a style="padding: 5pt;" href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki" target="_blank" rel="noopener noreferrer">More about settings</a>
                          </div>
                        </fieldset>

                        <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="triggerFaceAndMouthSynthesis(this.parentElement.parentElement);">Start processing</button>
                    </div>
                    </div>
                    `,
      },
    ],
    showButtons: false,
    showStepNumbers: false,
    showBullets: false,
    nextLabel: "Next",
    prevLabel: "Back",
    doneLabel: "Close",
  });
  introFaceAndMouth.setOption('keyboardNavigation', false).start();
  // IF USER ADD AUDIO FROM SYNTHESIS PANEL //
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
        var audioPreview = document.getElementById("previewDeepfakeAudio");
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

        audio.onloadedmetadata = function () {
            var audioLength = document.getElementById("audio-length");
            audioLength.innerText = audio.duration.toFixed(1); // rounded to 1 decimal place
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
}

// UPDATE PREVIEW //
async function handleFaceAndMouthAnimation(event, previewElement) {
    const messageAboutStatus = document.getElementById("message-about-status");
    let messageAboutStatusText = "";
    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        const fileUrl = window.URL.createObjectURL(file);
        const fileType = file.type.split('/')[0];
        previewElement.innerHTML = "";
        document.getElementById("previewDiv").style.width = "";
        document.getElementById("spanLoadMediaElement").style.height = "30px";

        let canvas;
        if (fileType === 'image') {
            messageAboutStatus.style.display = "flex";
            messageAboutStatus.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Choose a face to animate by tool","auto",targetLang);
            messageAboutStatus.innerHTML = `${messageAboutStatusText} <i class="fa-solid fa-draw-polygon" style="margin-left: 10px;"></i>`;
            canvas = await setupImageCanvas(previewElement, fileUrl, "55vh", "45vw");

            document.getElementById("fieldset-preprocessing").style.display = "block";
            document.getElementById("still-deepfake-div").style.display = "block";
            document.getElementById("expression-scale-deepfake-div").style.display = "block";
            document.getElementById("input-yaw-deepfake-div").style.display = "block";
            document.getElementById("input-pitch-deepfake-div").style.display = "block";
            document.getElementById("input-roll-deepfake-div").style.display = "block";

            document.getElementById("similar-coeff-face-div").style.display = "none";
            document.getElementById("use-experimental-functions-message").style.display = "none";

            document.getElementById("face-animation-parameters-windows").style.display = "flex";
        } else if (fileType === 'video') {
            document.getElementById("fieldset-preprocessing").style.display = "none";
            document.getElementById("still-deepfake-div").style.display = "none";
            document.getElementById("expression-scale-deepfake-div").style.display = "none";
            document.getElementById("input-yaw-deepfake-div").style.display = "none";
            document.getElementById("input-pitch-deepfake-div").style.display = "none";
            document.getElementById("input-roll-deepfake-div").style.display = "none";

            document.getElementById("similar-coeff-face-div").style.display = "block";
            document.getElementById("use-experimental-functions-message").style.display = "block";

            messageAboutStatus.style.display = "flex";
            messageAboutStatus.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Video is loading...","auto",targetLang);
            messageAboutStatus.innerHTML = `${messageAboutStatusText}`;
            canvas = await setupVideoTimeline(previewElement, fileUrl, "55vh", "45vw");

            document.getElementById("face-animation-parameters-windows").style.display = "flex";
            messageAboutStatus.style.background = getRandomColor();
            messageAboutStatusText = await translateWithGoogle("Choose a face to animate by tool","auto",targetLang);
            messageAboutStatus.innerHTML = `${messageAboutStatusText} <i class="fa-solid fa-draw-polygon" style="margin-left: 10px;"></i>`;
        }
        canvas.addEventListener('click', setPointOnCanvas);
    }
}

async function dragDropAudioDeepfakeFaceAnimation(event) {
    if (event.target.files.length === 0) {
        console.warn("No files selected");
        return;
    }

    const file = URL.createObjectURL(event.target.files[0]);

    // Set audio name and length to the label
    const audioLabel = document.getElementById("uploadAudioDeepfakeLabel");
    const filename = event.target.files[0].name;
    const truncatedFilename = filename.length > 20 ? filename.slice(0, 17) + "..." : filename;

    // Get audio length
    const audioElement = new Audio(file);
    audioElement.onloadedmetadata = async function () {
        const messageAboutStatus = document.getElementById("message-about-status");
        let messageAboutStatusText = "";

        const formattedDuration = formatTime(audioElement.duration);
        audioLabel.innerText = `${truncatedFilename} (${formattedDuration})`;

        // Update audio preview
        const preview = document.getElementById("previewDeepfakeAudio");
        preview.innerHTML = `
            <button id="audioDeepfakePlay" class="introjs-button" style="display:inline;margin-left: 5pt;">
                <i class="fa fa-play"></i>
                <i style="display: none;" class="fa fa-pause"></i>
            </button>
            <audio id="audioDeepfakeSrc" style="display:none;" controls preload="none">
                <source src="${file}">
                Your browser does not support audio.
            </audio>`;

        const playBtn = document.getElementById("audioDeepfakePlay");
        const audio = document.getElementById("audioDeepfakeSrc");

        playBtn.addEventListener("click", async function () {
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

        audio.addEventListener("ended", async function () {
            playBtn.children[0].style.display = "inline";
            playBtn.children[1].style.display = "none";
        });

        messageAboutStatus.style.display = "flex";
        messageAboutStatus.style.background = getRandomColor();
        messageAboutStatusText = await translateWithGoogle("Audio was loaded","auto",targetLang);
        messageAboutStatus.innerHTML = `${messageAboutStatusText}`;
    };
}
// UPDATE PREVIEW //

function drag(elem) {
    // Change the color of the text to black when dragging over
    elem.style.fontSize = "18px";

    // Add dragleave and dragend event listeners
    elem.addEventListener("dragleave", handleDragLeaveOrEnd);
    elem.addEventListener("dragend", handleDragLeaveOrEnd);
}

function drop(elem) {
    // Reset text color after drop
    elem.style.fontSize = "";
}

// Function to handle when drag leaves target or drag ends without dropping
function handleDragLeaveOrEnd(event) {
    // Reset text color
    event.currentTarget.style.fontSize = "";

    // Remove these listeners as they're no longer necessary
    event.currentTarget.removeEventListener("dragleave", handleDragLeaveOrEnd);
    event.currentTarget.removeEventListener("dragend", handleDragLeaveOrEnd);
}

// ANIMATE WINDOWS //
