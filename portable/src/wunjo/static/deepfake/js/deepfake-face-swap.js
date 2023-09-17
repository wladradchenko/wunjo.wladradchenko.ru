// FACE SWAP //
function deepfakeFaceSwap(button, audio_url = undefined, audio_name = undefined) {
    var introFaceSwap = introJs();
    introFaceSwap.setOptions({
        steps: [
            {
                element: button,
                title: 'Панель смены лица',
                position: 'right',
                intro: `
                    <div>
                        <fieldset style="padding: 5pt;margin: 13pt;margin-top: 0;flex-direction: row;display: flex;">
                            <legend>Настройки</legend>
                            <div>
                            <div style="padding: 5pt;">
                              <input type="checkbox" id="multiface-deepfake" name="multiface">
                              <label for="multiface-deepfake">Заменить все лица</label>
                            </div>
                            <div style="padding: 5pt;">
                              <input type="checkbox" id="similarface-deepfake" name="similarface">
                              <label for="similarface-deepfake">Несколько одинаковых лиц</label>
                            </div>
                            </div>
                            <div>
                            <div style="padding: 5pt;">
                              <input type="checkbox" id="enhancer-deepfake" name="enhancer">
                              <label for="enhancer-deepfake">Улучшение лица</label>
                            </div>
                            <div style="padding: 5pt;" id="background-enhancer-deepfake-message">
                              <input type="checkbox" id="background-enhancer-deepfake" name="background-enhancer">
                              <label for="background-enhancer-deepfake">Улучшение фона (долго)</label>
                            </div>
                            </div>
                        </fieldset>
                        <div>
                            <div style="padding: 5pt;margin-left: 7pt;">
                              <label for="similar-coeff-face">Коэффициент похожести лица</label>
                              <input type="number" title="Введите число" id="similar-coeff-face" name="similar-coeff" min="0.1" max="2" step="0.1" value="0.95" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 60pt;">
                            </div>
                        </div>
                    </div>
                    <div style="width: 450pt;columns: 2;display: flex;flex-direction: row;justify-content: space-around;">
                    <div style="width: 200pt;">
                        <div class="uploadTargetFile">
                            <div style="flex-direction: row;display: flex;margin-bottom: 10pt;justify-content: space-between;">
                                <button class="introjs-button" style="display: none;margin-right: 5pt;" id="clearButtonTarget">Очистить</button>
                                <button style="width: 100%;display: none;" class="introjs-button" id="drawButtonTarget" data-controlval="get-face">Выделить лицо для замены</button>
                            </div>
                            <span id="previewFaceSwapTarget" class="dragBox" style="height: 200pt;justify-content: center;">
                              Загрузить целевое изображение или видео
                            <input accept="image/*,video/*" type="file" onChange="dragDropImgOrVideo(event, 'previewFaceSwapTarget', 'canvasTarget', this, 'clearButtonTarget', 'drawButtonTarget');handleMetadataMedia(event, 'fieldsetTarget', 'videoLengthTarget', 'videoStartTarget');"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadTarget"  />
                            </span>
                        </div>

                        <fieldset id="fieldsetTarget" style="display: none; padding: 5pt;margin-top: 10pt; ">
                            <legend></legend>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;">
                              <label for="videoStartTarget">Старт видео (сек) </label>
                              <input type="number" title="Введите число" id="videoStartTarget" name="expression-scale" min="0" max="0" step="0.1" value="0" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 60pt;">
                            </div>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;"><text>Длительность видео (сек) </text><text id="videoLengthTarget">0</text></div>
                        </fieldset>
                    </div>
                    <div style="width: 200pt;">
                        <div class="uploadSourceFile">
                            <div style="flex-direction: row;display: flex;margin-bottom: 10pt;justify-content: space-between;">
                                <button class="introjs-button" style="display: none;margin-right: 5pt;" id="clearButtonSource">Очистить</button>
                                <button style="width: 100%;display: none;" class="introjs-button" id="drawButtonSource" data-controlval="get-face">Выделить лицо источник</button>
                            </div>
                            <span id="previewFaceSwapSource" class="dragBox" style="height: 200pt;justify-content: center;">
                              Загрузить исходное изображение или видео
                            <input accept="image/*,video/*" type="file" onChange="dragDropImgOrVideo(event, 'previewFaceSwapSource', 'canvasSource', this, 'clearButtonSource', 'drawButtonSource');handleMetadataMedia(event, 'fieldsetSource', 'videoLengthSource', 'videoStartSource');"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadSource"  />
                            </span>
                        </div>

                        <fieldset id="fieldsetSource" style="display: none; padding: 5pt;margin-top: 10pt; ">
                            <legend></legend>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;">
                              <label for="videoStartSource">Выбрать кадр </label>
                              <input type="number" title="Введите число" id="videoStartSource" name="expression-scale" min="0" max="0" step="0.1" value="0" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 60pt;">
                            </div>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: none;"><text></text><text id="videoLengthSource">0</text></div>
                        </fieldset>
                    </div>
                    </div>
                    <p id="message-face-swap" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                    <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="sendDataToFaceSwap(this.parentElement.parentElement);">Заменить лицо</button>
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
    introFaceSwap.start();
    availableFeaturesByCUDA(document.getElementById("background-enhancer-deepfake-message"));
    document.getElementById("videoStartTarget").addEventListener("change", function() {
        var videoElement = document.querySelector("#previewFaceSwapTarget video"); // get the video element inside the preview
        if (videoElement) {
            var startTime = parseFloat(this.value); // get the value of the input and convert it to a float
            videoElement.currentTime = startTime; // set the video's current playback time to the start time
        }
    });
    document.getElementById("videoStartSource").addEventListener("change", function() {
        var videoElement = document.querySelector("#previewFaceSwapSource video"); // get the video element inside the preview
        if (videoElement) {
            var startTime = parseFloat(this.value); // get the value of the input and convert it to a float
            videoElement.currentTime = startTime; // set the video's current playback time to the start time
        }
    });
};


function handleMetadataMedia(event, fieldsetControlId, videoLengthId, videoStartId) {
    const file = event.target.files[0];

    if (file.type.includes('image')) {
      handleImageMetadataMedia(fieldsetControlId,videoLengthId, videoStartId);
    } else if (file.type.includes('video')) {
      // You can use a Promise or setTimeout to wait until the metadata is loaded
      const video = document.createElement('video');
      video.setAttribute('src', URL.createObjectURL(file));
      video.onloadedmetadata = function() {
        handleVideoMetadataMedia(video, fieldsetControlId, videoLengthId, videoStartId);
      };
    }
}

function handleVideoMetadataMedia(video, fieldsetControlId, videoLengthId, videoStartId) {
    const fieldsetControl = document.getElementById(fieldsetControlId);
    fieldsetControl.style.display = "block";

    const videoLength = document.getElementById(videoLengthId);
    videoLength.innerText = video.duration.toFixed(1);

    const videoInputLength = document.getElementById(videoStartId);
    let videoMaxLength;
    if (video.duration.toFixed(1) > 0.1){
        videoMaxLength = video.duration.toFixed(1) - 0.1;
    } else {
        videoMaxLength = video.duration.toFixed(1);
    }
    videoInputLength.setAttribute('max', videoMaxLength.toString());
    videoInputLength.value = 0;
}

function handleImageMetadataMedia(fieldsetControlId, videoLengthId, videoStartId) {
    const fieldsetControl = document.getElementById(fieldsetControlId);
    fieldsetControl.style.display = "none";

    const videoLength = document.getElementById(videoLengthId);
    videoLength.innerText = 0;

    const videoInputLength = document.getElementById(videoStartId);
    videoInputLength.setAttribute('max', '0');
    videoInputLength.value = 0;
}

function sendDataToFaceSwap(elem) {
    // If process is free
    fetch('/synthesize_process/')
        .then(response => response.json())
        .then(data => {
            // Call the async function
            processAsyncFaceSwap(data, elem).then(() => {
                console.log("Start to fetch msg for deepfake");
            }).catch((error) => {
                console.log("Error to fetch msg for deepfake");
                console.log(error);
            });
        });
}

async function processAsyncFaceSwap(data, elem) {
    if (data.status_code === 200) {
        var synthesisDeepfakeTable = document.getElementById("table_body_deepfake_result");

        var messageFaceSwap = elem.querySelector("#message-face-swap");
        messageFaceSwap.innerHTML = "";

        // Get target content
        var previewFaceSwapTarget = elem.querySelector("#previewFaceSwapTarget");

        var canvasRectanglesTarget = previewFaceSwapTarget.querySelector('#canvasTarget');
        var canvasRectanglesListTarget = [];
        if (canvasRectanglesTarget) {
            canvasRectanglesListTarget = JSON.parse(canvasRectanglesTarget.dataset.rectangles);
        }

        var imgFaceSwapTarget = previewFaceSwapTarget.querySelector('img');
        var videoFaceSwapTarget = previewFaceSwapTarget.querySelector('video');
        var mediaNameTarget = "";
        var mediaBlobUrlTarget = "";
        var typeFileTarget = "";

        if (imgFaceSwapTarget) {
           typeFileTarget = "img";
           mediaBlobUrlTarget = imgFaceSwapTarget.src
           mediaNameTarget = "image_target_" + Date.now() + "_" + getRandomString(5);
        } else if (videoFaceSwapTarget) {
          typeFileTarget = "video";
          mediaBlobUrlTarget = videoFaceSwapTarget.src
          mediaNameTarget = "video_target_" + Date.now() + "_" + getRandomString(5);
        } else {
          var messageSetP = await translateWithGoogle("Вы не загрузили целевое изображение. Нажмите на окно загрузки изображения.", 'auto', targetLang);
          messageFaceSwap.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
        }
        if (mediaBlobUrlTarget) {
            fetch(mediaBlobUrlTarget)
                .then(res => res.blob())
                .then(blob => {
                    var file = new File([blob], mediaNameTarget);
                    uploadFile(file);
                });
        }

        // Get source content
        var previewFaceSwapSource = elem.querySelector("#previewFaceSwapSource");

        var canvasRectanglesSource = previewFaceSwapSource.querySelector('#canvasSource');
        var canvasRectanglesListSource = [];
        if (canvasRectanglesSource) {
            canvasRectanglesListSource = JSON.parse(canvasRectanglesSource.dataset.rectangles);
        }

        var imgFaceSwapSource = previewFaceSwapSource.querySelector('img');
        var videoFaceSwapSource = previewFaceSwapSource.querySelector('video');
        var mediaNameSource = "";
        var mediaBlobUrlSource = "";
        var typeFileSource = "";

        if (imgFaceSwapSource) {
           typeFileSource = "img";
           mediaBlobUrlSource = imgFaceSwapSource.src
           mediaNameSource = "image_source_" + Date.now() + "_" + getRandomString(5);
        } else if (videoFaceSwapSource) {
          typeFileSource = "video";
          mediaBlobUrlSource = videoFaceSwapSource.src
          mediaNameSource = "video_source_" + Date.now() + "_" + getRandomString(5);
        } else {
          var messageSetP = await translateWithGoogle("Вы не загрузили изображение для источника лица. Нажмите на окно загрузки изображения.", 'auto', targetLang);
          messageFaceSwap.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
        }
        if (mediaBlobUrlSource) {
            fetch(mediaBlobUrlSource)
                .then(res => res.blob())
                .then(blob => {
                    var file = new File([blob], mediaNameSource);
                    uploadFile(file);
                });
        }

        var multiface = elem.querySelector("#multiface-deepfake");
        var similarface = elem.querySelector("#similarface-deepfake");
        var enhancer = elem.querySelector("#enhancer-deepfake");
        if (enhancer.checked) {
            enhancer = "gfpgan";
        } else {
            enhancer = false;  // TODO need to set false (not RestoreFormer)
        }

        if (canvasRectanglesListTarget.length === 0) {
            messageFaceSwap.innerHTML += "<p style='margin-top: 5pt;'>Вы не выделили лицо. Нажмите на кнопку выделить лицо и выделите лицо на целевом контенте.</p>";
        }

        if (canvasRectanglesListSource.length === 0) {
            messageFaceSwap.innerHTML += "<p style='margin-top: 5pt;'>Вы не выделили лицо. Нажмите на кнопку выделить лицо и выделите лицо на источнике.</p>";
        }

        // advanced settings
        var backgroundEnhancerFaceSwap = elem.querySelector('#background-enhancer-deepfake');
        var videoStartValueTarget = elem.querySelector('#videoStartTarget').value;
        var videoStartValueSource = elem.querySelector('#videoStartSource').value;
        var similarCoeffFace = elem.querySelector('#similar-coeff-face').value;

        if (mediaNameTarget && mediaNameSource && canvasRectanglesListTarget.length > 0 && canvasRectanglesListSource.length > 0) {
            const buttonAnimationWindows = document.querySelector('#button-show-voice-window');
            buttonAnimationWindows.click();

            var predictParametersFaceSwap = {
                "face_target_fields": canvasRectanglesListTarget,
                "target_content": mediaNameTarget,
                "video_start_target": videoStartValueTarget,
                "type_file_target": typeFileTarget,
                "face_source_fields": canvasRectanglesListSource,
                "source_content": mediaNameSource,
                "video_start_source": videoStartValueSource,
                "type_file_source": typeFileSource,
                "multiface": multiface.checked,
                "similarface": similarface.checked,
                "enhancer": enhancer,
                "background_enhancer": backgroundEnhancerFaceSwap.checked,
                "similar_coeff": similarCoeffFace,
            };

            synthesisDeepfakeTable.innerHTML = "";

            fetch("/synthesize_face_swap/", {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(predictParametersFaceSwap)
            })

            const closeIntroButton = document.querySelector('.introjs-skipbutton');
            closeIntroButton.click();
        }
      } else {
        var synthesisDeepfakeTable = document.getElementById("table_body_deepfake_result");

        var messageFaceSwap = elem.querySelector("#message-face-swap");
        var messageSetP = await translateWithGoogle("Процесс занят. Дождитесь его окончания.", 'auto', targetLang);
        messageFaceSwap.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
      }
}