// RETOUCH //
function initiateRetouchAiPop(button) {
  var introRetouch = introJs();
  introRetouch.setOptions({
    steps: [
      {
        title: "Панель удаления объектов и ретуши",
        position: "right",
        intro: `
        <div style="width: 80vw; max-width: 90vw; height: 80vh; max-height: 90vh;display: flex;flex-direction: column;">
            <div id="div-general-upper"  style="display: flex;flex-direction: row;justify-content: space-around;height: 100%;">
                <div id="div-general-preview-media" style="width: 100%;">
                    <span class="dragBox" style="margin-bottom: 15px;display: flex;text-align: center;flex-direction: column;position: relative;justify-content: center;height: 100%;">
                          Загрузите изображение или видео
                        <input accept="image/*,video/*" type="file" onChange="handleRetouchAi(event, document.getElementById('preview-media'), this.parentElement);" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" />
                    </span>
                    <p id="message-about-status" style="text-align: center;color: #393939;height: 30px;display: none;justify-content: center;align-items: center;padding: 5px;margin-bottom: 15px;"></p>
                    <div id="preview-media" style="position: relative;max-width: 60vw; max-height:70vh;display: flex;flex-direction: column;align-items: center;">
                    </div>
                </div>
                <div id="div-control-panel" style="display: none;">
                    <fieldset style="padding: 5px;">
                        <legend>Режим</legend>
                        <div style="display: flex;flex-direction: column;justify-content: center;">
                            <div>
                                <input type="radio" id="retouch-automatic" name="retouch-mode" value="automatic" checked>
                                <label for="retouch-automatic">Автоматический режим</label>
                                <input style="margin-left: 25px;" type="radio" id="retouch-manually" name="retouch-mode" value="manually">
                                <label for="retouch-manually">Ручной режим</label>
                            </div>
                            <div style="justify-content: center;display: flex;">
                                <img id="preview-mask" style="display: none;max-width: 30vw;max-height:25vh;overflow: auto;margin-top: 25px;object-fit: contain;">
                            </div>
                        </div>
                    </fieldset>
                    <fieldset style="padding: 5px;display: flex; flex-direction: column;">
                        <legend>Маски</legend>
                        <button class="introjs-button" onclick="maskToList();">Добавить новый объект</button>
                        <div id="mask-timelines" style="overflow-y: auto;height: 20vh;"></div>
                    </fieldset>
                    <fieldset style="padding: 5pt;">
                        <legend>Выбор препроцессинга</legend>
                        <div>
                          <input type="radio" id="retouch-object" name="preprocessing_deepfake" value="resize">
                          <label for="retouch-object">Исправить окружение</label>
                        </div>
                        <div>
                          <input type="radio" id="retouch-face" name="preprocessing_deepfake" value="full" checked>
                          <label for="retouch-face">Исправить лицо</label>
                        </div>
                    </fieldset>
                    <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Обработать</button>
                </div>
            </div>
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
  introRetouch.start();
}

async function handleRetouchAi(event, previewElement, parentElement) {
    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        document.getElementById('div-general-preview-media').style.width = '';
        document.getElementById('div-general-upper').style.height = '';
        document.getElementById('div-control-panel').style.display = '';
        const previewMask = document.getElementById('preview-mask');

        const fileUrl = window.URL.createObjectURL(file);
        const fileType = file.type.split('/')[0];
        parentElement.style.height = "30px";
        previewElement.innerHTML = "";
        const messageElement = document.getElementById("message-about-status");

        let canvas;
        if (fileType === 'image') {
            displayMessage(messageElement, "Choose a point to get field by tool", '<i class="fa-solid fa-draw-polygon" style="margin-left: 10px;"></i>');
            // TODO Сообщение взависимости от режима, ручной или автоматический
            canvas = await setupImageCanvas(previewElement, fileUrl, "60vh", "45vw");

            const imagePreview = previewElement.getElementsByClassName("imageMedia")[0];
            previewMask.src = imagePreview.src;
        } else if (fileType === 'video') {
            displayMessage(messageElement, "Video is loading...");
            canvas = await setupVideoTimeline(previewElement, fileUrl, "60vh", "45vw");

            // TODO Сообщение взависимости от режима, ручной или автоматический
            displayMessage(messageElement, "Choose a point to get field by tool", '<i class="fa-solid fa-draw-polygon" style="margin-left: 10px;"></i>');

            const videoPreview = previewElement.getElementsByClassName("videoMedia")[0];

            previewMask.src = captureFrame(videoPreview);
        }

        previewMask.style.display = "";
        canvas.addEventListener('contextmenu', function(event) {
            event.preventDefault();
        });
        canvas.addEventListener('mousedown', setMultiplePointsForRetouchPreviewMask);
    }
}

function captureFrame(videoElem) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = videoElem.videoWidth;
    canvas.height = videoElem.videoHeight;
    ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);
    const imageUrl = canvas.toDataURL('image/jpeg');
    return imageUrl;
}

function setMultiplePointsForRetouchPreviewMask(event) {
    if (document.getElementById("retouch-automatic").checked) {
        triggerSendSegmentationDataMaskPreview(event, this)
    }
}

function retrieveMediaDetailsFramePreviewMask(mediaPreview) {
    const imageElements = mediaPreview.querySelectorAll(".imageMedia");
    const videoElements = mediaPreview.querySelectorAll(".videoMedia");
    const previewMask = document.getElementById("preview-mask")
    let mediaType = "";
    let mediaName = "";
    let mediaBlobUrl = "";
    let mediaCurrentTime = 0;

    if (imageElements.length > 0) {
        mediaType = "img";
        mediaBlobUrl = imageElements[0].src;
        mediaName = `image_${Date.now()}_${getRandomString(5)}`;
        mediaCurrentTime = 0;
    } else if (videoElements.length > 0) {
        mediaType = "video";
        // Ensure the video is paused at the frame you want to capture
        videoElements[0].pause();
        previewMask.src = captureFrame(videoElements[0])
        mediaBlobUrl = captureFrame(videoElements[0]);
        mediaName = `video_frame_${Date.now()}_${getRandomString(5)}`;
        mediaCurrentTime = videoElements[0].currentTime;
    }

    if (mediaBlobUrl) {
        fetch(mediaBlobUrl)
        .then((res) => res.blob())
        .then((blob) => {
          var file = new File([blob], mediaName);
          uploadFile(file);
        });
    }

    previewMask.setAttribute("current", mediaCurrentTime);

    return { mediaType, mediaName, mediaBlobUrl, mediaCurrentTime };
}

function triggerSendSegmentationDataMaskPreview(event, canvas) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => {
            if (data.status_code === 200) {
                setMultiplePointsOnCanvas(event);
                const objId = 1; // TODO will be get from img preview mask

                const pointsList = retrieveSelectedPointsList(canvas);
                const {
                    mediaType,
                    mediaName,
                    mediaBlobUrl,
                    mediaCurrentTime
                } = retrieveMediaDetailsFramePreviewMask(document.getElementById("preview-media"));

                sendSegmentationDataMaskPreview(mediaName, pointsList, mediaCurrentTime, objId);
            } else {
                displayMessage(document.getElementById("message-about-status"), "GPU process is busy...");
            }
        })
        .catch(error => {
            console.error("Error fetching the synthesis process status:", error);
        });
}


function sendSegmentationDataMaskPreview(mediaName, pointsList, mediaCurrentTime, objId) {
    const endpointUrl = "/create_segment_anything/";
    const payload = {
        source: mediaName,
        point_list: pointsList,
        current_time: mediaCurrentTime,
        obj_id: objId
    };
    // message for user
    displayMessage(document.getElementById("message-about-status"), "Waiting for model is loading for AI segmentation...");

    fetch(endpointUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 200) {
            // If the POST request was successful and the status is 200, fetch the segmentation data
            fetchSegmentAnythingAndSetCanvas();
        }
        console.log("Data successfully sent and received:", data);
    })
    .catch(error => {
        console.error("There was a problem with the fetch operation:", error);
    });
}



function fetchSegmentAnythingAndSetCanvas() {
    fetch("/get_segment_anything/")
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const segmentPreview = data.response;

            // Assuming there's only one current_time entry and one obj_id.
            // If there can be multiple, you'll need to loop through them.
            const currentTime = Object.keys(segmentPreview)[0];
            const objId = Object.keys(segmentPreview[currentTime])[0];
            const imageUrl = segmentPreview[currentTime][objId];
            const previewMask = document.querySelector('#preview-mask')

            // Create a canvas and set its source to the fetched image URL
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            const img = new Image();
            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                // copy style from original
                for (let prop in previewMask.style) {
                    if (previewMask.style.hasOwnProperty(prop)) {
                        canvas.style[prop] = previewMask.style[prop];
                    }
                }
                canvas.style.position = "absolute";
                ctx.drawImage(img, 0, 0);
            };
            img.src = imageUrl;

            // Append the canvas to previewMask
            const parentDiv = previewMask.parentNode;
            // Remove all prev canvas elements
            const allCanvases = parentDiv.querySelectorAll("canvas");
            allCanvases.forEach(canvas => canvas.remove());

            parentDiv.insertBefore(canvas, previewMask);
            displayMessage(document.getElementById("message-about-status"), "Mask is set");
        })
        .catch(error => {
            console.error("There was a problem with the fetch operation:", error);
        });
}

async function maskToList() {
    const previewMask = document.querySelector('#preview-mask');
    const parentDiv = previewMask.parentNode;
    const originalCanvas = parentDiv.querySelector("canvas");
    if (!originalCanvas) {
        displayMessage(document.getElementById("message-about-status"), "Before adding a mask, you need to create a point.");
        return null;
    }

    // Clone the original canvas content into a new canvas
    const clonedCanvas = document.createElement('canvas');
    clonedCanvas.width = originalCanvas.width;
    clonedCanvas.height = originalCanvas.height;
    const context = clonedCanvas.getContext('2d');
    const dataUrl = originalCanvas.toDataURL();
    const img = new Image();
    img.src = dataUrl;
    img.onload = function() {
        context.drawImage(img, 0, 0);
    };

    // Copy the styles from previewMask to the new canvas
    for (let prop in previewMask.style) {
        if (previewMask.style.hasOwnProperty(prop)) {
            clonedCanvas.style[prop] = previewMask.style[prop];
        }
    }

    // Modify the size and position of the new canvas
    clonedCanvas.style.maxWidth = "10vw";
    clonedCanvas.style.maxHeight = "10vh";
    clonedCanvas.style.position = "absolute";

    // Create a new div, append the new canvas and previewMask to it
    const newDiv = document.createElement('div');
    newDiv.style.position = "relative";
    newDiv.appendChild(clonedCanvas);
    // Adjust the size of the appended previewMask
    const clonedPreviewMask = previewMask.cloneNode(true);
    clonedPreviewMask.style.maxWidth = "10vw";
    clonedPreviewMask.style.maxHeight = "10vh";

    newDiv.appendChild(clonedPreviewMask);

    // Append the new div to maskTimelines
    const maskTimelines = document.getElementById("mask-timelines");
    maskTimelines.appendChild(newDiv);

    // Remove the original canvas
    originalCanvas.remove();

    // Clear general canvas
    const canvasGeneralElement = document.querySelectorAll(".canvasMedia")[0];
    const ctx = canvasGeneralElement.getContext('2d');
    if (canvasGeneralElement.hasAttribute("data-point-position")) {
        // For one point
        canvasGeneralElement.removeAttribute("data-point-position");
    }
    if (canvasGeneralElement.hasAttribute("data-points-list")) {
        // For one point
        canvasGeneralElement.removeAttribute("data-points-list");
    }
    // Clear the canvas
    ctx.clearRect(0, 0, canvasGeneralElement.width, canvasGeneralElement.height);
}
