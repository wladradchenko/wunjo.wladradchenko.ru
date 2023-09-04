// RETOUCH //
function deepfakeRetouch(button) {
    var introRetouch = introJs();
    introRetouch.setOptions({
        steps: [
            {
                element: document.getElementById('a-link-open-author'),
                title: 'Панель удаления объектов и ретуши',
                position: 'right',
                intro: `
                <div style="min-width: 700px;display: flex;flex-direction: row;" id="retouch">
                    <div class="uploadSourceRetouch">
                        <div style="flex-direction: row;display: flex;margin-bottom: 10pt;justify-content: space-between;">
                            <button class="introjs-button" style="display: none;margin-right: 5pt;" id="clearButtonRetouch">Очистить</button>
                            <button style="width: 100%;display: none;margin-right: 5pt;" class="introjs-button" id="drawButtonRetouch" data-controlval="get-face">Выделить</button>
                            <button style="width: 100%;display: none;" class="introjs-button" id="saveMaskRetouch" onclick="handleClickForSaveMaskRetouch('savedRetouchMask', 'canvasRetouch', 'previewRetouch')">Добавить</button>
                        </div>
                        <fieldset id="fieldsetLineWidthSlider" style="width: 100%; padding: 10pt;margin-bottom: 10pt;display: none;">
                            <legend>Толщина линии</legend>
                            <input style="width: 100%;" id="lineWidthSlider" class="range speech-train-split" type="range" min="1" max="100" step="1" value="5">
                            <div style="display: flex;justify-content: space-between;font-size: 10pt;color: #686868;">
                                <div>1</div>
                                <div>100</div>
                            </div>
                        </fieldset>
                        <span id="previewRetouch" class="dragBox" style="height: 450pt;width: 450pt;justify-content: center;">
                          Загрузить целевое изображение или видео
                        <input accept="image/*,video/*" type="file" onChange="dragDropRetouch(event, 'previewRetouch', 'canvasRetouch', this, 'clearButtonRetouch', 'drawButtonRetouch', 'lineWidthSlider', 'saveMaskRetouch');handleMetadataMedia(event, 'fieldsetSource', 'videoLength', 'videoFrame');document.getElementById('savedRetouchMask').innerHTML = '';"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadSource"  />
                        </span>
                        <fieldset id="fieldsetSource" style="display: none; padding: 5pt;margin-top: 10pt; ">
                            <legend></legend>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;">
                              <label for="videoFrame">Выбрать кадр </label>
                              <input type="number" title="Введите число" id="videoFrame" name="expression-scale" min="0" max="0" step="0.1" value="0" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 60pt;">
                            </div>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;"><text>Длительность видео (сек) </text><text id="videoLength">0</text></div>
                        </fieldset>
                    </div>
                    <div style="margin-left: 20pt;display: flex;flex-direction: column;justify-content: space-between;width: 300pt;">
                        <fieldset id="savedRetouchMaskFieldset" style="display:none; padding: 5pt;margin-top: 35pt;">
                            <legend>Маски</legend>
                            <div id="savedRetouchMask" style="display: flex;flex-direction: column;overflow-x: auto;max-height: 400pt;"></div>
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
                        <p id="message-retouch" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                        <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="sendDataToRetouch(this.parentElement.parentElement);">Начать обрабатывать</button>
                    </div>
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
    introRetouch.start();

    document.getElementById("videoFrame").addEventListener("change", function() {
        var videoElement = document.querySelector("#previewRetouch video"); // get the video element inside the preview
        if (videoElement) {
            var startTime = parseFloat(this.value); // get the value of the input and convert it to a float
            videoElement.currentTime = startTime; // set the video's current playback time to the start time
        }
    });
};

function dragDropRetouch(event, previewId, canvasId, uploadFileElem, clearButtonId, drawButtonId, lineWidthSliderId, saveMaskRetouchId) {
  var file = event.target.files[0];
  var uploadFileId = uploadFileElem.id;
  // Getting the video length

  var reader = new FileReader();
  reader.onload = async function(e) {
      let dimensions;
      var preview = document.getElementById(previewId);
      var widthPreview = parseFloat(preview.style.width);
      var heightPreview = parseFloat(preview.style.height);
      if (widthPreview > heightPreview) {
        var maxPreviewSide = widthPreview;
      } else {
        var maxPreviewSide = heightPreview;
      }
      var uploadFileElemOuterHTML = uploadFileElem.outerHTML;

      var savedRetouchMask = document.getElementById('savedRetouchMask')  // TODO

      if (file.type.includes('image')) {
        dimensions = await loadImage(e);
        var aspectRatio = dimensions.width / dimensions.height;
        if (dimensions.width >= dimensions.height) {
          preview.style.width = maxPreviewSide + 'pt';
          preview.style.height = maxPreviewSide / aspectRatio + 'pt';
          dimensions.element.setAttribute('width', '100%');
          dimensions.element.setAttribute('height', 'auto');
          savedRetouchMask.style.height = (maxPreviewSide * 0.9) / aspectRatio + 'pt';
        } else {
          preview.style.width = maxPreviewSide * aspectRatio + 'pt';
          preview.style.height = maxPreviewSide + 'pt';
          dimensions.element.setAttribute('width', 'auto');
          dimensions.element.setAttribute('height', '100%');
          savedRetouchMask.style.height = maxPreviewSide * 0.85 + 'pt';
        }
        dimensions.element.style.objectFit = 'cover';
        preview.innerHTML = `<canvas style="position: absolute;" id=${canvasId}></canvas>`;
        preview.appendChild(dimensions.element);
      } else if (file.type.includes('video')) {
        dimensions = await loadVideo(e);
        var aspectRatio = dimensions.width / dimensions.height;
        if (dimensions.width >= dimensions.height) {
          preview.style.width = maxPreviewSide + 'pt';
          preview.style.height = maxPreviewSide / aspectRatio + 'pt';
          dimensions.element.setAttribute('width', '100%');
          dimensions.element.setAttribute('height', 'auto');
          savedRetouchMask.style.height = (maxPreviewSide * 1.0) / aspectRatio + 'pt';
        } else {
          preview.style.width = maxPreviewSide * aspectRatio + 'pt';
          preview.style.height = maxPreviewSide + 'pt';
          dimensions.element.setAttribute('width', 'auto');
          dimensions.element.setAttribute('height', '100%');
          savedRetouchMask.style.height = maxPreviewSide * 0.9 + 'pt';
        }
        dimensions.element.setAttribute('preload', 'metadata');
        dimensions.element.style.objectFit = 'cover';
        preview.innerHTML = `<canvas style="position: absolute;"  id='${canvasId}'></canvas>`;
        preview.appendChild(dimensions.element);
      }
    preview.innerHTML += uploadFileElemOuterHTML;  // set prev parameters of upload input

    // DRAW RECTANGLES //
    var canvasField = document.getElementById(canvasId);
    var clearButton = document.getElementById(clearButtonId);
    var drawButton = document.getElementById(drawButtonId);
    drawButton.style.display = 'inline';
    var saveMaskRetouch = document.getElementById(saveMaskRetouchId);
    saveMaskRetouch.style.display = 'inline';
    var previewDeepfakeImg = document.getElementById(previewId);
    var uploadFileDeepfake = document.getElementById(uploadFileId);

    const lineWidthSlider = document.getElementById(lineWidthSliderId);
    document.getElementById('fieldsetLineWidthSlider').style.display = 'inline';
    document.getElementById('savedRetouchMaskFieldset').style.display = 'inline';
    lineWidthSlider.value = 2;

    // Set canvas width and height to match image or video size
    canvasField.width = previewDeepfakeImg.clientWidth;
    canvasField.height = previewDeepfakeImg.clientHeight;
    var canvasWidth = canvasField.width
    var canvasHeight = canvasField.height

    const ctx = canvasField.getContext('2d');
    let rects = [];

    canvasField.dataset.rectangles = JSON.stringify(rects);

    let handleMouseDown;
    let handleMouseMove;
    let handleMouseUp;

    let isFreeDrawing = false;  // Flag to check free drawing mode

    let lastX, lastY;  // Last mouse positions for free drawing

    function turnOnFreeDrawMode() {
      canvasField.addEventListener('mousedown', handleMouseDownFreeDraw);
      canvasField.addEventListener('mousemove', handleMouseMoveFreeDraw);
      canvasField.addEventListener('mouseup', handleMouseUpFreeDraw);
    }

    function turnOffFreeDrawMode() {
      canvasField.removeEventListener('mousedown', handleMouseDownFreeDraw);
      canvasField.removeEventListener('mousemove', handleMouseMoveFreeDraw);
      canvasField.removeEventListener('mouseup', handleMouseUpFreeDraw);
    }

    handleMouseDown = function(event) {
        // Start drawing a new rectangle
        isDrawing = true;
        startX = event.clientX - offsetX;
        startY = event.clientY - offsetY;
        currentX = startX;
        currentY = startY;
      }

      ctx.lineWidth = 2;
      lineWidthSlider.addEventListener('input', function() {
         ctx.lineWidth = this.value;
      });

      handleMouseMove = function(event) {
        if (isDrawing) {
          // Update the current rectangle
          currentX = event.clientX - offsetX;
          currentY = event.clientY - offsetY;
          render();
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
          ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
        } else {
            // record rects on value of button
            canvasField.dataset.rectangles = JSON.stringify(rects);
        }
      }

      handleMouseUp = function(event) {
            if (isDrawing) {
              // Add the new rectangle
              const x = Math.min(startX, currentX);
              const y = Math.min(startY, currentY);
              const width = Math.abs(currentX - startX);
              const height = Math.abs(currentY - startY);
              rects = [{x, y, width, height, canvasWidth, canvasHeight}];  // keep only one rectangles
              // rects.push({x, y, width, height});  // for multi rectangles
              // Render the current rectangles
              render();
              isDrawing = false;
            }
          }

    function handleMouseDownFreeDraw(event) {
      isFreeDrawing = true;
      const rect = canvasField.getBoundingClientRect();
      lastX = event.clientX - rect.left;
      lastY = event.clientY - rect.top;
    }

    function handleMouseMoveFreeDraw(event) {
      if (!isFreeDrawing) return;

      const rect = canvasField.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      ctx.strokeStyle = 'blue';
      ctx.lineJoin = "round";

      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(x, y);
      ctx.closePath();
      ctx.stroke();

      lastX = x;
      lastY = y;
    }

    function handleMouseUpFreeDraw(event) {
      isFreeDrawing = false;
    }

    // Event to toggle between rectangle drawing and free drawing
    drawButton.onclick = async function() {
      if (drawButton.getAttribute("data-controlval") === 'get-face') {
        drawButton.setAttribute("data-controlval", "put-content");
        drawButton.textContent = await translateWithGoogle("Выбор файла", 'auto', targetLang);
        turnOnFreeDrawMode();
        uploadFileDeepfake.disabled = true;
        canvasField.style.zIndex = 20;
        clearButton.style.display = 'inline';
      } else {
        drawButton.setAttribute("data-controlval", "get-face");
        drawButton.textContent = await translateWithGoogle("Выделить", 'auto', targetLang);
        turnOffFreeDrawMode();
        uploadFileDeepfake.disabled = false;
        canvasField.style.zIndex = 0;
        clearButton.style.display = 'none';
      }
    };


    // Clear Button Logic
    clearButton.addEventListener('click', () => {
      rects = [];
      ctx.clearRect(0, 0, canvasField.width, canvasField.height);
    });
    // DRAW RECTANGLES //
  };
  reader.readAsDataURL(file);
}

function handleClickForSaveMaskRetouch(savedRetouchMaskId, canvasId, previewId) {
    const preview = document.getElementById(previewId)
    const savedRetouchMask = document.getElementById(savedRetouchMaskId)
    const canvasField = document.getElementById(canvasId)
    const devImgElement = document.createElement('div');
    devImgElement.className = 'retouch-masks';
    devImgElement.style.display = 'flex';
    devImgElement.style.flexDirection = 'row';
    devImgElement.style.alignItems = 'center';
    devImgElement.style.justifyContent = 'space-around';
    devImgElement.style.margin = '5pt';
    devImgElement.style.padding = '5pt';
    devImgElement.style.boxShadow = 'rgba(0, 0, 0, 0.24) 0px 3px 8px';
    // Get the base64 image data from the canvas
    const base64Image = canvasField.toDataURL("image/png");

    // Check if the canvas is blank
    const ctx = canvasField.getContext('2d');
    ctx.clearRect(0, 0, canvasField.width, canvasField.height);
    const blankCanvasData = canvasField.toDataURL("image/png");

    if (base64Image === blankCanvasData) {
        console.log("Canvas is blank, not appending.");
        return; // Exit the function if the canvas is blank
    }

    // Create an image element
    const imgElement = document.createElement('img');

    // Set the source of the image element to the base64 image data
    imgElement.src = base64Image;

    // Optionally, set some styles or attributes
    imgElement.style.width = parseFloat(preview.style.width) / 4 + 'px'; // Example width
    imgElement.style.height = parseFloat(preview.style.height) / 4 + 'px'; // Example height

    // Get start frame and last
    const startFrame = document.getElementById('videoFrame').value;
    const endFrame = document.getElementById('videoLength').innerText;

    // if image than only one mask else a lot
    // Get target content
    var imgRetouchSource = preview.querySelector('img');
    var videoRetouchSource = preview.querySelector('video');

    if (imgRetouchSource) {
       savedRetouchMask.innerHTML = '';
       imgElement.style.backgroundImage = 'url(' + imgRetouchSource.src + ')';
    } else {
       const frameDataURL = getVideoFrameAsDataURL(videoRetouchSource);
       imgElement.style.backgroundImage = 'url(' + frameDataURL + ')';
    }

    imgElement.style.backgroundSize = 'contain';
    imgElement.className = 'canvas-mask-set';

    // Append the image element to the savedRetouchMask div
    devImgElement.appendChild(imgElement);

    if ('0' !== endFrame) {
        devImgElement.innerHTML += `
            <div style="margin-left: 5pt;margin-right: 5pt;">
                <label>Начало</label>
                <input class="start-frame-mask" type="number" title="Введите число" min="0" max="${endFrame}" step="0.1" value="${startFrame}">
            </div>
            <div style="margin-left: 5pt;margin-right: 5pt;">
                <label>Окончание</label>
                <input class="end-frame-mask" type="number" title="Введите число" min="0" max="${endFrame}" step="0.1" value="${endFrame}">
            </div>
            <button class="introjs-button" onclick="this.parentElement.remove();"><i class="fa-solid fa-trash"></i></button>
        `;
    } else {
        devImgElement.innerHTML += `
            <button class="introjs-button" onclick="this.parentElement.remove();"><i class="fa-solid fa-trash"></i></button>
        `;
    }

    savedRetouchMask.appendChild(devImgElement);
}


function getVideoFrameAsDataURL(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL();
}

function sendDataToRetouch(elem) {
    // If process is free
    fetch('/synthesize_process/')
        .then(response => response.json())
        .then(data => {
            // Call the async function
            processAsyncRetouch(data, elem).then(() => {
                console.log("Start to fetch msg for deepfake");
            }).catch((error) => {
                console.log("Error to fetch msg for deepfake");
                console.log(error);
            });
        });
}

async function processAsyncRetouch(data, elem) {
    if (data.status_code === 200) {
        var synthesisDeepfakeTable = document.getElementById("table_body_deepfake_result");

        var messageRetouch = elem.querySelector("#message-retouch");
        messageRetouch.innerHTML = "";

        // Get target content
        var previewRetouch = elem.querySelector("#previewRetouch");

        // Get preprocessing
        var preprocessingRetouchObject = elem.querySelector("#retouch-object");
        var modelType = 'retouch_face';
        if (preprocessingRetouchObject.checked) {
            modelType = "retouch_object";
        }

        // Save in tmp content for source
        var imgRetouchMain = previewRetouch.querySelector('img');
        var videoRetouchMain = previewRetouch.querySelector('video');
        var mediaNameMain = "";
        var mediaBlobUrlMain= "";

        if (imgRetouchMain) {
           mediaBlobUrlMain = imgRetouchMain.src
           mediaNameMain = "image_target_" + Date.now() + "_" + getRandomString(5);
        } else if (videoRetouchMain) {
          mediaBlobUrlMain = videoRetouchMain.src
          mediaNameMain = "video_target_" + Date.now() + "_" + getRandomString(5);
        } else {
          var messageSetP = await translateWithGoogle("Вы не загрузили целевое изображение. Нажмите на окно загрузки изображения.", 'auto', targetLang);
          messageRetouch.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
        }
        if (mediaBlobUrlMain) {
            fetch(mediaBlobUrlMain)
                .then(res => res.blob())
                .then(blob => {
                    var file = new File([blob], mediaNameMain);
                    uploadFile(file);
                });
        }

        // Get all mask, save them and get those parameters
        var savedRetouchMask = elem.querySelector("#savedRetouchMask");
        var retouchMasks = savedRetouchMask.querySelectorAll(".retouch-masks");

        // Create an array to store the results
        var resultsRetouchMasks = [];

        // Iterate over each retouch mask
        retouchMasks.forEach(function(mask) {
            // Find the image element with class 'canvas-mask-set' inside the current mask
            var imgElementMask = mask.querySelector('.canvas-mask-set');
            var imgSrcMask = imgElementMask ? imgElementMask.src : null;  // Check if the imgElement exists, if not set the src as null

            // Generate name
            var mediaNameMask = "mask_" + Date.now() + "_" + getRandomString(5) + ".png";

            if (imgSrcMask) {
                fetch(imgSrcMask)
                    .then(res => res.blob())
                    .then(blob => {
                        var file = new File([blob], mediaNameMask);
                        uploadFile(file);
                    });
            }

            // Find the input element with class 'start-frame-mask' inside the current mask
            var startFrameInputMask = mask.querySelector('.start-frame-mask');
            var startFrameValueMask = startFrameInputMask ? startFrameInputMask.value : null; // Check if the startFrameInput exists, if not set the value as null

            // Find the input element with class 'end-frame-mask' inside the current mask
            var endFrameInputMask = mask.querySelector('.end-frame-mask');
            var endFrameValueMask = endFrameInputMask ? endFrameInputMask.value : null; // Check if the endFrameInput exists, if not set the value as null

            // Store the results in an object and push it to the results array
            resultsRetouchMasks.push({
                mediaNameMask: mediaNameMask,
                startFrameMask: startFrameValueMask,
                endFrameMask: endFrameValueMask
            });
        });

        if (resultsRetouchMasks.length < 1) {
            messageRetouch.innerHTML += "<p style='margin-top: 5pt;'>Write here msg.</p>";
            return
        } else {
            const buttonAnimationWindows = document.querySelector('#button-show-voice-window');
            buttonAnimationWindows.click();

            var predictParametersRetouch = {
                "source": mediaNameMain,
                "mask": resultsRetouchMasks,
                "model_type": modelType,
            };

            synthesisDeepfakeTable.innerHTML = "";

            fetch("/synthesize_retouch/", {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(predictParametersRetouch)
            })

            const closeIntroButton = document.querySelector('.introjs-skipbutton');
            closeIntroButton.click();
        }
    };
};