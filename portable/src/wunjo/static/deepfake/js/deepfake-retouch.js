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
                        <p id="message-retouch" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                        <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="">Начать обрабатывать</button>
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
          savedRetouchMask.style.height = maxPreviewSide * 0.9 + 'pt';
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
          savedRetouchMask.style.height = (maxPreviewSide * 1.1) / aspectRatio + 'pt';
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
    devImgElement.style.display = 'flex';
    devImgElement.style.flexDirection = 'row';
    devImgElement.style.alignItems = 'center';
    // Get the base64 image data from the canvas
    // TODO Keep real size width, height in attributres
    const base64Image = canvasField.toDataURL("image/png");

    // Create an image element
    const imgElement = document.createElement('img');

    // Set the source of the image element to the base64 image data
    imgElement.src = base64Image;

    // Optionally, set some styles or attributes
    imgElement.style.width = parseFloat(preview.style.width) / 4 + 'px'; // Example width
    imgElement.style.height = parseFloat(preview.style.height) / 4 + 'px'; // Example height

    // Append the image element to the savedRetouchMask div
    devImgElement.appendChild(imgElement);
    devImgElement.innerHTML += `
        <div>
            <label>Старт</label>
            <input type="number" title="Введите число" min="0" max="0" step="0.1" value="0">
        </div>
        <div>
            <label>Окончание</label>
            <input type="number" title="Введите число" min="0" max="0" step="0.1" value="0">
        </div>
        <button class="introjs-button"><i class="fa-solid fa-trash"></i></button>
    `;

    savedRetouchMask.appendChild(devImgElement);
}
