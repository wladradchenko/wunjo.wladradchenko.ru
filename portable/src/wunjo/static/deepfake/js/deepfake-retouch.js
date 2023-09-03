// RETOUCH //
function deepfakeRetouch(button) {
    var introRetouch = introJs();
    introRetouch.setOptions({
        steps: [
            {
                element: button,
                title: 'Панель удаления объектов и ретуши',
                position: 'left',
                intro: `
                <div style="min-width: 500px;" id="retouch">
                    <div class="uploadSourceRetouch">
                        <div style="flex-direction: row;display: flex;margin-bottom: 10pt;justify-content: space-between;">
                            <button class="introjs-button" style="display: none;margin-right: 5pt;" id="clearButtonRetouch">Очистить</button>
                            <button style="width: 100%;display: none;" class="introjs-button" id="drawButtonRetouch" data-controlval="get-face">Выделить</button>
                        </div>
                        <span id="previewRetouch" class="dragBox" style="height: 400pt;width: 400pt;justify-content: center;">
                          Загрузить целевое изображение или видео
                        <input accept="image/*,video/*" type="file" onChange="dragDropRetouch(event, 'previewRetouch', 'canvasRetouch', this, 'clearButtonRetouch', 'drawButtonRetouch');"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadSource"  />
                        </span>
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
};

function dragDropRetouch(event, previewId, canvasId, uploadFileElem, clearButtonId, drawButtonId) {
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

      if (file.type.includes('image')) {
        dimensions = await loadImage(e);
        var aspectRatio = dimensions.width / dimensions.height;
        if (dimensions.width >= dimensions.height) {
          console.log(12345)
          preview.style.width = maxPreviewSide + 'pt';
          preview.style.height = maxPreviewSide / aspectRatio + 'pt';
          dimensions.element.setAttribute('width', '100%');
          dimensions.element.setAttribute('height', 'auto');
        } else {
          preview.style.width = maxPreviewSide * aspectRatio + 'pt';
          preview.style.height = maxPreviewSide + 'pt';
          dimensions.element.setAttribute('width', 'auto');
          dimensions.element.setAttribute('height', '100%');
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
        } else {
          preview.style.width = maxPreviewSide * aspectRatio + 'pt';
          preview.style.height = maxPreviewSide + 'pt';
          dimensions.element.setAttribute('width', 'auto');
          dimensions.element.setAttribute('height', '100%');
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
    var previewDeepfakeImg = document.getElementById(previewId);
    var uploadFileDeepfake = document.getElementById(uploadFileId);

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

    // Render the current rectangles
    function render() {
      ctx.clearRect(0, 0, canvasField.width, canvasField.height);
      for (const rect of rects) {
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
      }
    }

    function turnOnDrawMode() {
      // Turn on drawing mode
      let isDrawing = false;
      let startX, startY, currentX, currentY;

      // Store the offset values
      const rect = canvasField.getBoundingClientRect();
      const offsetX = rect.left + window.scrollX;
      const offsetY = rect.top + window.scrollY;

      handleMouseDown = function(event) {
        // Start drawing a new rectangle
        isDrawing = true;
        startX = event.clientX - offsetX;
        startY = event.clientY - offsetY;
        currentX = startX;
        currentY = startY;
      }

      handleMouseMove = function(event) {
        if (isDrawing) {
          // Update the current rectangle
          currentX = event.clientX - offsetX;
          currentY = event.clientY - offsetY;
          render();
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
          ctx.lineWidth = 2;
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

      canvasField.addEventListener('mousedown', handleMouseDown);
      canvasField.addEventListener('mousemove', handleMouseMove);
      canvasField.addEventListener('mouseup', handleMouseUp);
    }


    function turnOffDrawMode() {
      canvasField.removeEventListener('mousedown', handleMouseDown);
      canvasField.removeEventListener('mousemove', handleMouseMove);
      canvasField.removeEventListener('mouseup', handleMouseUp);
    }

    drawButton.onclick = async function() {
      // data-controlval="get-face"
      if (drawButton.getAttribute("data-controlval") === 'get-face') {
        drawButton.setAttribute("data-controlval", "put-content");
        drawButton.textContent = await translateWithGoogle("Выбор файла", 'auto', targetLang);
        turnOnDrawMode();
        uploadFileDeepfake.disabled = true;
        canvasField.style.zIndex = 20;
        clearButton.style.display = 'inline';
      } else {
        drawButton.setAttribute("data-controlval", "get-face");
        drawButton.textContent = await translateWithGoogle("Выделить лицо", 'auto', targetLang);
        turnOffDrawMode();
        uploadFileDeepfake.disabled = false;
        canvasField.style.zIndex = 0;
        clearButton.style.display = 'none';
      }
    };

    clearButton.onclick = function() {
      // Remove the last rectangle
      rects = [];
      // Render the current rectangles
      render();
      canvasField.dataset.rectangles = JSON.stringify(rects);
    };
    // DRAW RECTANGLES //
  };
  reader.readAsDataURL(file);
}