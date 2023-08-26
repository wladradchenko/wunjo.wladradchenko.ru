function sendDataToDeepfake(elem) {
    // If process is free
    fetch('/synthesize_process/')
        .then(response => response.json())
        .then(data => {
            // Call the async function
            processAsyncDeepfake(data, elem).then(() => {
                console.log("Start to fetch msg for deepfake");
            }).catch((error) => {
                console.log("Error to fetch msg for deepfake");
                console.log(error);
            });
        });
}


async function processAsyncDeepfake(data, elem) {
    if (data.status_code === 200) {
        var synthesisDeepfakeTable = document.getElementById("table_body_deepfake_result");

        var messageDeepfake = elem.querySelector("#message-deepfake");
        messageDeepfake.innerHTML = "";

        var previewDeepfakeImg = elem.querySelector("#previewDeepfakeImg");

        var canvasRectangles = previewDeepfakeImg.querySelector('#canvasDeepfake');
        var canvasRectanglesList = [];
        if (canvasRectangles) {
            canvasRectanglesList = JSON.parse(canvasRectangles.dataset.rectangles);
        }

        var imgDeepfakeSrc = previewDeepfakeImg.querySelector('img');
        var videoDeepfakeSrc = previewDeepfakeImg.querySelector('video');
        var mediaName = "";
        var mediaBlobUrl = "";
        var typeFile = "";

        if (imgDeepfakeSrc) {
           typeFile = "img";
           mediaBlobUrl = imgDeepfakeSrc.src
           mediaName = "image_" + Date.now();
        } else if (videoDeepfakeSrc) {
          typeFile = "video";
          mediaBlobUrl = videoDeepfakeSrc.src
          mediaName = "video_" + Date.now();
        } else {
          var messageSetP = await translateWithGoogle("Вы не загрузили изображение. Нажмите на окно загрузки изображения.", 'auto', targetLang);
          messageDeepfake.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
        }
        if (mediaBlobUrl) {
            fetch(mediaBlobUrl)
                .then(res => res.blob())
                .then(blob => {
                    var file = new File([blob], mediaName);
                    uploadFile(file);
                });
        }

       var audioDeepfakeSrc = elem.querySelector("#audioDeepfakeSrc");
       var audioName = "";
       if (audioDeepfakeSrc) {
           var audioBlobUrl = audioDeepfakeSrc.querySelector("source").src;
           audioName = "audio_" + Date.now();
           fetch(audioBlobUrl)
                .then(res => res.blob())
                .then(blob => {
                    var file = new File([blob], audioName);
                    uploadFile(file);
                });
       } else {
          var messageSetP = await translateWithGoogle("Вы не загрузили аудиофайл. Нажмите на кнопку загрузить аудиофайл.", 'auto', targetLang);
          messageDeepfake.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
        }

        var cover = elem.querySelector("#cover-deepfake");
        var resize = elem.querySelector("#resize-deepfake");
        var full = elem.querySelector("#full-deepfake");
        var preprocessing = "full";
        if (cover.checked) {
            preprocessing = "cover"
        } else if (resize.checked) {
            preprocessing = "resize"
        }

        var still = elem.querySelector("#still-deepfake");
        var enhancer = elem.querySelector("#enhancer-deepfake");
        if (enhancer.checked) {
            enhancer = "gfpgan";
        } else {
            enhancer = false;  // TODO need to set false (not RestoreFormer)
        }

        if (canvasRectanglesList.length === 0) {
            messageDeepfake.innerHTML += "<p style='margin-top: 5pt;'>Вы не выделили лицо. Нажмите на кнопку выделить лицо и выделите лицо на изображении.</p>";
        }

        // advanced settings
        var expressionScaleDeepfake = elem.querySelector('#expression-scale-deepfake');
        var inputYawDeepfake = elem.querySelector('#input-yaw-deepfake');
        var inputPitchDeepfake = elem.querySelector('#input-pitch-deepfake');
        var inputRollDeepfake = elem.querySelector('#input-roll-deepfake');
        var backgroundEnhancerDeepfake = elem.querySelector('#background-enhancer-deepfake');
        var videoStartValue = elem.querySelector('#video-start').value;

        if (mediaName && audioName && canvasRectanglesList.length > 0) {
            // Get a reference to the #status-message element
            const statusMessage = document.getElementById('status-message');

            const buttonAnimationWindows = document.querySelector('#button-show-voice-window');
            buttonAnimationWindows.click();

            var predictParametersDeepfake = {
                "face_fields": canvasRectanglesList,
                "source_image": mediaName,
                "driven_audio": audioName,
                "preprocess": preprocessing,
                "still": still.checked,
                "enhancer": enhancer,
                "expression_scale": expressionScaleDeepfake.value,
                "input_yaw": inputYawDeepfake.value,
                "input_pitch": inputPitchDeepfake.value,
                "input_roll": inputRollDeepfake.value,
                "background_enhancer": backgroundEnhancerDeepfake.checked,
                "type_file": typeFile,
                "video_start": videoStartValue,
            };

            synthesisDeepfakeTable.innerHTML = "";
            statusMessage.innerText = await translateWithGoogle("Подождите... Происходит обработка", 'auto', targetLang);

            fetch("/synthesize_deepfake/", {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(predictParametersDeepfake)
            })

            const closeIntroButton = document.querySelector('.introjs-skipbutton');
            closeIntroButton.click();
        }
      } else {
        var synthesisDeepfakeTable = document.getElementById("table_body_deepfake_result");

        var messageDeepfake = elem.querySelector("#message-deepfake");
        var messageSetP = await translateWithGoogle("Процесс занят. Дождитесь его окончания.", 'auto', targetLang);
        messageDeepfake.innerHTML = `<p style='margin-top: 5pt;'>${messageSetP}</p>`;
      }
}


function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload_tmp_deepfake', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Upload failed');
    }
    console.log('File uploaded');
  })
  .catch(error => {
    console.error(error);
  });
}

// ANIMATE WINDOWS //
function deepfakeGeneralPop(button, audio_url = undefined, audio_name = undefined) {
    var audioInputField = `
                          <div class="uploadOuterDeepfakeAudio" style="margin-top: 10pt;margin-bottom: 10pt;display: flex;">
                            <label id="uploadAudioDeepfakeLabel" for="uploadAudioDeepfake" class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Загрузить аудио</label>
                            <input style="width: 0;" accept="audio/*" type="file" onChange="dragDropAudio(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadAudioDeepfake"  />
                            <div id="previewDeepfakeAudio"></div>
                          </div>
                         `;

    if (audio_url) {
      var request = new XMLHttpRequest();
      request.open('GET', audio_url, true);
      request.responseType = 'blob';
      request.onload = function() {
        var audioInputLabel = document.getElementById('uploadAudioDeepfakeLabel');
        audioInputLabel.textContent = audio_name.length > 20 ? audio_name.slice(0, 20) + "..." : audio_name;

        var audioInputButton = document.getElementById('uploadAudioDeepfake');
        audioInputButton.disabled = true;

        var audioBlobMedia = URL.createObjectURL(request.response);
        var audioPreview = document.getElementById('previewDeepfakeAudio');
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
        audio.onloadedmetadata = function() {
            // Set audio length on the text element
            var audioLength = document.getElementById("audio-length");
            audioLength.innerText = audio.duration.toFixed(1);  // rounded to 2 decimal places
        };

        playBtn.addEventListener("click", function() {
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

        audio.addEventListener("ended", function() {
          playBtn.children[0].style.display = "inline";
          playBtn.children[1].style.display = "none";
        });
      };
      request.send();
    }


    var introDeepfake = introJs();
    introDeepfake.setOptions({
        steps: [
            {
                element: button,
                title: 'Панель анимации',
                position: 'left',
                intro: `
                    <div style="width: 450pt;columns: 2;display: flex;flex-direction: row;justify-content: space-around;">
                    <div style="width: 200pt;">
                        <div class="uploadOuterDeepfake">
                            <div style="flex-direction: row;display: flex;margin-bottom: 10pt;justify-content: space-between;">
                                <button class="introjs-button" style="display: none;margin-right: 5pt;" id="clearButton">Очистить</button>
                                <!--<button class="introjs-button" style="display: none; margin-left: 5pt;margin-right: 5pt;" id="undoButton"><i class="fa fa-solid fa-reply"></i></button>-->
                                <button style="width: 100%;display: none;" class="introjs-button" id="drawButton" data-controlval="get-face">Выделить лицо</button>
                            </div>
                            <span id="previewDeepfakeImg" class="dragBox" style="height: 200pt;justify-content: center;">
                              Загрузить изображение или видео
                            <input accept="image/*,video/*" type="file" onChange="dragDropImg(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadFileDeepfake"  />
                            </span>
                        </div>

                        <fieldset id="fieldset-control-duration" style="display: none; padding: 5pt;margin-top: 10pt; ">
                            <legend></legend>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;">
                              <label for="video-start">Старт видео (сек) </label>
                              <input type="number" title="Введите число" id="video-start" name="expression-scale" min="0" max="0" step="0.1" value="0" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 60pt;">
                            </div>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;"><text>Длительность аудио (сек) </text><text id="audio-length">0</text></div>
                            <div style="justify-content: space-between; margin-top: 5pt; margin-bottom: 5pt; display: flex;"><text>Длительность видео (сек) </text><text id="video-length">0</text></div>
                        </fieldset>

                        ${audioInputField}
                    </div>
                    <div style="width: 200pt;">
                        <fieldset style="padding: 5pt;">
                            <legend>Выбор препроцессинга</legend>
                            <div>
                              <input type="radio" id="cover-deepfake" name="preprocessing_deepfake" value="cover">
                              <label for="cover-deepfake">Обрезать</label>
                            </div>
                            <div>
                              <input type="radio" id="resize-deepfake" name="preprocessing_deepfake" value="resize">
                              <label for="resize-deepfake">Изменить размер</label>
                            </div>
                            <div>
                              <input type="radio" id="full-deepfake" name="preprocessing_deepfake" value="full" checked>
                              <label for="full-deepfake">Без изменений</label>
                            </div>
                        </fieldset>
                        <div style="padding: 5pt;">
                          <input type="checkbox" id="still-deepfake" name="still">
                          <label for="still-deepfake">Отключить движение головой</label>
                        </div>
                        <div style="padding: 5pt;">
                          <input type="checkbox" id="enhancer-deepfake" name="enhancer" checked>
                          <label for="enhancer-deepfake">Улучшение лица</label>
                        </div>
                        <fieldset style="margin-top:10pt;padding: 5pt;border-color: rgb(255 255 255 / 0%);">
                          <legend><button style="background: none;border: none;font-size: 12pt;cursor: pointer;text-decoration" onclick="document.getElementById('advanced-settings').style.display = (document.getElementById('advanced-settings').style.display === 'none') ? 'block' : 'none';this.parentElement.parentElement.style.borderColor = (this.parentElement.parentElement.style.borderColor === 'rgb(192, 192, 192)') ? 'rgb(255 255 255 / 0%)' : 'rgb(192, 192, 192)';">Продвинутые настройки</button></legend>
                          <div id="advanced-settings" style="display:none;">
                            <div style="justify-content: space-between;padding: 5pt; display: flex;">
                              <label for="expression-scale-deepfake">Выраженность мимики</label>
                              <input type="number" title="Введите число" id="expression-scale-deepfake" name="expression-scale" min="0.5" max="1.5" step="0.05" value="1.0" style="border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;width: 30pt;">
                            </div>
                            <div style="padding: 5pt;">
                              <label for="input-yaw-deepfake">Угол поворота по XY</label>
                              <input type="text" pattern="[0-9,]+" oninput="this.value = this.value.replace(/[^0-9,-]/g, '');" title="Введите числа через запятую" id="input-yaw-deepfake" name="input-yaw" style="width: 100%;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;">
                            </div>
                            <div style="padding: 5pt;">
                              <label for="input-pitch-deepfake">Угол поворота по YZ</label>
                              <input type="text" pattern="[0-9,]+" oninput="this.value = this.value.replace(/[^0-9,-]/g, '');" title="Введите числа через запятую" id="input-pitch-deepfake" name="input-pitch" style="width: 100%;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;">
                            </div>
                            <div style="padding: 5pt;">
                              <label for="input-roll-deepfake">Угол поворота по ZX</label>
                              <input type="text" pattern="[0-9,]+" oninput="this.value = this.value.replace(/[^0-9,-]/g, '');" title="Введите числа через запятую" id="input-roll-deepfake" name="input-roll" style="width: 100%;border-width: 2px;border-style: groove;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;">
                            </div>
                            <div style="padding: 5pt;" id="background-enhancer-deepfake-message">
                              <input type="checkbox" id="background-enhancer-deepfake" name="background-enhancer">
                              <label for="background-enhancer-deepfake">Улучшение фона (долго)</label>
                            </div>
                            <a style="padding: 5pt;" href="https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki" target="_blank" rel="noopener noreferrer">Подробнее о настройках</a>
                          </div>
                        </fieldset>
                        <p id="message-deepfake" style="color: red;margin-top: 5pt;text-align: center;font-size: 14px;"></p>
                        <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="sendDataToDeepfake(this.parentElement.parentElement);">Синтезировать видео</button>
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
    introDeepfake.start();
    currentProcessor(document.getElementById("background-enhancer-deepfake-message"));
    document.getElementById("video-start").addEventListener("change", function() {
        var videoElement = document.querySelector("#previewDeepfakeImg video"); // get the video element inside the preview
        if (videoElement) {
            var startTime = parseFloat(this.value); // get the value of the input and convert it to a float
            videoElement.currentTime = startTime; // set the video's current playback time to the start time
        }
    });
};

function dragDropImg(event) {
  var file = event.target.files[0];
  // Getting the video length

  var reader = new FileReader();
  reader.onload = function(e) {
    var preview = document.getElementById("previewDeepfakeImg");

    if (file.type.includes('image')) {
      // Get field element
      var fieldsetControl = document.getElementById("fieldset-control-duration");
      fieldsetControl.style.display = "none";

      var videoLength = document.getElementById("video-length");
      videoLength.innerText = 0;  // this is img
      var videoInputLength = document.getElementById("video-start");
      videoInputLength.setAttribute('max', '0');
      videoInputLength.value = 0;
      var previewImg = document.createElement("img");
      previewImg.setAttribute("src", e.target.result);
      previewImg.setAttribute('width', '100%');
      previewImg.setAttribute('height', '100%');
      previewImg.style.objectFit = 'cover';
      preview.innerHTML = `<canvas style="position: absolute;" id="canvasDeepfake"></canvas>`;
      preview.appendChild(previewImg)
    } else if (file.type.includes('video')) {
      var video = document.createElement('video');
      video.onloadedmetadata = function() {
        // Get field element
        var fieldsetControl = document.getElementById("fieldset-control-duration");
        var audioLength = document.getElementById("audio-length").innerText;
        if (audioLength !== '0'){
           fieldsetControl.style.display = "block";
        } else {
           fieldsetControl.style.display = "none";
        }

        var videoLength = document.getElementById("video-length");
        videoLength.innerText = video.duration.toFixed(1);
        var videoInputLength = document.getElementById("video-start");
        var videoMaxLength;
        if (video.duration.toFixed(1) > 0.1){
            videoMaxLength = video.duration.toFixed(1) - 0.1;
        } else {
            videoMaxLength = video.duration.toFixed(1);
        };
        videoInputLength.setAttribute('max', videoMaxLength.toString());
        videoInputLength.value = 0;

      };
      video.setAttribute('src', e.target.result);
      video.setAttribute('width', '100%');
      video.setAttribute('height', '100%');
      video.setAttribute('preload', 'metadata');
      video.style.objectFit = 'cover';
      preview.innerHTML = '<canvas style="position: absolute;"  id="canvasDeepfake"></canvas>';
      preview.appendChild(video);
    }
    preview.innerHTML += '<input accept="image/*,video/*" type="file" onChange="dragDropImg(event)" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadFileDeepfake"/>';

    // DRAW RECTANGLES //
    var canvasDeepfake = document.getElementById('canvasDeepfake');
    var clearButton = document.getElementById('clearButton');
    //var undoButton = document.getElementById('undoButton'); // undo
    var drawButton = document.getElementById('drawButton');
    drawButton.style.display = 'inline';
    var previewDeepfakeImg = document.getElementById('previewDeepfakeImg');
    var uploadFileDeepfake = document.getElementById('uploadFileDeepfake');

    // Set canvas width and height to match image or video size
    canvasDeepfake.width = previewDeepfakeImg.clientWidth;
    canvasDeepfake.height = previewDeepfakeImg.clientHeight;
    var canvasWidth = canvasDeepfake.width
    var canvasHeight = canvasDeepfake.height

    const ctx = canvasDeepfake.getContext('2d');
    let rects = [];

    canvasDeepfake.dataset.rectangles = JSON.stringify(rects);

    let handleMouseDown;
    let handleMouseMove;
    let handleMouseUp;

    // Render the current rectangles
    function render() {
      ctx.clearRect(0, 0, canvasDeepfake.width, canvasDeepfake.height);
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
      const rect = canvasDeepfake.getBoundingClientRect();
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
            canvasDeepfake.dataset.rectangles = JSON.stringify(rects);
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

      canvasDeepfake.addEventListener('mousedown', handleMouseDown);
      canvasDeepfake.addEventListener('mousemove', handleMouseMove);
      canvasDeepfake.addEventListener('mouseup', handleMouseUp);
    }


    function turnOffDrawMode() {
      canvasDeepfake.removeEventListener('mousedown', handleMouseDown);
      canvasDeepfake.removeEventListener('mousemove', handleMouseMove);
      canvasDeepfake.removeEventListener('mouseup', handleMouseUp);
    }

    drawButton.onclick = function() {
      // data-controlval="get-face"
      if (drawButton.getAttribute("data-controlval") === 'get-face') {
        drawButton.setAttribute("data-controlval", "put-content");
        drawButton.textContent = 'Выбор файла';
        turnOnDrawMode();
        uploadFileDeepfake.disabled = true;
        canvasDeepfake.style.zIndex = 20;
        clearButton.style.display = 'inline';
        //undoButton.style.display = 'inline';  // undo
      } else {
        drawButton.setAttribute("data-controlval", "get-face");
        drawButton.textContent = 'Выделить лицо';
        turnOffDrawMode();
        uploadFileDeepfake.disabled = false;
        canvasDeepfake.style.zIndex = 0;
        clearButton.style.display = 'none';
        //undoButton.style.display = 'none';  // undo
      }
    };
    // undo
    //undoButton.onclick = function() {
    //  // Remove the last rectangle
    //  rects.pop();
    //  // Render the current rectangles
    //  render();
    //  canvasDeepfake.dataset.rectangles = JSON.stringify(rects);
    //};

    clearButton.onclick = function() {
      // Remove the last rectangle
      rects = [];
      // Render the current rectangles
      render();
      canvasDeepfake.dataset.rectangles = JSON.stringify(rects);
    };
    // DRAW RECTANGLES //
  };
  reader.readAsDataURL(file);
}

function dragDropAudio(event) {
  var file = URL.createObjectURL(event.target.files[0]);
  // Get audio length
  var audioElement = new Audio(file);
  audioElement.onloadedmetadata = function() {
     // Set audio length on the text element
     audioLength = document.getElementById("audio-length");
     audioLength.innerText = audioElement.duration.toFixed(1);
     // Get field element and control display
    var fieldsetControl = document.getElementById("fieldset-control-duration");
    var videoLength = document.getElementById("video-length").innerText;
    if (videoLength !== '0'){
       fieldsetControl.style.display = "block";
    } else {
       fieldsetControl.style.display = "none";
    }
  };
  var reader = new FileReader();
  var preview = document.getElementById("previewDeepfakeAudio");
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

  playBtn.addEventListener("click", function() {
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

  audio.addEventListener("ended", function() {
      playBtn.children[0].style.display = "inline";
      playBtn.children[1].style.display = "none";
  });
}

function drag(elem) {
    elem.parentNode.className = 'draging dragBox dragBoxMain';
    // Check if the element has the specific border style applied
    var dragBoxes = document.querySelectorAll(".dragBoxMain");
    dragBoxes.forEach(function(box) {
        box.style.border = "none";
    });
}

function drop(elem) {
    elem.parentNode.className = 'dragBox dragBoxMain';
}
// ANIMATE WINDOWS //