function sendDataToTalker(elem) {
    // If process is free
    fetch('/synthesize_process/')
        .then(response => response.json())
        .then(data => {
          // Check the value of status_code
          if (data.status_code === 200) {
            var synthesisTalkerTable = document.getElementById("table_body_talker_result");

            var previewTalkerImg = elem.querySelector("#previewTalkerImg");

            var canvasRectangles = previewTalkerImg.querySelector('#canvasTalker');
            var canvasRectanglesList = [];
            if (canvasRectangles) {
                canvasRectanglesList = JSON.parse(canvasRectangles.dataset.rectangles);
            }

            var imgTalkerSrc = previewTalkerImg.querySelector('img');
            var videoTalkerSrc = previewTalkerImg.querySelector('video');
            var mediaName = "";
            var mediaBlobUrl = "";

            if (imgTalkerSrc) {
               mediaBlobUrl = imgTalkerSrc.src
               mediaName = "image_" + Date.now();
            } else if (videoTalkerSrc) {
              mediaBlobUrl = videoTalkerSrc.src
              mediaName = "video_" + Date.now();
            }
            if (mediaBlobUrl) {
                fetch(mediaBlobUrl)
                    .then(res => res.blob())
                    .then(blob => {
                        var file = new File([blob], mediaName);
                        uploadFile(file);
                    });
            }

           var audioTalkerSrc = elem.querySelector("#audioTalkerSrc");
           var audioName = "";
           if (audioTalkerSrc) {
               var audioBlobUrl = audioTalkerSrc.querySelector("source").src;
               audioName = "audio_" + Date.now();
               fetch(audioBlobUrl)
                    .then(res => res.blob())
                    .then(blob => {
                        var file = new File([blob], audioName);
                        uploadFile(file);
                    });
           }

            var cover = elem.querySelector("#cover-talker");
            var resize = elem.querySelector("#resize-talker");
            var full = elem.querySelector("#full-talker");
            var preprocessing = "full";
            if (cover.checked) {
                preprocessing = "cover"
            } else if (resize.checked) {
                preprocessing = "resize"
            }

            var still = elem.querySelector("#still-talker");
            var enhancer = elem.querySelector("#enhancer-talker");
            if (enhancer.checked) {
                enhancer = "gfpgan";
            } else {
                enhancer = "RestoreFormer";
            }

            if (mediaName && audioName && canvasRectanglesList.length > 0) {
                // Get a reference to the #status-message element
                const statusMessage = document.getElementById('status-message');

                const buttonAnimationWindows = document.querySelector('#button-show-voice-window');
                buttonAnimationWindows.click();

                var predictParametersTalker = {
                    "face_fields": canvasRectanglesList,
                    "source_image": mediaName,
                    "driven_audio": audioName,
                    "preprocess": preprocessing,
                    "still": still.checked,
                    "enhancer": enhancer
                };

                synthesisTalkerTable.innerHTML = "";
                statusMessage.style.display = 'inline';

                fetch("/synthesize_talker/", {
                    method: "POST",
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(predictParametersTalker)
                })

                const closeIntroButton = document.querySelector('.introjs-skipbutton');
                closeIntroButton.click();
            }
          }
        });
}


function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload_tmp_talker', {
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
function talkerGeneralPop(button, audio_url = undefined, audio_name = undefined) {
    var audioInputField = `
                          <div class="uploadOuterTalkerAudio" style="margin-top: 10pt;margin-bottom: 10pt;display: flex;">
                            <label id="uploadAudioTalkerLabel" for="uploadAudioTalker" class="introjs-button" style="text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Загрузить аудио</label>
                            <input style="width: 0;" accept="audio/*" type="file" onChange="dragDropAudio(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadAudioTalker"  />
                            <div id="previewTalkerAudio"></div>
                          </div>
                         `;

    if (audio_url) {
      var request = new XMLHttpRequest();
      request.open('GET', audio_url, true);
      request.responseType = 'blob';
      request.onload = function() {
        var audioInputLabel = document.getElementById('uploadAudioTalkerLabel');
        audioInputLabel.textContent = audio_name.length > 20 ? audio_name.slice(0, 20) + "..." : audio_name;

        var audioInputButton = document.getElementById('uploadAudioTalker');
        audioInputButton.disabled = true;

        var audioBlobMedia = URL.createObjectURL(request.response);
        var audioPreview = document.getElementById('previewTalkerAudio');
        audioPreview.innerHTML = `
          <button id="audioTalkerPlay" class="introjs-button" style="display:inline;margin-left: 5pt;">
            <i class="fa fa-play"></i>
            <i style="display: none;" class="fa fa-pause"></i>
          </button>
          <audio id="audioTalkerSrc" style="display:none;" controls="" preload="none">
            <source src="${audioBlobMedia}">
            Your browser does not support audio.
          </audio>
        `;
        var playBtn = document.getElementById("audioTalkerPlay");
        var audio = document.getElementById("audioTalkerSrc");

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


    var introTalker = introJs();
    introTalker.setOptions({
        steps: [
            {
                element: button,
                title: 'Панель анимации',
                position: 'left',
                intro: `
                    <div style="width: 200pt;">
                        <div class="uploadOuterTalker">
                            <div style="flex-direction: row;display: flex;margin-bottom: 10pt;justify-content: space-between;">
                                <button class="introjs-button" style="display: none;margin-right: 5pt;" id="clearButton">Очистить</button>
                                <!--<button class="introjs-button" style="display: none; margin-left: 5pt;margin-right: 5pt;" id="undoButton"><i class="fa fa-solid fa-reply"></i></button>-->
                                <button style="width: 100%;display: none;" class="introjs-button" id="drawButton">Выделить лицо</button>
                            </div>
                            <span id="previewTalkerImg" class="dragBox" style="height: 200pt;">
                              Загрузить изображение или видео
                            <input accept="image/*,video/*" type="file" onChange="dragDropImg(event)"  ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" id="uploadFileTalker"  />
                            </span>
                        </div>
                        ${audioInputField}
                        <fieldset style="padding: 5pt;">
                            <legend>Выбор препроцессинга</legend>
                            <div>
                              <input type="radio" id="cover-talker" name="preprocessing_talker" value="cover">
                              <label for="cover-talker">Обрезать</label>
                            </div>
                            <div>
                              <input type="radio" id="resize-talker" name="preprocessing_talker" value="resize">
                              <label for="resize-talker">Изменить размер</label>
                            </div>
                            <div>
                              <input type="radio" id="full-talker" name="preprocessing_talker" value="full" checked>
                              <label for="full-talker">Без изменений</label>
                            </div>
                        </fieldset>
                        <div style="padding: 5pt;">
                          <input type="checkbox" id="still-talker" name="still">
                          <label for="still-talker">Отключить движение головой</label>
                        </div>
                        <div style="padding: 5pt;">
                          <input type="checkbox" id="enhancer-talker" name="enhancer" checked>
                          <label for="enhancer-talker">Улучшение лица</label>
                        </div>
                        <button class="introjs-button" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;" onclick="sendDataToTalker(this.parentElement);">Синтезировать видео</button>
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
    introTalker.start();
};

function dragDropImg(event) {
  var file = event.target.files[0];
  var reader = new FileReader();
  reader.onload = function(e) {
    var preview = document.getElementById("previewTalkerImg");
    if (file.type.includes('image')) {
      var previewImg = document.createElement("img");
      previewImg.setAttribute("src", e.target.result);
      previewImg.setAttribute('width', '100%');
      previewImg.setAttribute('height', '100%');
      previewImg.style.objectFit = 'cover';
      preview.innerHTML = `<canvas style="position: absolute;" id="canvasTalker"></canvas>`;
      preview.appendChild(previewImg)
    } else if (file.type.includes('video')) {
      var video = document.createElement('video');
      video.setAttribute('src', e.target.result);
      video.setAttribute('width', '100%');
      video.setAttribute('height', '100%');
      video.setAttribute('preload', 'metadata');
      video.style.objectFit = 'cover';
      preview.innerHTML = '<canvas style="position: absolute;"  id="canvasTalker"></canvas>';
      preview.appendChild(video);
    }
    preview.innerHTML += '<input accept="image/*,video/*" type="file" onChange="dragDropImg(event)" ondragover="drag()" ondrop="drop()" id="uploadFileTalker"/>';

    // DRAW RECTANGLES //
    var canvasTalker = document.getElementById('canvasTalker');
    var clearButton = document.getElementById('clearButton');
    //var undoButton = document.getElementById('undoButton'); // undo
    var drawButton = document.getElementById('drawButton');
    drawButton.style.display = 'inline';
    var previewTalkerImg = document.getElementById('previewTalkerImg');
    var uploadFileTalker = document.getElementById('uploadFileTalker');

    // Set canvas width and height to match image or video size
    canvasTalker.width = previewTalkerImg.clientWidth;
    canvasTalker.height = previewTalkerImg.clientHeight;
    var canvasWidth = canvasTalker.width
    var canvasHeight = canvasTalker.height

    const ctx = canvasTalker.getContext('2d');
    let rects = [];

    canvasTalker.dataset.rectangles = JSON.stringify(rects);

    let handleMouseDown;
    let handleMouseMove;
    let handleMouseUp;

    // Render the current rectangles
    function render() {
      ctx.clearRect(0, 0, canvasTalker.width, canvasTalker.height);
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
      const rect = canvasTalker.getBoundingClientRect();
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
            canvasTalker.dataset.rectangles = JSON.stringify(rects);
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

      canvasTalker.addEventListener('mousedown', handleMouseDown);
      canvasTalker.addEventListener('mousemove', handleMouseMove);
      canvasTalker.addEventListener('mouseup', handleMouseUp);
    }


    function turnOffDrawMode() {
      canvasTalker.removeEventListener('mousedown', handleMouseDown);
      canvasTalker.removeEventListener('mousemove', handleMouseMove);
      canvasTalker.removeEventListener('mouseup', handleMouseUp);
    }

    drawButton.onclick = function() {
      if (drawButton.textContent === 'Выделить лицо') {
        drawButton.textContent = 'Выбор файла';
        turnOnDrawMode();
        uploadFileTalker.disabled = true;
        canvasTalker.style.zIndex = 20;
        clearButton.style.display = 'inline';
        //undoButton.style.display = 'inline';  // undo
      } else {
        drawButton.textContent = 'Выделить лицо';
        turnOffDrawMode();
        uploadFileTalker.disabled = false;
        canvasTalker.style.zIndex = 0;
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
    //  canvasTalker.dataset.rectangles = JSON.stringify(rects);
    //};

    clearButton.onclick = function() {
      // Remove the last rectangle
      rects = [];
      // Render the current rectangles
      render();
      canvasTalker.dataset.rectangles = JSON.stringify(rects);
    };
    // DRAW RECTANGLES //
  };
  reader.readAsDataURL(file);
}
function dragDropAudio(event) {
  var file = URL.createObjectURL(event.target.files[0]);
  var reader = new FileReader();
  var preview = document.getElementById("previewTalkerAudio");
  preview.innerHTML = `<button id="audioTalkerPlay" class="introjs-button" style="display:inline;margin-left: 5pt;">
                          <i class="fa fa-play"></i>
                          <i style="display: none;" class="fa fa-pause"></i>
                      </button>
                      <audio id="audioTalkerSrc" style="display:none;" controls preload="none">
                        <source src="${file}">
                        Your browser does not support audio.
                      </audio>`;

  var playBtn = document.getElementById("audioTalkerPlay");
  var audio = document.getElementById("audioTalkerSrc");

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
    elem.parentNode.className = 'draging dragBox';
}
function drop(elem) {
    elem.parentNode.className = 'dragBox';
}
// ANIMATE WINDOWS //