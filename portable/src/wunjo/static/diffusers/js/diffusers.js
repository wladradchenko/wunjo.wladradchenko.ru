// DIFFUSION ONLY FOR GPU
window.addEventListener('load', function() {
  const menubarGeneral = document.getElementById("menubar");

  // Create the 'a' element
  const buttonVideoToVideoDiffusion = document.createElement("a");
  buttonVideoToVideoDiffusion.id = "a-video-to-video-diffuser";
  buttonVideoToVideoDiffusion.style.color = "black";
  buttonVideoToVideoDiffusion.style.width = "3.2vw";
  buttonVideoToVideoDiffusion.style.height = "3.2vw";
  buttonVideoToVideoDiffusion.style.fontSize = "1.5rem";
  buttonVideoToVideoDiffusion.style.display = "none";
  buttonVideoToVideoDiffusion.title = "Open video to video conversion window by prompt";
  buttonVideoToVideoDiffusion.addEventListener("click", (event) =>
    initiateDiffusersPop(event.currentTarget)
  );

  // Create the 'i' element and append it to the 'a' element
  const icon = document.createElement("i");
  icon.className = "fa-solid fa-palette";
  buttonVideoToVideoDiffusion.appendChild(icon);

  // Append the 'a' element to the 'menubarGeneral'
  menubarGeneral.appendChild(buttonVideoToVideoDiffusion);
  document
    .getElementById("a-change-processor")
    .addEventListener("click", (event) => {
      availableFeaturesByCUDA(buttonVideoToVideoDiffusion);
    });
  availableFeaturesByCUDA(buttonVideoToVideoDiffusion);
});
// DIFFUSION ONLY FOR GPU


function initiateDiffusersPop(button) {
  var introRetouch = introJs();
  introRetouch.setOptions({
    steps: [
      {
        title: "Panel of video to video by prompt",
        position: "right",
        intro: `
        <div style="width: 80vw; max-width: 90vw; height: 80vh; max-height: 90vh;display: flex;flex-direction: column;">
            <div id="div-general-upper"  style="display: flex;flex-direction: row;justify-content: space-around;height: 100%;">
                <div id="div-general-preview-media" style="width: 100%;">
                    <span class="dragBox" style="margin-bottom: 15px;display: flex;text-align: center;flex-direction: column;position: relative;justify-content: center;height: 100%;">
                          Load image or video
                        <input accept="image/*,video/*" type="file" onChange="handleDiffuser(event, document.getElementById('preview-media'), this.parentElement);" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" />
                    </span>
                    <p id="message-about-status" style="text-align: center;color: #393939;height: 30px;display: none;justify-content: center;align-items: center;padding: 5px;margin-bottom: 15px;"></p>
                    <div id="preview-media" style="position: relative;max-width: 60vw; max-height:70vh;display: flex;flex-direction: column;align-items: center;">
                    </div>
                </div>
                <div id="div-control-panel" style="display: none;">
                    <div id="div-preview-mask" style="justify-content: center;display: flex;">
                    </div>
                    <fieldset style="padding: 5px;display: flex; flex-direction: column;margin-top: 30px;">
                        <legend>List of elements</legend>
                        <button class="introjs-button" onclick="maskDiffuserToList();">Add new element</button>
                        <div id="mask-timelines" style="overflow-y: auto;height: 20vh;"></div>
                    </fieldset>
                    <fieldset style="padding: 5pt;overflow-y: auto;max-height: 15vh;">
                        <div>
                            <select id="modelDiffusionDropdown" style="width: 100%;border: inset;border-color: rgb(192, 192, 192);background-color: #fff;padding: 1pt;margin: 0;margin-bottom: 10px;"></select>
                        </div>
                        <div>
                            <input type="checkbox" id="paintBackground" onclick="document.getElementById('paintBackgroundParam').style.display = this.checked ? 'block' : 'none';">
                            <label for="paintBackground">Change background</label>
                        </div>
                        <div id="paintBackgroundParam" style="display: none;">
                            <div class="diffuser-params" style="font-size: small;">
                                <div style="display: flex; flex-direction: row; justify-content: space-around; margin: 5px;">
                                    <div style="display: flex;">
                                        <label class="notranslate">Strength</label>
                                        <input id="diffusionBackgroundStrength" type="number" max="1" min="0.1" value="0.95" step="0.01" style="width: 50px; margin-left: 5px;">
                                    </div>
                                    <div>
                                        <label class="notranslate">Scale</label>
                                        <input id="diffusionBackgroundScale" type="number" max="10" min="1" value="7.5" step="0.5" style="width: 50px; margin-left: 5px;">
                                    </div>
                                    <div>
                                        <label class="notranslate">Seed</label>
                                        <input id="diffusionBackgroundSeed" type="number" max="1000" min="0" value="0" step="1" style="width: 50px; margin-left: 5px;">
                                    </div>
                                </div>
                            <textarea id="diffusionBackgroundPrompt" placeholder="Input your prompt to draw or input 'pass' to skip" style="min-height: 25px; max-height: 75px; max-width: 30vw; min-width: 30vw; padding: 5px; margin: 5px;"></textarea>
                            <textarea id="diffusionBackgroundNegativePrompt" placeholder="Input negative prompt" style="min-height: 25px; max-height: 75px; max-width: 30vw; min-width: 30vw; padding: 5px; margin: 5px;"></textarea>
                        </div>
                        </div>
                        <legend>General settings</legend>
                        <p style="margin-top:10px;" >Preprocessor</p>
                        <div style="display: flex;">
                            <div>
                                <input type="radio" id="diffuserLooseCfattn" name="translation" value="loose_cfattn" checked>
                                <label for="diffuserLooseCfattn" class="notranslate">Loose cfattn</label>
                            </div>
                            <div style="margin-left: 15px;">
                                <input type="radio" id="diffuserFreeu" name="translation" value="freeu">
                                <label for="diffuserFreeu" class="notranslate">Freeu</label>
                            </div>
                        </div>
                        <p style="margin-top:10px;" class="notranslate">ControlNet</p>
                        <div style="display: flex;">
                            <div>
                                <input type="radio" id="controlnetCanny" name="control_type" value="canny" checked>
                                <label for="controlnetCanny" class="notranslate">Canny</label>
                            </div>
                            <div style="margin-left: 15px;">
                                <input type="radio" id="controlnetHed" name="control_type" value="hed">
                                <label for="controlnetHed" class="notranslate">Hed</label>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <label for="intervalGeneration">Interval generation: </label>
                            <input type="number" id="intervalGeneration" name="interval" min="1" max="1000" value="20" style="width: 45px;">
                        </div>
                        <div style="margin-top: 15px;">
                            <label for="percentageInput">Matching of masks: </label>
                            <input type="number" id="percentageInput" name="percentage" min="1" max="50" value="25" style="width: 45px;"> %
                        </div>
                        <div style="margin-top: 15px;">
                            <label for="thicknessMask">Thickness mask: </label>
                            <input type="number" id="thicknessMask" name="thickness" min="1" max="100" value="10" style="width: 45px;">
                        </div>
                    </fieldset>
                    <button class="introjs-button" onclick="triggerDiffuser(this.parentElement.parentElement);" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Start processing</button>
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
  introRetouch.setOption('keyboardNavigation', false).start();

  const dropdownDiffusion = document.getElementById("modelDiffusionDropdown");
  const diffusionModelsListParsed = JSON.parse(diffusionModelsList);

  for (const key in diffusionModelsListParsed) {
     const option = document.createElement("option");
     option.text = key;
     option.value = diffusionModelsListParsed[key];
     dropdownDiffusion.add(option);
  }
}


async function maskDiffuserToList() {
    const previewMedia = document.getElementById("preview-media")
    const videoElements = previewMedia.querySelectorAll(".videoMedia");
    let endTime = 0;
    if (videoElements.length > 0) {
        endTime = videoElements[0].getAttribute("end");
    }

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
    clonedCanvas.style.boxShadow = "";

    const objId = originalCanvas.getAttribute("data-objId");
    const currentTime = originalCanvas.getAttribute("data-currentTime");
    const pointList = originalCanvas.getAttribute("data-pointList");

    clonedCanvas.setAttribute("data-objId", objId);
    clonedCanvas.setAttribute("data-currentTime", currentTime);
    clonedCanvas.setAttribute("data-pointList", pointList);  // Directly set the string value

    // Modify the size and position of the new canvas
    clonedCanvas.style.maxWidth = "10vw";
    clonedCanvas.style.maxHeight = "10vh";
    clonedCanvas.style.position = "absolute";

    // Create a new div, append the new canvas and previewMask to it
    const newDivGeneral = document.createElement('div');
    newDivGeneral.className = "mask-timeline";
    newDivGeneral.style = "display: flex;flex-direction: column;";
    const newDiv = document.createElement('div');
    newDiv.style = "display: flex;flex-direction: row;justify-content: space-around;align-items: center;";
    const newDivChild = document.createElement('div');
    newDivChild.style.position = "relative";
    newDivChild.appendChild(clonedCanvas);
    // Adjust the size of the appended previewMask
    const clonedPreviewMask = previewMask.cloneNode(true);
    clonedPreviewMask.style.maxWidth = "10vw";
    clonedPreviewMask.style.maxHeight = "10vh";
    clonedPreviewMask.style.boxShadow = "";

    newDivChild.appendChild(clonedPreviewMask);

    // Append the new div to maskTimelines
    const maskTimelines = document.getElementById("mask-timelines");
    newDiv.appendChild(newDivChild);

    const ulElem = document.createElement('ul');
    ulElem.style.fontSize = "small";
    ulElem.style.listStyleType = "none";
    ulElem.style.display = "flex";
    ulElem.style.flexDirection = "column";

    const liObjId = document.createElement('li');
    liObjId.innerText = "Id " + objId;
    retouchObjIdMask += 1;  // update global count for objId
    liObjId.style.margin = "5px";

    const timePattern = "^([0-1]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\\.\\d{1,3}$";

    const liCurrentTime = document.createElement('li');
    liCurrentTime.innerText = "Start: ";
    const currentTimeInput = document.createElement('input');
    currentTimeInput.value = formatTime(currentTime);
    currentTimeInput.setAttribute('pattern', timePattern);
    currentTimeInput.setAttribute('placeholder', '00:00:00.000');
    currentTimeInput.addEventListener('input', enforceTimeFormat);
    currentTimeInput.className = "mask-timeline-start-time";
    currentTimeInput.style.width = "90px";
    liCurrentTime.style.margin = "5px";
    liCurrentTime.appendChild(currentTimeInput);

    const liEndTime = document.createElement('li');
    liEndTime.innerText = "End: ";
    const endTimeInput = document.createElement('input');
    endTimeInput.value = formatTime(endTime);
    endTimeInput.setAttribute('pattern', timePattern);
    endTimeInput.setAttribute('placeholder', '00:00:00.000');
    endTimeInput.addEventListener('input', enforceTimeFormat);
    endTimeInput.className = "mask-timeline-end-time";
    endTimeInput.style.width = "90px";
    liEndTime.style.margin = "5px";
    liEndTime.appendChild(endTimeInput);

    ulElem.appendChild(liObjId);
    ulElem.appendChild(liCurrentTime);
    ulElem.appendChild(liEndTime);
    newDiv.appendChild(ulElem);

    function enforceTimeFormat(event) {
        const value = event.target.value;
        const validFormat = /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.\d{1,3})?$/;
        const interimFormat = /^([0-2]?[0-9]?:?[0-5]?[0-9]??:?[0-5]?[0-9]?(\.\d{0,3})?)?$/;

        // Ensure that only valid characters are entered
        if (!value.match(interimFormat)) {
            event.target.value = value.substring(0, value.length - 1);
        }

        // Check validity of the time format
        if (value.match(validFormat)) {
            // Convert time strings to Date objects for easy comparison
            const convertToTime = time => new Date(`1970-01-01T${time}Z`);
            const currentStartTime = convertToTime(currentTimeInput.value);
            const currentEndTime = convertToTime(endTimeInput.value);
            const maxEndTime = convertToTime(formatTime(endTime));

            if (event.target === currentTimeInput && (currentStartTime >= currentEndTime || currentStartTime > maxEndTime)) {
                event.target.style.borderColor = 'red';  // Invalid input
                // message for user
                displayMessage(document.getElementById("message-about-status"), "Invalid input, start time cannot be more than end time");
            } else if (event.target === endTimeInput && (currentEndTime <= currentStartTime || currentEndTime > maxEndTime)) {
                event.target.style.borderColor = 'red';  // Invalid input
                displayMessage(document.getElementById("message-about-status"), "Invalid input, end time cannot be less than start time and more than media duration");
            } else {
                event.target.style.borderColor = '';  // Valid input
                displayMessage(document.getElementById("message-about-status"), "Correct format");
            }
        } else {
            event.target.style.borderColor = 'red';  // Change border color to indicate invalid input
            displayMessage(document.getElementById("message-about-status"), "Invalid input, the correct format is 00:00:00.000");
        }
    }

    const removeBtn = document.createElement('button');
    removeBtn.className = "introjs-button";
    removeBtn.innerHTML = `<i class="fa-solid fa-eraser"></i>`;
    newDiv.appendChild(removeBtn);

    // Add diffusion additional params
    const diffuserParamsDiv = document.createElement('div');
    diffuserParamsDiv.className = 'diffuser-params';
    diffuserParamsDiv.style = "font-size: small;";
    // Params
    const diffuserParamsDivChild = document.createElement('div');
    diffuserParamsDivChild.style = "display: flex;flex-direction: row;justify-content: space-around;;margin: 5px;";

    const diffuserParamsDivChildStrength = document.createElement('div');
    diffuserParamsDivChildStrength.style = "display: flex;";
    const labelStrength = document.createElement('label');
    labelStrength.className = 'notranslate';
    labelStrength.innerText = "Strength";
    diffuserParamsDivChildStrength.appendChild(labelStrength);
    const inputStrength = document.createElement('input');
    inputStrength.className = "diffusionStrength";
    inputStrength.type = 'number';
    inputStrength.style = "width: 50px;margin-left: 5px;";
    inputStrength.setAttribute('max', '1');
    inputStrength.setAttribute('min', '0.1');
    inputStrength.setAttribute('value', '0.95');
    inputStrength.setAttribute('step', '0.01');
    diffuserParamsDivChildStrength.appendChild(inputStrength);
    diffuserParamsDivChild.appendChild(diffuserParamsDivChildStrength);

    const diffuserParamsDivChildScale = document.createElement('div');
    const labelScale = document.createElement('label');
    labelScale.className = 'notranslate';
    labelScale.innerText = "Scale";
    diffuserParamsDivChildScale.appendChild(labelScale);
    const inputScale = document.createElement('input');
    inputScale.className = "diffusionScale";
    inputScale.type = 'number';
    inputScale.style = "width: 50px;margin-left: 5px;";
    inputScale.setAttribute('max', '10');
    inputScale.setAttribute('min', '1');
    inputScale.setAttribute('value', '7.5');
    inputScale.setAttribute('step', '0.5');
    diffuserParamsDivChildScale.appendChild(inputScale);
    diffuserParamsDivChild.appendChild(diffuserParamsDivChildScale);

    const diffuserParamsDivChildSeed = document.createElement('div');
    const labelSeed = document.createElement('label');
    labelSeed.className = 'notranslate';
    labelSeed.innerText = "Seed";
    diffuserParamsDivChildSeed.appendChild(labelSeed);
    const inputSeed = document.createElement('input');
    inputSeed.className = "diffusionSeed";
    inputSeed.type = 'number';
    inputSeed.style = "width: 50px;margin-left: 5px;";
    inputSeed.setAttribute('max', '1000');
    inputSeed.setAttribute('min', '0');
    inputSeed.setAttribute('value', '0');
    inputSeed.setAttribute('step', '1');
    diffuserParamsDivChildSeed.appendChild(inputSeed);
    diffuserParamsDivChild.appendChild(diffuserParamsDivChildSeed);

    diffuserParamsDiv.appendChild(diffuserParamsDivChild);
    // Prompt
    const textareaPrompt = document.createElement('textarea');
    textareaPrompt.className = "diffusionPrompt";
    textareaPrompt.style = "min-height: 25px;max-height: 75px;max-width: 30vw;min-width: 30vw;padding: 5px;margin: 5px;";
    textareaPrompt.placeholder = "Input your prompt to draw or input 'pass' to skip";
    const textareaNegativePrompt = document.createElement('textarea');
    textareaNegativePrompt.className = "diffusionNegativePrompt";
    textareaNegativePrompt.style = "min-height: 25px;max-height: 75px;max-width: 30vw;min-width: 30vw;padding: 5px;margin: 5px;";
    textareaNegativePrompt.placeholder = "Input negative prompt";
    diffuserParamsDiv.appendChild(textareaPrompt);
    diffuserParamsDiv.appendChild(textareaNegativePrompt);

    newDivGeneral.appendChild(newDiv);
    newDivGeneral.appendChild(diffuserParamsDiv);
    maskTimelines.appendChild(newDivGeneral);

    removeBtn.addEventListener('click', function() {
        // Get the grandparent of the button
        const grandparent = this.parentNode.parentNode;
        // Remove the grandparent from the DOM
        grandparent.remove();
    });

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


async function handleDiffuser(event, previewElement, parentElement) {
    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        document.getElementById('div-general-preview-media').style.width = '';
        document.getElementById('div-general-upper').style.height = '';
        document.getElementById('div-control-panel').style.display = '';
        document.getElementById('mask-timelines').innerHTML = "";
        document.getElementById('div-preview-mask').innerHTML = `<img id="preview-mask" style="display: none;max-width: 30vw;max-height:25vh;overflow: auto;margin-top: 25px;object-fit: contain;box-shadow: rgba(240, 46, 170, 0.4) -5px 5px, rgba(240, 46, 170, 0.3) -10px 10px, rgba(240, 46, 170, 0.2) -15px 15px, rgba(240, 46, 170, 0.1) -20px 20px, rgba(240, 46, 170, 0.05) -25px 25px;">`;
        const previewMask = document.getElementById('preview-mask');

        const fileUrl = window.URL.createObjectURL(file);
        const fileType = file.type.split('/')[0];
        parentElement.style.height = "30px";
        previewElement.innerHTML = "";
        const messageElement = document.getElementById("message-about-status");

        let canvas;
        if (fileType === 'image') {
            displayMessage(messageElement, "Choose a point to get field by tool", '<i class="fa-solid fa-draw-polygon" style="margin-left: 10px;"></i>');
            canvas = await setupImageCanvas(previewElement, fileUrl, "60vh", "45vw");

            const imagePreview = previewElement.getElementsByClassName("imageMedia")[0];
            previewMask.src = imagePreview.src;
        } else if (fileType === 'video') {
            displayMessage(messageElement, "Video is loading...");
            canvas = await setupVideoTimeline(previewElement, fileUrl, "60vh", "45vw");

            displayMessage(messageElement, "Choose a point to get field by tool", '<i class="fa-solid fa-draw-polygon" style="margin-left: 10px;"></i>');

            const videoPreview = previewElement.getElementsByClassName("videoMedia")[0];

            previewMask.src = captureFrame(videoPreview);
        }

        previewMask.style.display = "";
        canvas.addEventListener('contextmenu', function(event) {
            event.preventDefault();
        });
        canvas.addEventListener('mousedown', setMultiplePointsForDiffuserPreviewMask);
    }
}

function setMultiplePointsForDiffuserPreviewMask(event) {
    triggerSendSegmentationDataMaskPreview(event, this)
}

function triggerDiffuser(elem) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => processDiffuser(data, elem))
        .catch(error => {
            console.error("Error fetching the synthesis process status:", error);
        });
}

async function processDiffuser(data, element) {
    const messageAboutStatus = element.querySelector("#message-about-status")
    if (data.status_code !== 200) {
        displayMessage(messageAboutStatus, "Invalid input, end time can not be less than start time and more than media duration");
        return;
    }

    const synthesisTable = document.getElementById("table_body_deepfake_result");
    const maskTimelinesList = element.querySelectorAll(".mask-timeline");
    const dataDict = {};

    // Background
    const paintBackgroundChecked = element.querySelector("#paintBackground").checked;

    if (maskTimelinesList.length === 0 && !paintBackgroundChecked) {
        displayMessage(messageAboutStatus, "You need to add mask before run or checked change background");
        return null
    }

    const targetDetails = retrieveMediaDetails(element.querySelector("#preview-media"));
    const mediaStartNumber = parseFloat(targetDetails.mediaStart).toFixed(2);
    const mediaEndNumber = parseFloat(targetDetails.mediaEnd).toFixed(2);

    for (const timeline of maskTimelinesList) {
        // Extracting data from canvas
        const canvas = timeline.querySelector("canvas");
        const objid = canvas.getAttribute("data-objid");
        const pointList = JSON.parse(canvas.getAttribute("data-pointlist"));

        // Extracting start and end times from inputs
        const startTimeStr = timeline.querySelector(".mask-timeline-start-time").value;
        const endTimeStr = timeline.querySelector(".mask-timeline-end-time").value;

        // Convert time from format "00:00:00" to float (seconds)
        const startTime = convertTimeToSeconds(startTimeStr);
        const endTime = convertTimeToSeconds(endTimeStr);

        // Get parameters for diffusion
        const inputStrength = timeline.querySelector(".diffusionStrength").value;
        const inputSeed = timeline.querySelector(".diffusionSeed").value;
        const inputScale = timeline.querySelector(".diffusionScale").value;
        let textareaPrompt = timeline.querySelector(".diffusionPrompt").value.trim();
        let textareaNegativePrompt = timeline.querySelector(".diffusionNegativePrompt").value.trim();

        if (textareaPrompt === "")  {
            displayMessage(messageAboutStatus, "You need to add prompt for each mask");
            return null
        } else {
            textareaPrompt = await translateWithGoogle(textareaPrompt, "auto", "en");
            textareaPrompt = textareaPrompt.toLowerCase();
        };

        if (textareaNegativePrompt !== "")  {
            textareaNegativePrompt = await translateWithGoogle(textareaNegativePrompt, "auto", "en");
            textareaNegativePrompt = textareaNegativePrompt.toLowerCase();
        };

        if (mediaStartNumber > startTime) {
            console.log("Skip mask because start time less than cut start time", mediaStartNumber, startTime);
        } else {
            // Populate the dataDict
            dataDict[objid] = {
                "start_time": startTime,
                "end_time": endTime,
                "point_list": pointList,
                "input_strength": inputStrength,
                "input_seed": inputSeed,
                "input_scale": inputScale,
                "prompt": textareaPrompt,
                "n_prompt": textareaNegativePrompt
            };
        };
    };

    if (!paintBackgroundChecked && Object.keys(dataDict).length === 0) {
        displayMessage(messageAboutStatus, "All mask start time less than cut start time");
        return null
    }

    // Background dict
    if (paintBackgroundChecked) {
        const diffusionBackgroundStrength = element.querySelector("#diffusionBackgroundStrength").value;
        const diffusionBackgroundScale = element.querySelector("#diffusionBackgroundScale").value;
        const diffusionBackgroundSeed = element.querySelector("#diffusionBackgroundSeed").value;
        let diffusionBackgroundPrompt = element.querySelector("#diffusionBackgroundPrompt").value.trim();
        let diffusionBackgroundNegativePrompt = element.querySelector("#diffusionBackgroundNegativePrompt").value.trim();

        if (diffusionBackgroundPrompt === "") {
            displayMessage(messageAboutStatus, "You checked change background but not set prompt");
            return null
        } else {
            diffusionBackgroundPrompt = await translateWithGoogle(diffusionBackgroundPrompt, "auto", "en");
            diffusionBackgroundPrompt = diffusionBackgroundPrompt.toLowerCase();
        };

        if (diffusionBackgroundNegativePrompt !== "")  {
            diffusionBackgroundNegativePrompt = await translateWithGoogle(diffusionBackgroundNegativePrompt, "auto", "en");
            diffusionBackgroundNegativePrompt = diffusionBackgroundNegativePrompt.toLowerCase();
        };

        dataDict["background"] = {
            "start_time": mediaStartNumber,
            "end_time": mediaEndNumber,
            "point_list": null,
            "input_strength": diffusionBackgroundStrength,
            "input_seed": diffusionBackgroundSeed,
            "input_scale": diffusionBackgroundScale,
            "prompt": diffusionBackgroundPrompt,
            "n_prompt": diffusionBackgroundNegativePrompt
        };
    };

    // Interval generation
    const intervalGeneration = element.querySelector("#intervalGeneration").value;

    // Preprocessor
    const diffuserLooseCfattn = element.querySelector("#diffuserLooseCfattn").checked;
    const diffuserFreeu = element.querySelector("#diffuserFreeu").checked;
    let preprocessor = "loose_cfattn";
    if (diffuserFreeu) {
        preprocessor = "freeu";
    };

    // Control Net
    const controlnetCanny = element.querySelector("#controlnetCanny").checked;
    const controlnetHed = element.querySelector("#controlnetHed").checked;
    let controlnet = "canny";
    if (controlnetHed) {
        controlnet = "hed";
    };

    // Segment field
    let percentageInputElem = element.querySelector("#percentageInput");
    let percentageInput = parseInt(percentageInputElem.value);
    let percentageInputMin = parseInt(percentageInputElem.getAttribute('min'));
    let percentageInputMax = parseInt(percentageInputElem.getAttribute('max'));

    if (isNaN(percentageInput) || percentageInput < percentageInputMin || percentageInput > percentageInputMax) {
        percentageInput = 25; // default value
    }

    let thicknessMaskElem = element.querySelector("#thicknessMask");
    let thicknessMask = parseInt(thicknessMaskElem.value);
    let thicknessMaskMin = parseInt(thicknessMaskElem.getAttribute('min'));
    let thicknessMaskMax = parseInt(thicknessMaskElem.getAttribute('max'));

    if (isNaN(thicknessMask) || thicknessMask < thicknessMaskMin || thicknessMask > thicknessMaskMax) {
        thicknessMask = 10; // default value
    }

    // get stable diffusion model
    const modelDiffusionDropdown = element.querySelector("#modelDiffusionDropdown");
    let selectedDiffusionModelValue;
    if (modelDiffusionDropdown.selectedIndex !== -1) {
        selectedDiffusionModelValue = modelDiffusionDropdown.value;
    } else {
        selectedDiffusionModelValue = null;
    }

    const diffuserParameters = {
        source: targetDetails.mediaName,
        source_start: mediaStartNumber,
        source_end: mediaEndNumber,
        source_type: targetDetails.mediaType,
        masks: dataDict,
        interval_generation: intervalGeneration,
        controlnet: controlnet,
        preprocessor: preprocessor,
        segment_percentage: percentageInput,
        thickness_mask: thicknessMask,
        sd_model_name: selectedDiffusionModelValue
    };

    console.log(JSON.stringify(diffuserParameters, null, 4));

    fetch("/synthesize_diffuser/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(diffuserParameters)
    });

    // This open display result for deepfake videos
    closeTutorial();
}