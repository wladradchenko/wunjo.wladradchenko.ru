let retouchObjIdMask = 1;


function toggleRadioOnOrNothing(radio, name) {
    const wasChecked = radio.hasAttribute('data-checked');

    // Reset all radio buttons in the group
    document.querySelectorAll(`input[name=${name}]`).forEach(r => {
        r.removeAttribute('data-checked');
        r.checked = false;
    });

    // If the clicked radio was previously checked, leave it unchecked
    // Otherwise, check it and mark it with the data attribute
    if (!wasChecked) {
        radio.setAttribute('data-checked', 'true');
        radio.checked = true;
    }
};


function showExtraOptionsRetouch() {
    const improvedRetouchObject = document.getElementById("improved-retouch-object");
    var extraOptions = document.getElementById('extraOptions');
    if (improvedRetouchObject.getAttribute('data-checked') === 'true') {
        extraOptions.style.display = 'block';
        availableFeaturesByCUDA(document.getElementById('upscaleCheckboxDiv')); // Show the extra options
    } else {
        extraOptions.style.display = 'none'; // Hide the extra options
    }
};


// RETOUCH //
function initiateRetouchAiPop(button) {
  var introRetouch = introJs();
  introRetouch.setOptions({
    steps: [
      {
        title: "Panel of remove and retouch",
        position: "right",
        intro: `
        <div style="width: 80vw; max-width: 90vw; height: 80vh; max-height: 90vh;display: flex;flex-direction: column;">
            <div id="div-general-upper"  style="display: flex;flex-direction: row;justify-content: space-around;height: 100%;">
                <div id="div-general-preview-media" style="width: 100%;">
                    <span class="dragBox" style="margin-bottom: 15px;display: flex;text-align: center;flex-direction: column;position: relative;justify-content: center;height: 100%;">
                          Load image or video
                        <input accept="image/*,video/*" type="file" onChange="handleRetouchAi(event, document.getElementById('preview-media'), this.parentElement);" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" />
                    </span>
                    <p id="message-about-status" style="text-align: center;color: #393939;height: 30px;display: none;justify-content: center;align-items: center;padding: 5px;margin-bottom: 15px;"></p>
                    <div id="preview-media" style="position: relative;max-width: 60vw; max-height:70vh;display: flex;flex-direction: column;align-items: center;">
                    </div>
                </div>
                <div id="div-control-panel" style="display: none;">
                    <div id="div-preview-mask" style="justify-content: center;display: flex;">
                    </div>
                    <fieldset style="padding: 5px;display: flex; flex-direction: column;margin-top: 30px;">
                        <legend>List of masks</legend>
                        <button class="introjs-button" onclick="maskToList();">Add new mask</button>
                        <div id="mask-timelines" style="overflow-y: auto;max-height: 20vh;"></div>
                    </fieldset>
                    <fieldset style="padding: 5pt;margin-top:10px;overflow-y: auto;max-height: 15vh">
                        <legend>Processing mode</legend>
                        <div>
                            <input type="radio" id="retouch-face" name="preprocessing_retouch" value="face" onclick="toggleRadioOnOrNothing(this, this.name);showExtraOptionsRetouch();">
                            <label for="retouch-face">Retouch face</label>
                        </div>
                        <div>
                            <input type="radio" id="retouch-object" name="preprocessing_retouch" value="object" onclick="toggleRadioOnOrNothing(this, this.name);showExtraOptionsRetouch();">
                            <label for="retouch-object">Remove object</label>
                        </div>
                        <div id="improvedRetouchObjectDiv">
                            <input type="radio" id="improved-retouch-object" name="preprocessing_retouch" value="remove_object" onclick="toggleRadioOnOrNothing(this, this.name);showExtraOptionsRetouch();">
                            <label for="improved-retouch-object">Improve remove object</label>
                        </div>
                        <div id="extraOptions" style="display: none;">
                            <div>
                                <label for="blurCoefficient">Thickness of mask:</label>
                                <input type="number" id="blurCoefficient" name="blurCoefficient" min="1" max="100" value="10" style="width: 45px;"> %
                            </div>
                            <div id="upscaleCheckboxDiv">
                                <label for="upscaleCheckbox">
                                    <input type="checkbox" id="upscaleCheckbox" name="upscale"> Restore size
                                </label>
                            </div>
                        </div>
                        <div style="margin-top: 15px;">
                            <label for="percentageInput">Matching of masks: </label>
                            <input type="number" id="percentageInput" name="percentage" min="1" max="50" value="25" style="width: 45px;"> %
                        </div>
                        <div>
                            <input type="checkbox" id="get-mask" value="#ffffff">
                            <label for="get-mask">Save mask</label>
                        </div>
                        <div id="colorPanel" style="display: none;">
                            <div id="maskColorDiv" style="align-items: center;display: flex;">
                                <label for="maskColor">Choose a mask color:</label>
                                <input style="border: none;" type="color" id="maskColor" name="maskColor" value="#ffffff">
                            </div>
                            <div>
                                <input type="checkbox" id="get-mask-transparent">
                                <label for="get-mask-transparent">Transparent</label>
                            </div>
                        </div>
                    </fieldset>
                    <button class="introjs-button" onclick="triggerRetouchAi(this.parentElement.parentElement);" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Start processing</button>
                </div>
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
  introRetouch.setOption('keyboardNavigation', false).start();

  availableFeaturesByCUDA(document.getElementById("improvedRetouchObjectDiv"));

  document.getElementById("get-mask").addEventListener("change", function() {
        if (this.checked) {
            document.getElementById("colorPanel").style.display = "block";
        } else {
            document.getElementById("colorPanel").style.display = "none";
        }
    });

    document.getElementById("maskColor").addEventListener("change", function() {
        this.style.backgroundColor = "transparent";
        document.getElementById("get-mask").value = this.value;
    });

    document.getElementById("get-mask-transparent").addEventListener("change", function() {
        if (this.checked) {
            document.getElementById("maskColor").disabled = true;
            document.getElementById("maskColor").style.backgroundColor = "transparent";
            document.getElementById("maskColor").style.borderColor = "none";
            document.getElementById("maskColorDiv").style.display = "none";
        } else {
            document.getElementById("maskColor").disabled = false;
            document.getElementById("maskColor").style.backgroundColor = "transparent";
            document.getElementById("maskColor").style.borderColor = "none";
            document.getElementById("maskColorDiv").style.display = "block";
        }
  });
}

async function handleRetouchAi(event, previewElement, parentElement) {
    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        document.getElementById('div-general-preview-media').style.width = '';
        document.getElementById('div-general-upper').style.height = '';
        document.getElementById('div-control-panel').style.display = '';
        document.getElementById('mask-timelines').innerHTML = "";
        document.getElementById('div-preview-mask').innerHTML = `<img id="preview-mask" style="display: none;padding:10px;max-width: 30vw;max-height:25vh;overflow: auto;margin-top: 25px;object-fit: contain;box-shadow: rgba(0, 0, 0, 0.4) 0px 2px 4px, rgba(0, 0, 0, 0.3) 0px 7px 13px -3px, rgba(0, 0, 0, 0.2) 0px -3px 0px inset;background: #f7db4d;">`;
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
    triggerSendSegmentationDataMaskPreview(event, this)
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
                const objId = retouchObjIdMask;

                const pointsList = retrieveSelectedPointsList(canvas);
                const {
                    mediaType,
                    mediaName,
                    mediaBlobUrl,
                    mediaCurrentTime
                } = retrieveMediaDetailsFramePreviewMask(document.getElementById("preview-media"));

                sendSegmentationDataMaskPreview(mediaName, mediaBlobUrl, pointsList, mediaCurrentTime, objId);
            } else {
                displayMessage(document.getElementById("message-about-status"), "GPU process is busy...");
            }
        })
        .catch(error => {
            console.error("Error fetching the synthesis process status:", error);
        });
}


function sendSegmentationDataMaskPreview(mediaName, mediaBlobUrl, pointsList, mediaCurrentTime, objId) {
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
        // mediaBlobUrl
        if (data.status === 200) {
            const currentMediaPreview = document.getElementById("preview-media")
            const currentImageElements = currentMediaPreview.querySelectorAll(".imageMedia");
            const currentVideoElements = currentMediaPreview.querySelectorAll(".videoMedia");
            let currentMediaBlobUrl = "";
            if (currentImageElements.length > 0) {
                currentMediaBlobUrl = currentImageElements[0].src;
            } else if (currentVideoElements.length > 0) {
                currentVideoElements[0].pause();
                currentMediaBlobUrl = captureFrame(currentVideoElements[0]);
            }
            if (currentMediaBlobUrl == mediaBlobUrl) {
                // If the POST request was successful and the status is 200, fetch the segmentation data
                fetchSegmentAnythingAndSetCanvas();
            } else {
                // message for user
                displayMessage(document.getElementById("message-about-status"), "Media was changed and previous mask will not update...");
            };
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
            const imageUrl = segmentPreview[currentTime][objId]["mask"];
            const pointList = segmentPreview[currentTime][objId]["point_list"];
            const previewMask = document.querySelector('#preview-mask');

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
                canvas.style.maxWidth = previewMask.style.maxWidth;
                canvas.style.maxHeight = previewMask.style.maxHeight;
                canvas.style.marginTop = previewMask.style.marginTop;
                ctx.drawImage(img, 0, 0);

                // Add attributes to the canvas
                canvas.setAttribute("data-objId", objId);
                canvas.setAttribute("data-currentTime", currentTime);
                canvas.setAttribute("data-pointList", JSON.stringify(pointList)); // Convert pointList to string to store as an attribute
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
    clonedCanvas.style.marginTop = "25px";

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
    newDiv.className = "mask-timeline";
    maskTimelines.appendChild(newDiv);

    removeBtn.addEventListener('click', function() {
        // Get the grandparent of the button
        const grandparent = this.parentNode;
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

function triggerRetouchAi(elem) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => processRetouchAi(data, elem))
        .catch(error => {
            console.error("Error fetching the synthesis process status:", error);
        });
}

async function processRetouchAi(data, element) {
    const messageAboutStatus = element.querySelector("#message-about-status")
    if (data.status_code !== 200) {
        displayMessage(messageAboutStatus, "Invalid input, end time can not be less than start time and more than media duration");
        return;
    }

    const maskTimelinesList = element.querySelectorAll(".mask-timeline");
    const dataDict = {};

    if (maskTimelinesList.length === 0) {
        displayMessage(messageAboutStatus, "You need to add mask before run");
        return null
    }

    const targetDetails = retrieveMediaDetails(element.querySelector("#preview-media"));
    const mediaStartNumber = parseFloat(targetDetails.mediaStart).toFixed(2);
    const mediaEndNumber = parseFloat(targetDetails.mediaEnd).toFixed(2);

    maskTimelinesList.forEach(timeline => {
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

        if (mediaStartNumber > startTime) {
            console.log("Skip mask because start time less than cut start time", targetDetails.mediaStart, startTime);
        } else {
            // Populate the dataDict
            dataDict[objid] = {
                "start_time": startTime,
                "end_time": endTime,
                "point_list": pointList
            };
        };
    });

    if (Object.keys(dataDict).length === 0) {
        displayMessage(messageAboutStatus, "All mask start time less than cut start time");
        return null
    }

    const preprocessingRetouchFace = element.querySelector("#retouch-face").getAttribute('data-checked');
    const preprocessingRetouchObject = element.querySelector("#retouch-object").getAttribute('data-checked');
    const preprocessingRetouchImprovedObject = element.querySelector("#improved-retouch-object").getAttribute('data-checked');
    const preprocessingCheckboxSaveMask = element.querySelector("#get-mask");
    const preprocessingCheckboxSaveMaskTransparent = element.querySelector("#get-mask-transparent").checked;

    let maskColor = null;
    if (preprocessingCheckboxSaveMask.checked) {
        if (preprocessingCheckboxSaveMaskTransparent) {
            maskColor = "transparent";
        } else {
            maskColor = preprocessingCheckboxSaveMask.value;
        }
    }

    let retouchModel = null
    if (preprocessingRetouchFace) {
        retouchModel = "retouch_face"
    } else if (preprocessingRetouchObject) {
        retouchModel = "retouch_object"
    } else if (preprocessingRetouchImprovedObject) {
        retouchModel = "improved_retouch_object"
    }

    if (!maskColor && !retouchModel) {
        displayMessage(messageAboutStatus, "You need to choose preprocessing or save mask");
        return null
    }

    // Get value for segmentation parameters and improve remove object
    let blurCoefficientInput = element.querySelector("#blurCoefficient");
    let blurCoefficient = blurCoefficientInput.value;
    let blurCoefficientMin = parseInt(blurCoefficientInput.getAttribute('min'));
    let blurCoefficientMax = parseInt(blurCoefficientInput.getAttribute('max'));

    if (blurCoefficient === '' || blurCoefficient < blurCoefficientMin || blurCoefficient > blurCoefficientMax) {
        blurCoefficient = 1; // default value
    }

    const upscaleCheckbox = element.querySelector("#upscaleCheckbox").checked;

    let percentageInputElem = element.querySelector("#percentageInput");
    let percentageInput = parseInt(percentageInputElem.value);
    let percentageInputMin = parseInt(percentageInputElem.getAttribute('min'));
    let percentageInputMax = parseInt(percentageInputElem.getAttribute('max'));

    if (isNaN(percentageInput) || percentageInput < percentageInputMin || percentageInput > percentageInputMax) {
        percentageInput = 25; // default value
    }

    const retouchAiParameters = {
        source: targetDetails.mediaName,
        source_start: mediaStartNumber,
        source_end: mediaEndNumber,
        source_type: targetDetails.mediaType,
        model_type: retouchModel,
        mask_color: maskColor,
        masks: dataDict,
        blur: blurCoefficient,
        upscale: upscaleCheckbox,
        segment_percentage: percentageInput
    };

    console.log(JSON.stringify(retouchAiParameters, null, 4));

    fetch("/synthesize_retouch/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(retouchAiParameters)
    });

    // This open display result for deepfake videos
    closeTutorial();
}