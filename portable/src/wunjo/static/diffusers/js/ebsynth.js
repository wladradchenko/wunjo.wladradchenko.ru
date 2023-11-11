// DIFFUSION EBSYNTH ONLY FOR GPU
window.addEventListener('load', function() {
  const menubarGeneral = document.getElementById("menubar");

  // Create the 'a' element
  const buttonEbsynth = document.createElement("a");
  buttonEbsynth.id = "a-only-ebsynth";
  buttonEbsynth.style.color = "black";
  buttonEbsynth.style.width = "3.2vw";
  buttonEbsynth.style.height = "3.2vw";
  buttonEbsynth.style.fontSize = "1.5rem";
  buttonEbsynth.style.display = "none";
  buttonEbsynth.title = "Open video style change by images";
  buttonEbsynth.addEventListener("click", (event) =>
    initiateEbsynthPop(event.currentTarget)
  );

  // Create the 'i' element and append it to the 'a' element
  const icon = document.createElement("i");
  icon.className = "fa-solid fa-brush";
  buttonEbsynth.appendChild(icon);

  // Append the 'a' element to the 'menubarGeneral'
  menubarGeneral.appendChild(buttonEbsynth);
  document
    .getElementById("a-change-processor")
    .addEventListener("click", (event) => {
      availableFeaturesByCUDA(buttonEbsynth);
    });
  availableFeaturesByCUDA(buttonEbsynth);
});
// DIFFUSION EBSYNTH ONLY FOR GPU

function initiateEbsynthPop(button) {
  var introRetouch = introJs();
  introRetouch.setOptions({
    steps: [
      {
        title: "Panel of video style change by images",
        position: "right",
        intro: `
        <div style="width: 80vw; max-width: 90vw; height: 80vh; max-height: 90vh;display: flex;flex-direction: column;">
            <div id="div-general-upper"  style="display: flex;flex-direction: row;justify-content: space-around;height: 100%;">
                <div id="div-general-preview-media" style="width: 100%;">
                    <span class="dragBox" style="margin-bottom: 15px;display: flex;text-align: center;flex-direction: column;position: relative;justify-content: center;height: 100%;">
                          Load video
                        <input accept="video/*" type="file" onChange="handleEbsynth(event, document.getElementById('preview-media'), this.parentElement);" ondragover="drag(this.parentElement)" ondrop="drop(this.parentElement)" />
                    </span>
                    <p id="message-about-status" style="text-align: center;color: #393939;height: 30px;display: none;justify-content: center;align-items: center;padding: 5px;margin-bottom: 15px;"></p>
                    <div id="preview-media" style="position: relative;max-width: 60vw; max-height:70vh;display: flex;flex-direction: column;align-items: center;">
                    </div>
                </div>
                <div id="div-control-panel" style="display: none;">
                    <fieldset style="padding: 5px;display: flex; flex-direction: column;margin-top: 30px;max-width:20vw;">
                        <legend>List of frames</legend>
                        <div id="mask-timelines" style="overflow-y: auto;max-height: 65vh;"></div>
                    </fieldset>
                    <button class="introjs-button" onclick="triggerEbsynth(this.parentElement.parentElement);" style="background: #f7db4d;margin-top: 10pt;text-align: center;width: 100%;padding-right: 0 !important;padding-left: 0 !important;padding-bottom: 0.5rem !important;padding-top: 0.5rem !important;">Start processing</button>
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
};


async function handleEbsynth(event, previewElement, parentElement) {
    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        document.getElementById('div-general-preview-media').style.width = '';
        document.getElementById('div-general-upper').style.height = '';
        document.getElementById('div-control-panel').style.display = 'none';
        document.getElementById('mask-timelines').innerHTML = "";

        const fileUrl = window.URL.createObjectURL(file);
        const fileType = file.type.split('/')[0];
        parentElement.style.height = "30px";
        previewElement.innerHTML = "";
        const messageElement = document.getElementById("message-about-status");

        let canvas;
        if (fileType === 'video') {
            displayMessage(messageElement, "Video is loading...");
            canvas = await setupVideoTimeline(previewElement, fileUrl, "60vh", "45vw");

            displayMessage(messageElement, "Choose a frame to change and click ones");

            const videoPreview = previewElement.getElementsByClassName("videoMedia")[0];

            // Extract first and last frames
            let firstFrameUrl = await extractFrameEbsynth(videoPreview, 0);
            let lastFrameUrl = await extractFrameEbsynth(videoPreview, videoPreview.duration);

            const maskTimelines = document.getElementById('mask-timelines');

            const firstFrameTime = 0; // First frame is always 0
            const lastFrameTime = videoPreview.duration; // Last frame number

            const firstFrameDiv = await createFrameDivEbsynth(firstFrameUrl, firstFrameTime);
            const lastFrameDiv = await createFrameDivEbsynth(lastFrameUrl, lastFrameTime);

            insertFrameDivSortedEbsynth(maskTimelines, firstFrameDiv);
            insertFrameDivSortedEbsynth(maskTimelines, lastFrameDiv);

            document.getElementById('div-control-panel').style.display = '';

            setupAttributeChangeListeners(videoPreview);

            // Set new frame if user click on canvas
            async function setFrameByCurrentTimeEbsynth(event) {
                // Get the current time of the video
                const currentTime = videoPreview.currentTime;
                // Check if a div for the new start frame already exists
                if (!frameDivExistsEbsynth(maskTimelines, currentTime)) {
                    // Extract the frame at the current time
                    const frameUrl = await extractFrameEbsynth(videoPreview, currentTime);
                    // Create a frame div for this frame
                    const frameDiv = await createFrameDivEbsynth(frameUrl, currentTime);
                    // Insert the frame div sorted in maskTimelines
                    insertFrameDivSortedEbsynth(maskTimelines, frameDiv);
                }
            }
            canvas.addEventListener('mousedown', setFrameByCurrentTimeEbsynth);

            // Set information about resolution
            const maxDeviceResolution = 1280;
            const useLimitResolution = true;
            setVideoResolution(previewElement, maxDeviceResolution, useLimitResolution);
        }

        canvas.addEventListener('contextmenu', function(event) {
            event.preventDefault();
        });
    }
}


function setupAttributeChangeListeners(videoElement) {
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.type === 'attributes') {
                if (mutation.attributeName === 'start') {
                    const newStart = videoElement.getAttribute('start');
                    handleStartChangeEbsynth(newStart, videoElement);
                } else if (mutation.attributeName === 'end') {
                    const newEnd = videoElement.getAttribute('end');
                    handleEndChangeEbsynth(newEnd, videoElement);
                }
            }
        });
    });

    observer.observe(videoElement, { attributes: true });
}


// Add these functions to handle the changes
async function handleStartChangeEbsynth(newStart, videoPreview) {
    // Convert newStart to seconds if necessary
    const newStartTime = parseFloat(newStart);
    const maskTimelines = document.getElementById('mask-timelines');

    // Remove divs with frameNumber less than newStartTime
    Array.from(maskTimelines.children).forEach(child => {
        if (parseFloat(child.getAttribute("frameNumber")) < newStartTime) {
            child.remove();
        }
    });

    // Check if a div for the new start frame already exists
    if (!frameDivExistsEbsynth(maskTimelines, newStartTime)) {
        const newStartFrameUrl = await extractFrameEbsynth(videoPreview, newStartTime);
        const newStartFrameDiv = await createFrameDivEbsynth(newStartFrameUrl, newStartTime);
        insertFrameDivSortedEbsynth(maskTimelines, newStartFrameDiv);
    }
}

async function handleEndChangeEbsynth(newEnd, videoPreview) {
    // Convert newEnd to seconds if necessary
    const newEndTime = parseFloat(newEnd);
    const maskTimelines = document.getElementById('mask-timelines');

    // Remove divs with frameNumber greater than newEndTime
    Array.from(maskTimelines.children).forEach(child => {
        if (parseFloat(child.getAttribute("frameNumber")) > newEndTime) {
            child.remove();
        }
    });

    // Check if a div for the new end frame already exists
    if (!frameDivExistsEbsynth(maskTimelines, newEndTime)) {
        const newEndFrameUrl = await extractFrameEbsynth(videoPreview, newEndTime);
        const newEndFrameDiv = await createFrameDivEbsynth(newEndFrameUrl, newEndTime);
        insertFrameDivSortedEbsynth(maskTimelines, newEndFrameDiv);
    }
}


function frameDivExistsEbsynth(parentElement, frameNumber) {
    const roundedFrameNumber = parseFloat(frameNumber).toFixed(3); // Round to 3 decimal places
    return Array.from(parentElement.children).some(child =>
        child.getAttribute("frameNumber") === roundedFrameNumber);
}


function insertFrameDivSortedEbsynth(parentElement, newDiv) {
    const newFrameNumber = parseFloat(newDiv.getAttribute("frameNumber"));
    let inserted = false;

    // Iterate through existing divs and find the correct position
    Array.from(parentElement.children).forEach(child => {
        const childFrameNumber = parseFloat(child.getAttribute("frameNumber"));
        if (newFrameNumber < childFrameNumber && !inserted) {
            parentElement.insertBefore(newDiv, child);
            inserted = true;
        }
    });

    // If the new div has the highest frame number, append it at the end
    if (!inserted) {
        parentElement.appendChild(newDiv);
    }
}

async function createFrameDivEbsynth(frameUrl, frameNumber) {
    const frameDiv = document.createElement('div');
    frameDiv.classList.add("mask-timeline");
    const roundedFrameNumber = frameNumber.toFixed(3); // Round and keep as a string
    frameDiv.setAttribute("frameNumber", roundedFrameNumber);
    frameDiv.style.position = 'relative';
    frameDiv.style.display = 'inline-block'; // Keeps the div from taking full width

    const frameImage = document.createElement('img');
    frameImage.src = frameUrl;
    frameImage.style.maxWidth = '20vw';
    frameImage.style.maxHeight = '20vh';
    frameImage.style.display = 'block'; // Ensures the image takes the width of its content

    const removeButton = createOverlayButtonEbsynth('<i class="fa-solid fa-trash"></i>');
    // Position the remove button, for example at the top right
    removeButton.style.top = '0';
    removeButton.style.right = '0';
    removeButton.addEventListener('click', () => {
        frameDiv.remove();
    });
    // Update style
    applyEbsynthButtonStyles(removeButton);

    // Change Button (File Input) functionality
    const changeButton = document.createElement('input');
    changeButton.type = 'file';
    changeButton.accept = 'image/*';
    changeButton.style.opacity = '0'; // Hide the default file input
    changeButton.style.position = 'absolute';
    changeButton.style.top = '0';
    changeButton.style.left = '0';
    changeButton.style.width = '30px'; // Adjust as necessary
    changeButton.style.height = '30px'; // Adjust as necessary
    changeButton.style.zIndex = '10';
    changeButton.addEventListener('change', (event) => {
        if (event.target.files && event.target.files[0]) {
            const newImageUrl = URL.createObjectURL(event.target.files[0]);
            frameImage.src = newImageUrl;
        }
    });

    // Add a visible button icon over the file input
    const changeButtonIcon = createOverlayButtonEbsynth('<i class="fa-solid fa-cloud-arrow-up"></i>');
    changeButtonIcon.style.top = '0';
    changeButtonIcon.style.left = '0';
    changeButtonIcon.addEventListener('click', () => changeButton.click()); // Trigger file input on icon click
    // Update style
    applyEbsynthButtonStyles(changeButtonIcon);

    // Create a text element for the frame number
    const frameNumberText = document.createElement('span');
    frameNumberText.textContent = formatTime(frameNumber);
    frameNumberText.style.position = 'absolute';
    frameNumberText.style.bottom = '0';
    frameNumberText.style.left = '0';
    frameNumberText.style.color = 'white';
    frameNumberText.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    frameNumberText.style.padding = '2px';
    frameNumberText.style.fontSize = 'small';

    // Append elements to frameDiv
    frameDiv.appendChild(frameImage);
    frameDiv.appendChild(changeButton); // Actual file input (hidden)
    frameDiv.appendChild(changeButtonIcon); // Visible icon
    frameDiv.appendChild(removeButton);
    frameDiv.appendChild(frameNumberText);

    return frameDiv;
}

function applyEbsynthButtonStyles(button) {
    // Applying the .ebsynth-buttons class styles
    button.style.width = '18pt';
    button.style.height = '18pt';
    button.style.display = 'inline';
    button.style.border = 'none';
    button.style.borderRadius = '50%';
    button.style.transition = 'box-shadow 0.3s ease';
    button.style.margin = '4px';

    // Adding hover effect
    button.onmouseover = function() {
        this.style.borderColor = 'rgb(230, 231, 238)';
        this.style.boxShadow = 'rgb(184, 185, 190) 2px 2px 5px inset, rgb(255, 255, 255) -3px -3px 7px inset';
    };
    button.onmouseout = function() {
        this.style.borderColor = '';
        this.style.boxShadow = '';
    };
}

function createOverlayButtonEbsynth(i) {
    const button = document.createElement('button');
    button.innerHTML = i;
    button.style.position = 'absolute'; // Position button absolutely within its parent div
    button.style.zIndex = 10; // Ensure button is above the image
    // Add any additional styling here
    return button;
}

async function extractFrameEbsynth(video, time) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        video.currentTime = time;
        video.addEventListener('seeked', function onSeeked() {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            video.removeEventListener('seeked', onSeeked);
            resolve(canvas.toDataURL());
        });
    });
}


function triggerEbsynth(elem) {
    fetch("/synthesize_process/")
        .then(response => response.json())
        .then(data => processEbsynth(data, elem))
        .catch(error => {
            console.error("Error fetching the synthesis process status:", error);
        });
}


async function processEbsynth(data, element) {
    const messageAboutStatus = element.querySelector("#message-about-status")
    if (data.status_code !== 200) {
        displayMessage(messageAboutStatus, "Invalid input, end time can not be less than start time and more than media duration");
        return;
    }

    const maskTimelinesList = element.querySelectorAll(".mask-timeline");
    const dataDict = {};
    const targetDetails = retrieveMediaDetails(element.querySelector("#preview-media"));
    const mediaStartNumber = parseFloat(targetDetails.mediaStart).toFixed(2);
    const mediaEndNumber = parseFloat(targetDetails.mediaEnd).toFixed(2);
    let objid = 0;

    for (const timeline of maskTimelinesList) {
        const imgFrame = timeline.querySelector("img");
        const frameNumber = timeline.getAttribute("frameNumber");

        let mediaBlobUrl = imgFrame.src;
        let mediaName = `image_${Date.now()}_${getRandomString(5)}`;

        if (mediaBlobUrl) {
            fetch(mediaBlobUrl)
                .then((res) => res.blob())
                .then((blob) => {
                    var file = new File([blob], mediaName);
                    uploadFile(file); // Ensure this function is defined elsewhere
                })
                .catch(error => console.error('Error uploading file:', error));
        }

        dataDict[objid] = {
            "img_name": mediaName, // Assuming you meant mediaName here
            "frame_time": frameNumber
        };

        objid++;
    }

    if (Object.keys(dataDict).length === 0) {
        displayMessage(messageAboutStatus, "You need to add minimum one frame by click on video preview");
        return null
    }

    const ebsynthParameters = {
        source: targetDetails.mediaName,
        source_start: mediaStartNumber,
        source_end: mediaEndNumber,
        source_type: targetDetails.mediaType,
        masks: dataDict
    };

    console.log(JSON.stringify(ebsynthParameters, null, 4));

    fetch("/synthesize_only_ebsynth/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ebsynthParameters)
    });

    // This open display result for deepfake videos
    closeTutorial();
}