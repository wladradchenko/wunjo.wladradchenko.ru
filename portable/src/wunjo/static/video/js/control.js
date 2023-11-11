// WORK WITH VIDEO //
async function setupVideoTimeline(parentElement, videoUrl, videoDisplayHeight='45vh', videoDisplayWidth = '45vw') {
    // CREATE HTML //
    // Create the required elements
    const videoContainer = document.createElement('div');
    videoContainer.className = "ui-widget-content";
    videoContainer.style = "display: flex;overflow: hidden;z-index: 1;position: relative;";

    const videoElement = document.createElement('video');
    videoElement.className = "videoMedia";
    videoElement.src = videoUrl;
    videoElement.setAttribute("loop", "");
    videoElement.setAttribute("muted", "");
    videoElement.style.opacity = "1.0";
    videoElement.style = "position: relative; width: auto;";
    videoElement.style.maxWidth = videoDisplayWidth;
    videoElement.style.maxHeight = videoDisplayHeight;
    videoElement.style.zIndex = "1";

    const canvasElement = document.createElement('canvas');
    canvasElement.className = "canvasMedia";
    canvasElement.style.position = "absolute";
    canvasElement.style.left = "0";
    canvasElement.style.top = "0";
    canvasElement.style.zIndex = "2";

    const spanResolutionElement = document.createElement('span');
    spanResolutionElement.className = "spanResolution";
    spanResolutionElement.style.position = "absolute";
    spanResolutionElement.style.left = "0";
    spanResolutionElement.style.top = "0";
    spanResolutionElement.style.color = "white";
    spanResolutionElement.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    spanResolutionElement.style.padding = "2px";
    spanResolutionElement.style.fontSize = "small";
    spanResolutionElement.style.zIndex = "10";
    spanResolutionElement.style.display = "none";
    videoContainer.appendChild(spanResolutionElement);

    videoContainer.appendChild(canvasElement);

    const controlDiv = document.createElement('div');
    controlDiv.className = "timeline-control-panel";
    controlDiv.style.zIndex = "5";
    controlDiv.style.display = "none";

    const controlFirstSubDiv = document.createElement('div');
    controlFirstSubDiv.style.alignItems = "center";
    controlFirstSubDiv.style.display = "flex";

    const currentTimeSpan = document.createElement('span');
    currentTimeSpan.style.fontSize = "10px";
    currentTimeSpan.style.color = "black";
    currentTimeSpan.innerText = "00:00:00";

    const splashTimeSpan = document.createElement('span');
    splashTimeSpan.innerText = "/";
    splashTimeSpan.style = "margin-left: 5px; margin-right: 5px;";

    const endTimeSpan = document.createElement('span');
    endTimeSpan.style.fontSize = "10px";
    endTimeSpan.style.color = "black";
    endTimeSpan.innerText = "00:00:00";

    controlFirstSubDiv.appendChild(currentTimeSpan);
    controlFirstSubDiv.appendChild(splashTimeSpan);
    controlFirstSubDiv.appendChild(endTimeSpan);

    const controlSecondSubDiv = document.createElement('div');

    const playToggleBtn = document.createElement('button');
    playToggleBtn.className = 'timeline-buttons';
    playToggleBtn.innerHTML = '<i class="fa fa-play"></i>';

    const muteToggle = document.createElement('button');
    muteToggle.className = 'timeline-buttons';
    muteToggle.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>';

    const filmstripDiv = document.createElement('div');
    filmstripDiv.className = "timeline";
    filmstripDiv.id = "filmstrip";
    filmstripDiv.style = "display: flex; overflow-x: auto;flex-direction: column;";
    filmstripDiv.style.zIndex = "5";

    // Append the elements
    videoContainer.appendChild(videoElement);
    controlSecondSubDiv.appendChild(playToggleBtn);
    controlSecondSubDiv.appendChild(muteToggle);

    const controlThirdSubDiv = document.createElement('div');

    const drawCanvasClear = document.createElement('button');
    drawCanvasClear.className = 'timeline-buttons';
    drawCanvasClear.innerHTML = '<i class="fa-solid fa-trash"></i>';

    const drawCanvasButton = document.createElement('button');
    drawCanvasButton.className = 'timeline-buttons';
    drawCanvasButton.innerHTML = '<i class="fa-solid fa-draw-polygon"></i>';

    const plusSize = document.createElement('button');
    plusSize.className = 'timeline-buttons';
    plusSize.innerHTML = '<i class="fa-solid fa-magnifying-glass-plus"></i>';

    const minusSize = document.createElement('button');
    minusSize.className = 'timeline-buttons';
    minusSize.innerHTML = '<i class="fa-solid fa-magnifying-glass-minus"></i>';

    controlThirdSubDiv.appendChild(drawCanvasClear);
    controlThirdSubDiv.appendChild(drawCanvasButton);
    controlThirdSubDiv.appendChild(plusSize);
    controlThirdSubDiv.appendChild(minusSize);

    controlDiv.appendChild(controlFirstSubDiv)
    controlDiv.appendChild(controlSecondSubDiv);
    controlDiv.appendChild(controlThirdSubDiv);

    parentElement.appendChild(videoContainer);
    parentElement.appendChild(controlDiv);
    parentElement.appendChild(filmstripDiv);
    // CREATE HTML //

    let video_size = {'w': 0, 'h': 0};
    let filename = 'in.mp4';
    let selected_file = null;
    let numberOfFrames = 10; // TODO including first and last frame

    // DRAW CANVAS LOGICAL //
    let isCanvasOnTop = true; // To track the toggle state

    function toggleCanvasVideoZIndex() {
        if (isCanvasOnTop) {
            drawCanvasButton.innerHTML = '<i class="fa-solid fa-ruler-combined"></i>';
            canvasElement.style.zIndex = '1';
            videoElement.style.zIndex = '2';
            videoElement.style.opacity = "0.8";
        } else {
            drawCanvasButton.innerHTML = '<i class="fa-solid fa-draw-polygon"></i>';
            canvasElement.style.zIndex = '2';
            videoElement.style.zIndex = '1';
            videoElement.style.opacity = "1.0";
        }
        isCanvasOnTop = !isCanvasOnTop; // Toggle the state
    }

    function toggleCanvasZIndexOff() {
        if (isCanvasOnTop) {
            drawCanvasButton.innerHTML = '<i class="fa-solid fa-ruler-combined"></i>';
            canvasElement.style.zIndex = '1';
            videoElement.style.zIndex = '2';
            videoElement.style.opacity = "0.8";
            isCanvasOnTop = false;
        }
    }
    // DRAW CANVAS LOGICAL //

    drawCanvasButton.addEventListener('click', toggleCanvasVideoZIndex);

    // ZOOM //
    let scaleFactor = 1;

    plusSize.addEventListener("click", function() {
        scaleFactor += 0.1; // Increase the zoom by 10%
        videoElement.style.transform = `scale(${scaleFactor})`;
        canvasElement.style.transform = `scale(${scaleFactor})`;
        // Set canvas off
        toggleCanvasZIndexOff();
    });

    minusSize.addEventListener("click", function() {
        scaleFactor = Math.max(1, scaleFactor - 0.1); // Decrease the zoom by 10% but don't go below 1 (original size)
        videoElement.style.transform = `scale(${scaleFactor})`;
        videoElement.style.left = '0px';
        videoElement.style.top = '0px';
        canvasElement.style.transform = `scale(${scaleFactor})`;
        canvasElement.style.left = '0px';
        canvasElement.style.top = '0px';
        // Set canvas off
        toggleCanvasZIndexOff();
    });

    let isDraggingZoom = false;
    let prevX = 0;
    let prevY = 0;

    videoElement.addEventListener('mousedown', function(e) {
        if (scaleFactor > 1) { // Only allow dragging if zoomed in
            isDraggingZoom = true;
            prevXZoom = e.clientX;
            prevYZoom = e.clientY;
        }
    });

    videoElement.addEventListener('mousemove', function(e) {
        if (isDraggingZoom) {
            const deltaX = e.clientX - prevXZoom;
            const deltaY = e.clientY - prevYZoom;

            let left = parseInt(videoElement.style.left || '0');
            let top = parseInt(videoElement.style.top || '0');

            // Calculate new left and top values
            left += deltaX;
            top += deltaY;

            // Constraints to ensure the video doesn't get dragged out of view
            const maxLeft = (scaleFactor - 1) * video_size["offsetWidth"] * 0.5;
            const minLeft = (scaleFactor - 1) * video_size["offsetWidth"] * -0.5;
            const maxTop = (scaleFactor - 1) * video_size["offsetHeight"] * 0.5;
            const minTop = (scaleFactor - 1) * video_size["offsetHeight"] * -0.5;

            left = Math.min(maxLeft, Math.max(minLeft, left));
            top = Math.min(maxTop, Math.max(minTop, top));

            videoElement.style.left = `${left}px`;
            videoElement.style.top = `${top}px`;
            canvasElement.style.left = `${left}px`;
            canvasElement.style.top = `${top}px`;

            prevXZoom = e.clientX;
            prevYZoom = e.clientY;
        }
    });


    videoElement.addEventListener('mouseup', function() {
        isDraggingZoom = false;
    });
    // ZOOM //

    muteToggle.addEventListener("click", function() {
        videoElement.muted = !videoElement.muted;
        if (!videoElement.muted) {
            muteToggle.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>';
        } else {
            muteToggle.innerHTML = '<i class="fa-solid fa-volume-low"></i>';
        }
    });

    videoElement.addEventListener("loadedmetadata", function(e) {
        this.setAttribute("start", 0);
        this.setAttribute("end", this.duration);

        video_size = {'w': this.videoWidth, 'h': this.videoHeight, 'offsetWidth': this.offsetWidth, "offsetHeight": this.offsetHeight};

        // Calculate the aspect ratio of the original video
        let aspectRatio = video_size.w / video_size.h;
        // Determine which offset dimension is the limiting factor
        if (video_size.offsetWidth / video_size.offsetHeight < aspectRatio) {
            // offsetWidth is the limiting factor, adjust offsetHeight
            video_size.offsetHeight = video_size.offsetWidth / aspectRatio;
        } else {
            // offsetHeight is the limiting factor, adjust offsetWidth
            video_size.offsetWidth = video_size.offsetHeight * aspectRatio;
        }

        this.width = video_size["offsetWidth"]  // float to int
        this.height = video_size["offsetHeight"]  // float to int
        canvasElement.width = video_size["offsetWidth"]
        canvasElement.height = video_size["offsetHeight"]
    });

    videoElement.addEventListener("loadeddata", function() {
        videoElement.pause();
    });

    videoElement.addEventListener("pause", function(e) {
        console.log('Paused: ', e.target.currentTime);
    });

    playToggleBtn.addEventListener("click", function() {
        if (videoElement.paused) {
            videoElement.play();
            playToggleBtn.innerHTML = '<i class="fa fa-pause"></i>';
        } else {
            videoElement.pause();
            playToggleBtn.innerHTML = '<i class="fa fa-play"></i>';
        }
    });

    const getBlobUrl = async (videoUrl) => {
        const blob = await (await fetch(videoUrl)).blob();
        return URL.createObjectURL(blob);
    }


    const generateVideoThumbnails = (src) => {
        return new Promise((resolve) => {
            const canvas = document.createElement("canvas");
            const video = document.createElement("video");

            video.autoplay = true;
            video.muted = true;
            video.src = src;
            video.crossOrigin = "anonymous";

            const frames = [];

            video.onloadeddata = async () => {
                let ctx = canvas.getContext("2d");

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // set 60 from .timeline img
                const numRepeat = videoElement.offsetWidth / ((60 / canvas.height * canvas.width) * 10)

                video.pause();

                const interval = video.duration / (numberOfFrames - 1);

                for (let i = 0; i < numberOfFrames; i++) {
                    const seekTime = i * interval;
                    video.currentTime = seekTime;
                    await new Promise(r => video.onseeked = r); // wait for frame to load after seeking

                    // Adjust the canvas size based on numRepeat
                    canvas.width = video.videoWidth * numRepeat;
                    canvas.height = video.videoHeight;

                    // Repeatedly draw the video frame onto the canvas
                    for (let repeatIndex = 0; repeatIndex < numRepeat; repeatIndex++) {
                        ctx.drawImage(video, repeatIndex * video.videoWidth, 0, video.videoWidth, video.videoHeight);
                    }

                    const dataUrl = canvas.toDataURL("image/png");
                    const blob = await (await fetch(dataUrl)).blob();
                    const blobUrl = URL.createObjectURL(blob);
                    frames.push(blobUrl);
                }

                resolve(frames);
            };
        });
    };

    // Generate filmstrips after loading video
    let blobUrl = await getBlobUrl(videoElement.src);
    let thumbnails = await generateVideoThumbnails(blobUrl);

    filmstripDiv.innerHTML = '';
    // CREATE ELEMENTS FOR TIMELINES //
    // Play slider
    const timelinePlaySlider = document.createElement('div');
    timelinePlaySlider.classList.add('timeline-play-slider');
    filmstripDiv.appendChild(timelinePlaySlider);
    // Cut slider
    const timelineDivSlider = document.createElement('div');
    timelineDivSlider.classList.add('timeline-div-slider');
    filmstripDiv.appendChild(timelineDivSlider);
    // Marker data about time
    const timeMarkerDiv = document.createElement('div');
    timeMarkerDiv.classList.add('time-marker-div');
    filmstripDiv.appendChild(timeMarkerDiv);
    // Timeline div with images
    const timelineDiv = document.createElement('div');
    timelineDiv.classList.add('timeline-div');
    filmstripDiv.appendChild(timelineDiv);

    // TIMELINES IMAGES //
    // Load all images and then append them to the DOM in order
    const loadThumbnail = (thumbUrl, index) => {
        return new Promise((resolve) => {
            let img = document.createElement('img');
            img.src = thumbUrl;
            img.onload = () => {
                let thumbnailWidth = img.clientWidth;

                // Store the image, its width, and index in an object
                resolve({ img, thumbnailWidth, index });
            };
        });
    };

    const loadedThumbnails = await Promise.all(thumbnails.map(loadThumbnail));

    // Now append each loaded thumbnail to the DOM in the correct order
    loadedThumbnails.forEach(({ img, thumbnailWidth, index }) => {
        timelineDiv.appendChild(img);
        let thumbnailWidthLocal = img.clientWidth;

        console.log("Thumbnail index:", index); // Debugging line

        // TIMELINES TIME //
        // Add time marker every 1 frame
        if (index % 2 === 0) {
            console.log("Adding time marker at index:", index); // Debugging line
            const timeMarker = document.createElement('div');
            timeMarker.classList.add('time-marker');
            timeMarker.innerText = formatTime(index * (videoElement.duration / (numberOfFrames - 1)));
            timeMarkerDiv.appendChild(timeMarker);
        }
        // TIMELINES TIME //
    });
    // TIMELINES IMAGES //
    // Set correct size as in width for timeline images full
    timeMarkerDiv.style.width = video_size["offsetWidth"] + "px" // timelineDiv.scrollWidth + "px";
    timelineDivSlider.style.width = video_size["offsetWidth"] + "px"  // timelineDiv.scrollWidth + "px";
    // Create elements to control cut
    const timelineDivSliderToggleLeft = document.createElement('div');
    timelineDivSliderToggleLeft.classList.add('timeline-div-slider-toggle-left');
    timelineDivSlider.appendChild(timelineDivSliderToggleLeft);
    const timelineDivSliderToggleRight = document.createElement('div');
    timelineDivSliderToggleRight.classList.add('timeline-div-slider-toggle-right');
    timelineDivSlider.appendChild(timelineDivSliderToggleRight);

    // CONTROL CUT SLIDERS //
    let videoStartTime = 0;
    let videoEndTime = videoElement.duration;
    endTimeSpan.innerText = formatTime(videoEndTime);

    function updateVideoTimes() {
        const totalWidth = timelineDiv.scrollWidth;
        const currentMarginLeft = parseInt(getComputedStyle(timelineDivSlider).marginLeft) || 0;
        const currentWidth = timelineDivSlider.offsetWidth;

        // Calculate start and end times based on percentages
        videoStartTime = (currentMarginLeft / totalWidth) * videoElement.duration;
        videoEndTime = ((currentMarginLeft + currentWidth) / totalWidth) * videoElement.duration;

        // Add attributes for video
        videoElement.setAttribute("start", videoStartTime);
        videoElement.setAttribute("end", videoEndTime);

        if (videoStartTime > videoElement.currentTime) {
            videoElement.currentTime = videoStartTime; // Update video's current time to start time
        }
    }

    let isDraggingLeft = false;
    let isDraggingRight = false;

    timelineDivSliderToggleLeft.addEventListener("mousedown", function() {
        isDraggingLeft = true;
        document.addEventListener("mousemove", dragLeft);
        document.addEventListener("mouseup", stopDragLeft);
    });

    timelineDivSliderToggleRight.addEventListener("mousedown", function() {
        isDraggingRight = true;
        document.addEventListener("mousemove", dragRight);
        document.addEventListener("mouseup", stopDragRight);
    });

    function dragLeft(e) {
        // TODO limit 1 second or 1 frame?
        const min_slider_width = timelineDiv.scrollWidth / videoElement.duration;

        const change = e.movementX;
        const currentMarginLeft = parseInt(getComputedStyle(timelineDivSlider).marginLeft) || 0;
        const currentWidth = timelineDivSlider.offsetWidth;

        // Calculate potential new margin and width
        let newMargin = currentMarginLeft + change;
        let newWidth = currentWidth - change;

        // Ensure new margin and width don't surpass the timeline's width
        if (newMargin + newWidth > timelineDiv.scrollWidth) {
            newMargin = timelineDiv.scrollWidth - newWidth;
        }

        // Ensure they don't go beyond other limits
        newMargin = Math.max(newMargin, 0);
        newWidth = Math.min(Math.max(newWidth, min_slider_width), timelineDiv.scrollWidth);

        timelineDivSlider.style.marginLeft = newMargin + "px";
        timelineDivSlider.style.width = newWidth + "px";

        videoElement.pause()
        playToggleBtn.innerHTML = '&#9654;';
        timelinePlaySlider.style.display = 'none';
        let clickPosition = e.clientX - timelineDiv.getBoundingClientRect().left; // Position of click within the timelineDiv
        let proportion = clickPosition / timelineDiv.scrollWidth; // Proportion of the click position relative to the full width of the timelineDiv
        let newTime = proportion * videoElement.duration; // Convert the proportion to a time in the video
        videoElement.currentTime = newTime; // Set the video's current time
        timelinePlaySlider.style.left = clickPosition + "px"; // Set the position of the timelinePlaySlider
    }

    function dragRight(e) {
        // TODO limit 1 second or 1 frame?
        const min_slider_width = timelineDiv.scrollWidth / videoElement.duration;

        const change = e.movementX;
        const currentWidth = timelineDivSlider.offsetWidth;

        // Calculate new values but ensure they don't go beyond limits
        const maxAllowedWidth = timelineDiv.scrollWidth - parseInt(getComputedStyle(timelineDivSlider).marginLeft);
        const newWidth = Math.min(Math.max(currentWidth + change, min_slider_width), maxAllowedWidth);

        timelineDivSlider.style.width = newWidth + "px";
        timelineDivSlider.style.width = newWidth + "px";

        videoElement.pause()
        playToggleBtn.innerHTML = '&#9654;';
        timelinePlaySlider.style.display = 'none';
        let clickPosition = e.clientX - timelineDiv.getBoundingClientRect().left; // Position of click within the timelineDiv
        let proportion = clickPosition / timelineDiv.scrollWidth; // Proportion of the click position relative to the full width of the timelineDiv
        let newTime = proportion * videoElement.duration; // Convert the proportion to a time in the video
        videoElement.currentTime = newTime; // Set the video's current time
        timelinePlaySlider.style.left = clickPosition + "px"; // Set the position of the timelinePlaySlider
    }

    // Add the updateVideoTimes function to the stopDragLeft and stopDragRight functions
    function stopDragLeft() {
        isDraggingLeft = false;
        document.removeEventListener("mousemove", dragLeft);
        document.removeEventListener("mouseup", stopDragLeft);

        updateVideoTimes();
        timelinePlaySlider.style.display = 'inline';
    }

    function stopDragRight() {
        isDraggingRight = false;
        document.removeEventListener("mousemove", dragRight);
        document.removeEventListener("mouseup", stopDragRight);

        updateVideoTimes();
        timelinePlaySlider.style.display = 'inline';
    }

    // Ensure video stops playing when it reaches the end time
    videoElement.addEventListener("timeupdate", function() {
        if (videoElement.currentTime >= videoEndTime) {
            // video.pause();
            videoElement.currentTime = videoStartTime;
        }
    });
    // CONTROL CUT SLIDERS //
    // CONTROL PLAY SLIDER //
    timelineDivSlider.addEventListener("click", function(e) {
        if (e.target.classList.contains('timeline-div-slider-toggle-left') || e.target.classList.contains('timeline-div-slider-toggle-right')) {
            // Stop processing this event
            return;
        }
        let clickPosition = e.clientX - timelineDiv.getBoundingClientRect().left; // Position of click within the timelineDiv
        let proportion = clickPosition / timelineDiv.scrollWidth; // Proportion of the click position relative to the full width of the timelineDiv
        let newTime = proportion * videoElement.duration; // Convert the proportion to a time in the video

        // Ensure the newTime is within the bounds of videoStartTime and videoEndTime
        newTime = Math.max(videoStartTime, newTime);
        newTime = Math.min(videoEndTime, newTime);

        videoElement.currentTime = newTime; // Set the video's current time
        timelinePlaySlider.style.left = clickPosition + "px"; // Set the position of the timelinePlaySlider
    });

    videoElement.addEventListener("timeupdate", function() {
        // Calculate which thumbnail corresponds to the current time
        const index = Math.floor(videoElement.currentTime);
        const selectedThumb = filmstripDiv.children[index];

        // Scroll to the selected thumbnail
        if (selectedThumb) {
            selectedThumb.scrollIntoView({ behavior: "smooth", inline: "center" });
        }

        // Play slider
        const leftPosition = (videoElement.currentTime / videoElement.duration) * timelineDiv.scrollWidth;
        timelinePlaySlider.style.left = leftPosition + "px";

        // Set current time
        currentTimeSpan.innerText = formatTime(videoElement.currentTime);
    });
    // CONTROL PLAY SLIDER //

    // CLEAR CANVAS //
    drawCanvasClear.addEventListener('click', function() {
        const ctx = canvasElement.getContext('2d');

        if (canvasElement.hasAttribute("data-point-position")) {
            // For one point
            canvasElement.removeAttribute("data-point-position");
        }

        if (canvasElement.hasAttribute("data-points-list")) {
            // If using list of points
            let pointsList = JSON.parse(canvasElement.getAttribute("data-points-list"));

            if (pointsList.length > 0) {
                pointsList.pop();  // Remove the last point from the list
                canvasElement.setAttribute("data-points-list", JSON.stringify(pointsList));
            }

            // Clear the canvas
            ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

            // Redraw all remaining points on the canvas
            pointsList.forEach(point => {
                // Here, re-use your logic to draw a point on the canvas.
                // Example:
                ctx.fillStyle = point.color;
                ctx.beginPath();
                ctx.arc(point.x, point.y, 8, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            });
        } else {
            // If there's no "data-points-list" attribute, just clear the canvas.
            ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }
    });
    // CLEAR CANVAS //

    controlDiv.style.display = "flex";
    controlDiv.style.width = "100%";

    return canvasElement;
};
// WORK WITH VIDEO //

// WORK WITH IMAGES //
async function setupImageCanvas(parentElement, imageUrl, imageDisplayHeight = '45vh', imageDisplayWidth = '45vw') {
    // CREATE HTML //
    // Create the required elements
    const imageContainer = document.createElement('div');
    imageContainer.className = "ui-widget-content";
    imageContainer.style = "position: relative; overflow: hidden;z-index: 1;position: relative;";

    const spanResolutionElement = document.createElement('span');
    spanResolutionElement.className = "spanResolution";
    spanResolutionElement.style.position = "absolute";
    spanResolutionElement.style.left = "0";
    spanResolutionElement.style.top = "0";
    spanResolutionElement.style.color = "white";
    spanResolutionElement.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    spanResolutionElement.style.padding = "2px";
    spanResolutionElement.style.fontSize = "small";
    spanResolutionElement.style.zIndex = "10";
    spanResolutionElement.style.display = "none";
    imageContainer.appendChild(spanResolutionElement);

    const imageElement = document.createElement('img');
    imageElement.className = "imageMedia";
    imageElement.src = imageUrl;
    imageElement.style = "position: relative;width: auto;height:auto;mix-blend-mode: multiply;";
    imageElement.style.maxWidth = imageDisplayWidth;
    imageElement.style.maxHeight = imageDisplayHeight;
    imageElement.style.zIndex = "1";

    imageElement.addEventListener("load", function(e) {
        imageSize = {'w': this.width, 'h': this.height, 'offsetWidth': this.offsetWidth, "offsetHeight": this.offsetHeight};
        // Calculate the aspect ratio of the original video
        let aspectRatio = imageSize.w / imageSize.h;
        // Determine which offset dimension is the limiting factor
        if (imageSize.offsetWidth / imageSize.offsetHeight < aspectRatio) {
            // offsetWidth is the limiting factor, adjust offsetHeight
            imageSize.offsetHeight = imageSize.offsetWidth / aspectRatio;
        } else {
            // offsetHeight is the limiting factor, adjust offsetWidth
            imageSize.offsetWidth = imageSize.offsetHeight * aspectRatio;
        }

        const imgWidth = this.width;
        const imgHeight = this.height;
        canvasElement.width = imgWidth;
        canvasElement.height = imgHeight;
    });

    const canvasElement = document.createElement('canvas');
    canvasElement.className = "canvasMedia";
    canvasElement.style.position = "absolute";
    canvasElement.style.left = "0";
    canvasElement.style.top = "0";
    canvasElement.style.zIndex = "2";

    imageContainer.appendChild(canvasElement);
    imageContainer.appendChild(imageElement);

    parentElement.appendChild(imageContainer);

    let imageSize = {'w': 0, 'h': 0};

    // DRAW CANVAS LOGICAL //
    let isCanvasOnTop = true; // To track the toggle state

    function toggleCanvasImageZIndex() {
        if (isCanvasOnTop) {
            drawCanvasButton.innerHTML = '<i class="fa-solid fa-ruler-combined"></i>';
            canvasElement.style.zIndex = '1';
            imageElement.style.zIndex = '2';
        } else {
            drawCanvasButton.innerHTML = '<i class="fa-solid fa-draw-polygon"></i>';
            canvasElement.style.zIndex = '2';
            imageElement.style.zIndex = '1';
        }
        isCanvasOnTop = !isCanvasOnTop; // Toggle the state
    }

    function toggleCanvasZIndexOff() {
        if (isCanvasOnTop) {
            drawCanvasButton.innerHTML = '<i class="fa-solid fa-ruler-combined"></i>';
            canvasElement.style.zIndex = '1';
            imageElement.style.zIndex = '2';
            isCanvasOnTop = false;
        }
    }

    const drawCanvasButton = document.createElement('button');
    drawCanvasButton.className = 'timeline-buttons';
    drawCanvasButton.innerHTML = '<i class="fa-solid fa-draw-polygon"></i>';
    drawCanvasButton.addEventListener('click', toggleCanvasImageZIndex);
    // DRAW CANVAS LOGICAL //

    // ZOOM //
    let scaleFactor = 1;

    const plusSize = document.createElement('button');
    plusSize.className = 'timeline-buttons';
    plusSize.innerHTML = '<i class="fa-solid fa-magnifying-glass-plus"></i>';

    const minusSize = document.createElement('button');
    minusSize.className = 'timeline-buttons';
    minusSize.innerHTML = '<i class="fa-solid fa-magnifying-glass-minus"></i>';

    const controlDiv = document.createElement('div');
    controlDiv.className = "timeline-control-panel";
    controlDiv.appendChild(plusSize);
    controlDiv.appendChild(minusSize);
    parentElement.appendChild(controlDiv);

    plusSize.addEventListener("click", function() {
        scaleFactor += 0.1; // Increase the zoom by 10%
        imageElement.style.transform = `scale(${scaleFactor})`;
        canvasElement.style.transform = `scale(${scaleFactor})`;
        toggleCanvasZIndexOff();
    });

    minusSize.addEventListener("click", function() {
        scaleFactor = Math.max(1, scaleFactor - 0.1); // Decrease the zoom by 10% but don't go below 1 (original size)
        imageElement.style.transform = `scale(${scaleFactor})`;
        canvasElement.style.transform = `scale(${scaleFactor})`;
        toggleCanvasZIndexOff();
    });

    let isDraggingZoom = false;
    let prevX = 0;
    let prevY = 0;

    imageElement.addEventListener('mousedown', function(e) {
        if (scaleFactor > 1) { // Only allow dragging if zoomed in
            isDraggingZoom = true;
            prevX = e.clientX;
            prevY = e.clientY;
        }
    });

    imageElement.addEventListener('mousemove', function(e) {
        if (isDraggingZoom) {
            const deltaX = e.clientX - prevX;
            const deltaY = e.clientY - prevY;

            let left = parseInt(imageElement.style.left || '0');
            let top = parseInt(imageElement.style.top || '0');

            // Calculate new left and top values
            left += deltaX;
            top += deltaY;

            // Constraints to ensure the video doesn't get dragged out of view
            const maxLeft = (scaleFactor - 1) * imageSize["offsetWidth"] * 0.5;
            const minLeft = (scaleFactor - 1) * imageSize["offsetWidth"] * -0.5;
            const maxTop = (scaleFactor - 1) * imageSize["offsetHeight"] * 0.5;
            const minTop = (scaleFactor - 1) * imageSize["offsetHeight"] * -0.5;

            left = Math.min(maxLeft, Math.max(minLeft, left));
            top = Math.min(maxTop, Math.max(minTop, top));

            imageElement.style.left = `${left}px`;
            imageElement.style.top = `${top}px`;
            canvasElement.style.left = `${left}px`;
            canvasElement.style.top = `${top}px`;

            prevX = e.clientX;
            prevY = e.clientY;
        }
    });

    imageElement.addEventListener('mouseup', function() {
        isDraggingZoom = false;
    });
    // ZOOM //

    // CLEAR CANVAS //
    const drawCanvasClear = document.createElement('button');
    drawCanvasClear.className = 'timeline-buttons';
    drawCanvasClear.innerHTML = '<i class="fa-solid fa-trash"></i>';

    drawCanvasClear.addEventListener('click', function() {
        const ctx = canvasElement.getContext('2d');

        if (canvasElement.hasAttribute("data-point-position")) {
            // For one point
            canvasElement.removeAttribute("data-point-position");
        }

        if (canvasElement.hasAttribute("data-points-list")) {
            // If using list of points
            let pointsList = JSON.parse(canvasElement.getAttribute("data-points-list"));

            if (pointsList.length > 0) {
                pointsList.pop();  // Remove the last point from the list
                canvasElement.setAttribute("data-points-list", JSON.stringify(pointsList));
            }

            // Clear the canvas
            ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

            // Redraw all remaining points on the canvas
            pointsList.forEach(point => {
                // Here, re-use your logic to draw a point on the canvas.
                // Example:
                ctx.fillStyle = point.color;
                ctx.beginPath();
                ctx.arc(point.x, point.y, 8, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            });
        } else {
            // If there's no "data-points-list" attribute, just clear the canvas.
            ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }
    });
    // CLEAR CANVAS //

    controlDiv.appendChild(drawCanvasClear);
    controlDiv.appendChild(drawCanvasButton);

    return canvasElement;
};
// WORK WITH IMAGES //

// GET IMAGE OR VIDEO BEFORE SEND IN BEFORE //
function retrieveMediaDetails(mediaPreview) {
    const imageElements = mediaPreview.querySelectorAll(".imageMedia");
    const videoElements = mediaPreview.querySelectorAll(".videoMedia");
    const audioElements = mediaPreview.querySelectorAll(".audioMedia");
    let mediaType = "";
    let mediaName = "";
    let mediaBlobUrl = "";
    let mediaStart = 0;
    let mediaEnd = 0;
    let mediaCurrentTime = 0;

    if (imageElements.length > 0) {
        mediaType = "img";
        mediaBlobUrl = imageElements[0].src;
        mediaName = `image_${Date.now()}_${getRandomString(5)}`;
        mediaStart = 0;
        mediaEnd = 0;
        mediaCurrentTime = 0;
    } else if (videoElements.length > 0) {
        mediaType = "video";
        mediaBlobUrl = videoElements[0].src;
        mediaName = `video_${Date.now()}_${getRandomString(5)}`;
        mediaStart = videoElements[0].getAttribute("start");
        mediaEnd = videoElements[0].getAttribute("end");
        mediaCurrentTime = videoElements[0].currentTime;
    } else if (audioElements.length > 0) {
        mediaType = "audio";
        mediaBlobUrl = audioElements[0].src;
        mediaName = `audio_${Date.now()}_${getRandomString(5)}`;
        mediaStart = audioElements[0].getAttribute("start");
        mediaEnd = audioElements[0].getAttribute("end");
        mediaCurrentTime = audioElements[0].currentTime;
    }

    if (mediaBlobUrl) {
        fetch(mediaBlobUrl)
        .then((res) => res.blob())
        .then((blob) => {
          var file = new File([blob], mediaName);
          uploadFile(file);
        });
    }

    return { mediaType, mediaName, mediaBlobUrl, mediaStart, mediaEnd, mediaCurrentTime };
}
// GET IMAGE OR VIDEO BEFORE SEND IN BEFORE //

// CANVAS DRAW AS SET ONE POINT //
function setPointOnCanvas(event) {
    const canvas = event.target;
    const ctx = canvas.getContext('2d');

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Get the coordinates where the user clicked
    const x = event.offsetX;
    const y = event.offsetY;

    // Set the styles for the point
    ctx.fillStyle = 'lightblue';  // Fill color
    ctx.strokeStyle = 'black';   // Border color
    ctx.lineWidth = 3;           // Border thickness
    ctx.shadowColor = 'black';   // Shadow color
    ctx.shadowBlur = 5;          // Shadow blur level
    ctx.shadowOffsetX = 3;       // Horizontal shadow offset
    ctx.shadowOffsetY = 3;       // Vertical shadow offset

    // Set the point
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke(); // Add the border
    // Write point position
    canvas.dataset.pointPosition = JSON.stringify({ x: x, y: y, canvasWidth:canvas.width, canvasHeight:canvas.height });
}

function setMultiplePointsOnCanvas(event) {
    const canvas = event.target;
    const ctx = canvas.getContext('2d');

    // Get the coordinates where the user clicked
    const x = event.offsetX;
    const y = event.offsetY;

    // Determine the color based on the mouse button clicked
    let fillColor;
    if (event.button === 0) {
        fillColor = 'lightblue';  // Left click: light blue
    } else if (event.button === 2) {
        fillColor = 'red';        // Right click: red
    } else {
        return;  // Do nothing for other buttons
    }

    // Retrieve the points list from the canvas's dataset (or initialize it if it doesn't exist)
    let pointsList = canvas.dataset.pointsList ? JSON.parse(canvas.dataset.pointsList) : [];

    // Add the point to the points list
    pointsList.push({ x: x, y: y, canvasWidth:canvas.width, canvasHeight:canvas.height, color: fillColor });

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Redraw all points from the points list
    for (let point of pointsList) {
        ctx.fillStyle = point.color;
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'black';
        ctx.shadowBlur = 5;
        ctx.shadowOffsetX = 3;
        ctx.shadowOffsetY = 3;

        ctx.beginPath();
        ctx.arc(point.x, point.y, 8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke(); // Add the border
    }

    // Store the updated points list in the canvas's dataset
    canvas.dataset.pointsList = JSON.stringify(pointsList);
}

// Get data from canvas to send in backend
function retrieveSelectedPointsList(canvasElement) {
    if (!canvasElement.dataset.pointsList) {
        return null
    }
    return canvasElement ? JSON.parse(canvasElement.dataset.pointsList) : null;
}

function retrieveSelectedFaceData(mediaPreview) {
    const canvasElement = mediaPreview.querySelector(".canvasMedia");
    if (!canvasElement.dataset.pointPosition) {
        return null
    }
    return canvasElement ? JSON.parse(canvasElement.dataset.pointPosition) : null;
}
// CANVAS DRAW AS SET ONE POINT //

function getRandomColor() {
    const colors = ["#ff6af1", "#f7db4d", "#6ae1ff", "#6aff94", "#6ab0ff"];
    const randomIndex = Math.floor(Math.random() * colors.length);
    return colors[randomIndex];
}

//function formatTime(seconds) {
//    const hrs = Math.floor(seconds / 3600);
//    const mins = Math.floor((seconds % 3600) / 60);
//    const secs = Math.floor(seconds % 60);
//    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
//}

function formatTime(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const millis = Math.floor((seconds * 1000) % 1000);  // Extracting milliseconds
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`;
}

//function convertTimeToSeconds(timeStr) {
//    const [hrs, mins, secs] = timeStr.split(":").map(Number);
//    return (hrs * 3600) + (mins * 60) + secs;
//}

function convertTimeToSeconds(timeStr) {
    const [hrsMinsSecs, millis = '0'] = timeStr.split(".");
    const [hrs, mins, secs] = hrsMinsSecs.split(":").map(Number);
    return (hrs * 3600) + (mins * 60) + secs + (Number(millis) / 1000);
}

// CLOSE INTROJS //
function closeTutorial() {
    const tutorialCloseButton = document.querySelector(".introjs-skipbutton");
    tutorialCloseButton.click();
}
// CLOSE INTROJS //

// UPLOAD FILE TO TMP //
function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload_tmp", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Upload failed");
      }
      console.log("File uploaded");
    })
    .catch((error) => {
      console.error(error);
    });
}
// UPLOAD FILE TO TMP //

async function displayMessage(element, message, partition = undefined) {
    const translatedMessage = await translateWithGoogle(message, "auto", targetLang);
    if (partition) {
        element.innerHTML = `${translatedMessage} ${partition}`;
    } else {
        element.innerText = translatedMessage;
    }
    element.style.display = "flex";
    element.style.background = getRandomColor();
}


// RESOLUTION INFORMATION //
async function setVideoResolution(previewElement, maxDeviceResolution, useLimitResolution) {
    const videoElements = previewElement.getElementsByClassName("videoMedia");
    const imageElements = previewElement.getElementsByClassName("imageMedia");
    const spanResolution = previewElement.getElementsByClassName("spanResolution");

    if (imageElements.length === 0 && videoElements.length === 0) {
        console.error("Media element not found");
        return;
    }

    let realWidth;
    let realHeight;
    let resolution;

    if (imageElements.length > 0) {
        realWidth = imageElements[0].naturalWidth;
        realHeight = imageElements[0].naturalHeight;
    } else if (videoElements.length > 0) {
        realWidth = videoElements[0].videoWidth;
        realHeight = videoElements[0].videoHeight;
    }

    resolution = Math.max(realWidth, realHeight);

    if (useLimitResolution && resolution > maxDeviceResolution) {
        if (realWidth > realHeight) {
            realHeight = (maxDeviceResolution / realWidth) * realHeight;
            realWidth = maxDeviceResolution;
        } else {
            realWidth = (maxDeviceResolution / realHeight) * realWidth;
            realHeight = maxDeviceResolution;
        }
    }

    spanResolution[0].innerText = `${realWidth.toFixed(0)} x ${realHeight.toFixed(0)}`;
    spanResolution[0].style.display = "";
}


async function fetchVramResolution(previewElement, useLimitResolution, gpuTable) {
    fetch('/get_vram')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.gpu_vram) {
            console.log('GPU VRAM:', data.gpu_vram, 'GB');

            // Calculate maxDeviceResolution based on VRAM
            let maxDeviceResolution = Math.max(...Object.entries(gpuTable)
                .filter(([key,]) => key <= data.gpu_vram)
                .map(([,val]) => val));

            if (typeof maxDeviceResolution === 'undefined') {
                maxDeviceResolution = Math.min(...Object.values(gpuTable)); // Default to smallest if VRAM is less than any key
            }

            console.log('Max Device Resolution:', maxDeviceResolution);

            setVideoResolution(previewElement, maxDeviceResolution, useLimitResolution);
        } else {
            console.log('GPU VRAM not available');
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
    });
}
// RESOLUTION INFORMATION //