async function sendTextToSpeech() {
  var voiceCardContainers = document.querySelectorAll(".voice-card-container");
  var synthesisTable = document.getElementById("table_body_result");
  var allCardSend = [];

  for (var i = 0; i < voiceCardContainers.length; i++) {
    var toggleVoice = voiceCardContainers[i].querySelector(".toggle-div-voice");

    if (toggleVoice.classList.contains("active")) {
      var textareaText = voiceCardContainers[i].querySelector(".text-input");
      if (textareaText.value) {
        var multiSelectVoice = voiceCardContainers[i].querySelectorAll(
          ".model-checkbox-value"
        );
        var checkedValues = [];

        for (var j = 0; j < multiSelectVoice.length; j++) {
          if (multiSelectVoice[j].checked) {
            checkedValues.push(multiSelectVoice[j].value);
          }
        }
        var pitchFactor = voiceCardContainers[i].querySelector(".pitch-range");
        var speedFactor = voiceCardContainers[i].querySelector(".rate-range");
        var volumeFactor = voiceCardContainers[i].querySelector(".volume-range");
        var settingTranslation = voiceCardContainers[i].querySelector(".setting-tts");
        var autoTranslation = settingTranslation.getAttribute("automatic-translate") === "true";
        var langTranslation = settingTranslation.getAttribute("value-translate");
        var useVoiceCloneOnAudio = settingTranslation.getAttribute("voice-audio-clone") === "true";
        var voiceCloneBlobUrl = settingTranslation.getAttribute("blob-audio-src");

        var voiceCloneName = "";

        if (useVoiceCloneOnAudio) {
          if (voiceCloneBlobUrl === "") {
            useVoiceCloneOnAudio = false;
            console.warn("Voice clone name is empty. Disabling voice clone.");
          } else {
            voiceCloneName = "rtvc_audio_" + Date.now() + "_" + getRandomString(5);
            console.log(voiceCloneName)
            try {
              const res = await fetch(voiceCloneBlobUrl);
              const blob = await res.blob();
              var file = new File([blob], voiceCloneName);
              uploadFile(file, voiceCloneName);
            } catch (error) {
              console.error("An error occurred while fetching the voice clone blob:", error);
            }
          }
        }

        if (checkedValues.length > 0 || useVoiceCloneOnAudio === true) {
          var oneCardSend = {
            text: textareaText.value,
            voice: checkedValues,
            rate: speedFactor.value,
            pitch: pitchFactor.value,
            volume: volumeFactor.value,
            auto_translation: autoTranslation,
            lang_translation: langTranslation,
            use_voice_clone_on_audio: useVoiceCloneOnAudio,
            rtvc_audio_clone_voice: voiceCloneName,
          };

          allCardSend.push(oneCardSend);
        }
      }
    }
  }
  // Call the async function
  try {
    await processAsyncSynthesis(allCardSend, synthesisTable);
    console.log("Start to fetch msg for voice");
  } catch (error) {
    console.log("Error to fetch msg for voice");
    console.log(error);
  }
}

async function processAsyncSynthesis(allCardSend, synthesisTable) {
  if (allCardSend.length > 0) {
    console.log("/synthesize_speech/")
    console.log(JSON.stringify(allCardSend, null, 4));

    await fetch("/synthesize_speech/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(allCardSend),
    });
  } else {
    sendPrintToBackendConsole("Choose your voice or turn on the flag!");
  }
}

window.addEventListener("DOMContentLoaded", (event) => {
  const buttonRunSynthesis = document.getElementById("button-run-synthesis");
  if (buttonRunSynthesis) {
    buttonRunSynthesis.addEventListener("click", sendTextToSpeech);
  }

  function pollGeneralSynthesizedResults() {
    var synthesisTable = document.getElementById("table_body_result");
    fetch("/synthesize_result/")
      .then((response) => {
        if (!response.ok) throw response;
        return response.json();
      })
      .then((response) => {
        response_code = response["response_code"];
        results = response["response"];
        // Get the number of existing rows in the table
        const numExistingRows = synthesisTable.querySelectorAll("tr").length;
        // Calculate the number of new rows
        numNewRows = results.length - numExistingRows;

        if (response_code === 0 && results && numNewRows > 0) {
          // Get only the new results
          newResults = results.slice(-numNewRows);
          newResults.forEach(function (model_ans, index) {
            const videoID = `video-${index + numExistingRows}`;
            const audioId = `audio-${index + numExistingRows}`;
            const playBtnId = `play-${index + numExistingRows}`;
            const pauseBtnId = `pause-${index + numExistingRows}`;
            const downloadBtnId = `download-${index + numExistingRows}`;
            // Get the existing span element by its ID
            const existingSpans = synthesisTable.querySelectorAll(".dataResponseResult");
            let duplicateFound = false;

            // Loop through each existing span to check if there's a duplicate
            existingSpans.forEach(span => {
                if (span.textContent === model_ans.request_date) {
                    duplicateFound = true;
                }
            });

            // If no duplicate found, insert the new span
            if (!duplicateFound) {
                const spanHTML = `<span class="dataResponseResult" style="padding: 15px;font-size: 12px;">${model_ans.request_date}</span>`;
                synthesisTable.insertAdjacentHTML("beforeend", spanHTML);
            }

            if (model_ans.request_mode === "speech") {
                 synthesisTable.insertAdjacentHTML(
                  "beforeend",
                          `<tr data-type="speech" style="text-align: center;font-size: 12px;">
                            <td class="notranslate" style="display: flex;flex-direction: row;align-items: center;padding-left: 10pt;padding-right: 10pt;">
                              <div class="marquee-container" style="background: #f7db4d;box-shadow: rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px;">
                                <span class="notranslate marquee" style="width: 80pt;padding-left: 10px;padding-right: 10px;">${model_ans.voice}</span>
                              </div>
                              <div class="buttons" style="width: 70pt;justify-content: center;scale:0.8;">
                                <button id="${playBtnId}" style="width: 30pt;height: 30pt;display:inline;margin-left:0 !important;margin-right:0 !important"><i class="fa fa-play"></i><i style="display: none;" class="fa fa-pause"></i></button>
                                <a style="margin-left: 5pt;margin-right: 5pt;" href="${
                                  model_ans.response_url
                                }" download="audio.wav">
                                    <button class="download" style="width: 30pt;height: 30pt;margin-left:0 !important;margin-right:0 !important"><i class="fa fa-download"></i></button>
                                </a>
                              </div>
                              <audio id="${audioId}" style="display:none;" controls preload="none">
                                <source src="${
                                  model_ans.response_url
                                }" type="audio/wav">
                                Your browser does not support audio.
                              </audio>
                            ${
                              model_ans.request_information.length > 20
                                ? model_ans.request_information.slice(0, 20) + "..."
                                : model_ans.request_information
                            }
                            </td>
                          </tr>
                          `
                );

                // Transition of label
                applyMarqueeTransition(synthesisTable);

                const playBtn = document.getElementById(playBtnId);
                const audio = document.getElementById(audioId);

                playBtn.addEventListener("click", function () {
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

                audio.addEventListener("ended", function () {
                  playBtn.children[0].style.display = "inline";
                  playBtn.children[1].style.display = "none";
                });
            } else if (model_ans.request_mode === "deepfake") {
                synthesisTable.insertAdjacentHTML(
                  "beforeend",
                  `
                  <tr data-type="deepfake" style="text-align: center; font-size: 12px;">
                    <td class="notranslate" style="display: flex;flex-direction: row;align-items: center;padding-left: 10pt;padding-right: 10pt;">
                      <div class="marquee-container" style="background: #f7db4d;box-shadow: rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px;">
                        <span class="notranslate marquee" style="width: 80pt;padding-left: 10px;padding-right: 10px;">${model_ans.mode}</span>
                      </div>
                      <div class="buttons" style="width: 70pt;justify-content: center;scale:0.8;">
                        <button id="${playBtnId}" style="width: 30pt;height: 30pt;display:inline;margin-left:0 !important;margin-right:0 !important"><i class="fa fa-play"></i><i style="display: none;" class="fa fa-pause"></i></button>
                        <a href="${model_ans.response_url}" download="video.mp4" style="margin-left: 5pt;margin-right: 5pt;">
                            <button class="download" style="width: 30pt;height: 30pt;margin-left:0 !important;margin-right:0 !important"><i class="fa fa-download"></i></button>
                        </a>
                      </div>
                    ${
                      model_ans.request_information.length > 20
                        ? model_ans.request_information.slice(0, 20) + "..."
                        : model_ans.request_information
                    }
                  </tr>
                  `
                );

                // Transition of label
                applyMarqueeTransition(synthesisTable);

                // Check the file extension to determine the media type
                let mediaURLElementResult = model_ans.response_url;
                let extensionElementResult = mediaURLElementResult.split(".").pop();
                let mediaPlayElementResult = "<div>Unsupported media format.</div>";
                if (
                  ["mp4", "avi", "mkv", "mov", "flv", "wmv"].includes(
                    extensionElementResult
                  )
                ) {
                  mediaPlayElementResult = `
                            <div>
                              <video style="border: 2px dashed #000;" id="${videoID}" width="400" height="auto" controls>
                                <source src="${mediaURLElementResult}" type="video/${extensionElementResult}">
                                Your browser does not support the video tag.
                              </video>
                            </div>
                          `;
                } else if (
                  ["jpg", "jpeg", "png", "gif"].includes(extensionElementResult)
                ) {
                  mediaPlayElementResult = `
                            <div>
                              <img src="${mediaURLElementResult}" id="${videoID}" style="border: 2px dashed #000;" width="250" height="auto" />
                            </div>
                          `;
                }

                const playBtn = document.getElementById(playBtnId);
                playBtn.addEventListener("click", function () {
                  var introVideoDeepfake = introJs();
                  introVideoDeepfake.setOptions({
                    steps: [
                      {
                        element: document.getElementsByClassName("synthesized_field")[0],
                        position: "left",
                        intro: `${mediaPlayElementResult}`,
                      },
                    ],
                    showButtons: false,
                    showStepNumbers: false,
                    showBullets: false,
                    nextLabel: "Next",
                    prevLabel: "Back",
                    doneLabel: "Close",
                  });
                  introVideoDeepfake.setOption('keyboardNavigation', false).start();
                });
            };
           });
            };
        });
  };

  // Define the URL of the /synthesize_process/ page
  const synthesizeProcessUrl = "/synthesize_process/";

  // Define a function to fetch the /synthesize_process/ page and check the status
  function checkStatus() {
    // Fetch the /synthesize_process/ page
    fetch(synthesizeProcessUrl)
      .then((response) => response.json())
      .then((data) => {
        // Check the value of status_code
        if (data.status_code === 200) {
          // If status_code is 200, hide the #status-message element
          document.getElementById("lds-ring").style.display = "none"; // Indicate loading
          setTimeout(checkStatus, 2000); // 5 seconds
          buttonRunSynthesis.disabled = false;
        } else {
          // If status_code is not 200, wait for the next interval to check again
          document.getElementById("lds-ring").style.display = "flex";
          setTimeout(checkStatus, 2000); // 5 seconds
          buttonRunSynthesis.disabled = true;
        }
      })
      .catch((error) => {
        // If an error occurs, log it to the console and wait for the next interval to check again
        console.error(error);
        setTimeout(checkStatus, 5000);
      });
  }

  // Start checking the status
  checkStatus();
  setInterval(pollGeneralSynthesizedResults, 2000);
});


function applyMarqueeTransition(synthesisTable) {
    let lastInsertedRow = synthesisTable.lastElementChild;
    let marqueeElement = lastInsertedRow.querySelector('.marquee');

    // Temporarily remove fixed width to get accurate content width
    marqueeElement.style.width = 'auto';
    let textWidth = marqueeElement.offsetWidth;

    if (textWidth > 100) {
        // Reapply fixed width (since it was set to 'auto' temporarily)
        marqueeElement.style.width = '120px';

        // Dynamically create the marquee animation with the acquired text width
        let animationName = `marquee${Date.now()}`;  // Unique animation name to prevent conflicts
        let animationCSS = `
            @keyframes ${animationName} {
                0%   { transform: translateX(0%); }
                25%  { transform: translateX(0%); }
                50%  { transform: translateX(calc(-1 * ${textWidth}px + 100px)); }
                75%  { transform: translateX(0%); }
                100% { transform: translateX(0%); }
            }
        `;

        // Create a style element specifically for this marquee
        let style = document.createElement('style');
        style.innerHTML = animationCSS;

        // Attach this style to the marquee element itself
        marqueeElement.appendChild(style);

        // Apply the newly created animation to the marquee element
        marqueeElement.style.animation = `${animationName} 10s linear infinite`;
    }
}
