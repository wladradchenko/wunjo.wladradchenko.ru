function sendTextToSpeech() {
    var voiceCardContainers = document.querySelectorAll('.voice-card-container');
    var synthesisTable = document.getElementById("table_body_speech_result");
    var allCardSend = [];

    for (var i = 0; i < voiceCardContainers.length; i++) {
        var toggleVoice = voiceCardContainers[i].querySelector('.toggle-div-voice');

        if (toggleVoice.classList.contains('active')) {
            var textareaText = voiceCardContainers[i].querySelector('.text-input');
            if (textareaText.value) {
                var multiSelectVoice = voiceCardContainers[i].querySelectorAll(".model-checkbox-value")
                var checkedValues = [];

                for (var j = 0; j < multiSelectVoice.length; j++) {
                  if (multiSelectVoice[j].checked) {
                    checkedValues.push(multiSelectVoice[j].value);
                  }
                }
                var pitchFactor = voiceCardContainers[i].querySelector(".pitch-range");
                var speedFactor = voiceCardContainers[i].querySelector(".rate-range");
                var volumeFactor = voiceCardContainers[i].querySelector(".volume-range");

                textareaText.readOnly = true;

                if (checkedValues.length > 0) {
                    var oneCardSend = {
                        "text": textareaText.value,
                        "voice": checkedValues,
                        "rate": speedFactor.value,
                        "pitch": pitchFactor.value,
                        "volume": volumeFactor.value
                    };

                    allCardSend.push(oneCardSend);
                };
            }
        }
    };

    // Get a reference to the #status-message element
    const statusMessage = document.getElementById('status-message');

    if (allCardSend.length > 0) {
        synthesisTable.innerHTML = "";
        statusMessage.innerText = "Подождите... Происходит обработка";

        fetch("/synthesize_speech/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(allCardSend)
        })
    } else {
        statusMessage.innerText = "Выберите голос или включите флаг!";

        setTimeout(() => {
            statusMessage.innerText = "";
        }, 2000);
    }
};

window.addEventListener("DOMContentLoaded", (event) => {
    const buttonRunSynthesis = document.getElementById("button-run-synthesis");
    if (buttonRunSynthesis) {
      buttonRunSynthesis.addEventListener('click', sendTextToSpeech);
    }

    function pollSynthesizedResults() {
        var synthesisTable = document.getElementById("table_body_speech_result");

        fetch("/synthesize_speech_result/")
            .then((response) => {
                if (!response.ok) throw response;
                return response.json();
            })
            .then((response) => {
                //synthesisTable.innerHTML = "";
                response_code = response["response_code"];
                results = response["response"];

                // Get the number of existing rows in the table
                const numExistingRows = synthesisTable.getElementsByTagName("tr").length;
                // Calculate the number of new rows
                numNewRows = results.length - numExistingRows;

                if (response_code === 0 && results && numNewRows > 0) {
                  // Get only the new results
                  newResults = results.slice(-numNewRows);

                  newResults.forEach(function (model_ans, index) {
                    const audioId = `audio-${index + numExistingRows}`;
                    const videoId = `video-windows-${index + numExistingRows}`;
                    const playBtnId = `play-${index + numExistingRows}`;
                    const pauseBtnId = `pause-${index + numExistingRows}`;
                    const downloadBtnId = `download-${index + numExistingRows}`;

                    synthesisTable.insertAdjacentHTML(
                      "beforeend",
                      `
                      <tr style="height: 40pt;text-align: center;">
                        <td style="width: 33%;">
                          <div class="buttons" style="justify-content: center;">
                            <button id="${playBtnId}" style="width: 30pt;height: 30pt;display:inline;margin-left:0 !important;margin-right:0 !important"><i class="fa fa-play"></i><i style="display: none;" class="fa fa-pause"></i></button>
                            <a style="margin-left: 5pt;margin-right: 5pt;" href="${model_ans.response_audio_url}" download="audio.wav">
                                <button class="download" style="width: 30pt;height: 30pt;margin-left:0 !important;margin-right:0 !important"><i class="fa fa-download"></i></button>
                            </a>
                            <button id="${videoId}" onclick="deepfakeGeneralPop(event.target, '${model_ans.response_audio_url}', '${model_ans.recognition_text}');" class="synthesis-video" style="width: 30pt;height: 30pt;margin-left:0 !important;margin-right:0 !important"><i class="fa fa-solid fa-film"></i></button>
                          </div>
                          <audio id="${audioId}" style="display:none;" controls preload="none">
                            <source src="${model_ans.response_audio_url}" type="audio/wav">
                            Your browser does not support audio.
                          </audio>
                        </td>
                        <td style="width: 25%;">${model_ans.recognition_text.length > 10 ? model_ans.recognition_text.slice(0, 10) + "..." : model_ans.recognition_text}</td>
                        <td style="width: 20%;">${model_ans.voice}</td>
                      </tr>
                      `
                    );

                    const playBtn = document.getElementById(playBtnId);
                    const audio = document.getElementById(audioId);

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

                  });
                } else if (!results || numNewRows < 1) {
                    //
                } else {
                    alert("Error: " + response);
                }
                var textareaTextAll = document.querySelectorAll('.text-input');
                for (var k = 0; k < textareaTextAll.length; k++) {
                  textareaTextAll[k].readOnly = false;
                }
            })
            .catch((err) => {
                console.log(err);
            });
    };
    // function for deepfake
    function pollSynthesizedDeepfakeResults() {
        var synthesisDeepfakeTable = document.getElementById("table_body_deepfake_result");

        fetch("/synthesize_deepfake_result/")
            .then((response) => {
                if (!response.ok) throw response;
                return response.json();
            })
            .then((response) => {
                response_code = response["response_code"];
                results = response["response"];

                // Get the number of existing rows in the table
                const numExistingRows = synthesisDeepfakeTable.getElementsByTagName("tr").length;
                // Calculate the number of new rows
                numNewRows = results.length - numExistingRows;

                if (response_code === 0 && results && numNewRows > 0) {
                  // Get only the new results
                  newResults = results.slice(-numNewRows);

                  newResults.forEach(function (model_ans, index) {
                    const videoID = `video-${index + numExistingRows}`;
                    const playBtnId = `video-play-${index + numExistingRows}`;
                    const pauseBtnId = `video-pause-${index + numExistingRows}`;
                    const downloadBtnId = `video-download-${index + numExistingRows}`;

                    synthesisDeepfakeTable.insertAdjacentHTML(
                      "beforeend",
                      `
                      <tr style="height: 40pt;">
                        <td>
                          <div class="buttons" style="justify-content: center;">
                            <button id="${playBtnId}" style="width: 30pt;height: 30pt;display:inline;"><i class="fa fa-play"></i><i style="display: none;" class="fa fa-pause"></i></button>
                            <a href="${model_ans.response_video_url}" download="video.mp4">
                                <button class="download" style="width: 30pt;height: 30pt;"><i class="fa fa-download"></i></button>
                            </a>
                          </div>
                        </td>
                        <td style="text-align: center;">${model_ans.response_video_date}</td>
                      </tr>
                      `
                    );

                    const playBtn = document.getElementById(playBtnId);
                    playBtn.addEventListener("click", function() {
                        var introVideoDeepfake = introJs();
                        introVideoDeepfake.setOptions({
                            steps: [
                                {
                                    element: playBtn,
                                    title: "Результат синтеза",
                                    position: 'left',
                                    intro: `<div><video style="border: 2px dashed #000;" id="${videoID}" width="250" height="auto" controls>
                                              <source src="${model_ans.response_video_url}" type="video/mp4">
                                            Your browser does not support the video tag.
                                            </video></div>`,
                                }
                            ],
                              showButtons: false,
                              showStepNumbers: false,
                              showBullets: false,
                              nextLabel: 'Продолжить',
                              prevLabel: 'Вернуться',
                              doneLabel: 'Закрыть'
                        });
                        introVideoDeepfake.start();
                    });
                  });
                } else if (!results || numNewRows < 1) {
                    //
                } else {
                    alert("Error: " + response);
                }
                var textareaTextAll = document.querySelectorAll('.text-input');
                for (var k = 0; k < textareaTextAll.length; k++) {
                  textareaTextAll[k].readOnly = false;
                }
            })
            .catch((err) => {
                console.log(err);
            });
    };

    const buttonTurnOnSpeechSynthesisWindows = document.getElementById("button-show-animation-window");
    if (buttonTurnOnSpeechSynthesisWindows) {
      buttonTurnOnSpeechSynthesisWindows.addEventListener('click', pollSynthesizedResults);
    }

    const buttonTurnOnAnimationSynthesisWindows = document.getElementById("button-show-voice-window");
    if (buttonTurnOnAnimationSynthesisWindows) {
      buttonTurnOnAnimationSynthesisWindows.addEventListener('click', pollSynthesizedDeepfakeResults);
    }

    var intervalSynthesizedResultsId;

    // Get a reference to the #status-message element
    const statusMessage = document.getElementById('status-message');

    // Define the URL of the /synthesize_process/ page
    const synthesizeProcessUrl = '/synthesize_process/';

    // Define a function to fetch the /synthesize_process/ page and check the status
    function checkStatus() {
      // Fetch the /synthesize_process/ page
      fetch(synthesizeProcessUrl)
        .then(response => response.json())
        .then(data => {
          // Check the value of status_code
          if (data.status_code === 200) {
            if (!intervalSynthesizedResultsId) {
                setTimeout(pollSynthesizedResults, 5000);
                setTimeout(pollSynthesizedDeepfakeResults, 5000);
                intervalSynthesizedResultsId = undefined;
            }
            // If status_code is 200, hide the #status-message element
            statusMessage.innerText = data.message;
            setTimeout(checkStatus, 5000);  // 5 seconds
            buttonRunSynthesis.disabled = false;
          } else {
            setTimeout(pollSynthesizedResults, 5000);
            setTimeout(pollSynthesizedDeepfakeResults, 5000);
            intervalSynthesizedResultsId = data.status_code;
            // If status_code is not 200, wait for the next interval to check again
            statusMessage.innerText = data.message;
            setTimeout(checkStatus, 5000);  // 5 seconds
            buttonRunSynthesis.disabled = true;
          }
        })
        .catch(error => {
          // If an error occurs, log it to the console and wait for the next interval to check again
          console.error(error);
          setTimeout(checkStatus, 5000);
        });
    }

    // Start checking the status
    checkStatus();
});
