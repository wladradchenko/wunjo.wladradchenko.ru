
/// CONSOLE UPDATE LOGICAL ///
function updateConsoleLog() {
    fetch('/console_log', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())  // Parse JSON data
    .then(data => {
        // Reverse the array elements
        const reversedData = data.reverse();
        // Remove empty lines
        const nonEmptyData = reversedData.filter(line => line.trim() !== "");
        // Join the reversed array elements into a single string separated by newlines
        const logText = nonEmptyData.join('\n');
        // Update the element with id="console-log" with the logs
        document.getElementById('console-log').innerText = logText;
    })
    .catch(error => {
        console.error('Log field deleted, message from setInterval:', error);
    });
}


// Update console log initially
updateConsoleLog();

// Update console log every 5 seconds
consoleBackendLogSetInterval = setInterval(updateConsoleLog, 1000);  // It is ok if check each one second?
/// CONSOLE UPDATE LOGICAL ///


/// INFORMATION USER ABOUT MISTAKE ///
function sendPrintToBackendConsole(message) {
    fetch('/console_log_print', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "print": message
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 200) {
            console.log("Successfully sent print message to backend.");
            updateConsoleLog();
        } else {
            console.error("Failed to send print message to backend.");
        }
    })
    .catch(error => {
        console.error("An error occurred:", error);
    });
}
/// INFORMATION USER ABOUT MISTAKE ///