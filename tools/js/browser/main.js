const {
  app,
  screen,
  BrowserWindow,
  ipcMain,
  Notification,
  session,
} = require("electron");
const path = require("node:path");

function createWindow(url) {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  const options = {
    backgroundColor: "#0d0d0d00",
    width: width,
    height: height,
    title: "Wunjo",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nativeWindowOpen: false,
      nodeIntegration: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  };

  if (process.platform === "linux") {
    options.icon = path.join(`${__dirname}/64x64.png`);
  }

  const mainWindow = new BrowserWindow(options);

  mainWindow.setMenu(null);
  mainWindow.loadURL(url);

  // Debounce the resize event
  let resizeTimeout;
  mainWindow.on("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      mainWindow.webContents.send("resize");
    }, 500);
  });

  // Memory management
  setInterval(() => {
    const memoryUsage = process.getProcessMemoryInfo();
    if (memoryUsage.privateBytes > 150 * 1024 * 1024) {
      console.log("Memory usage high, triggering garbage collection");
      global.gc();
    }
  }, 60000);

  // Download process monitoring
  session.defaultSession.on("will-download", (event, item, webContents) => {
    // Create a custom notification popup window
    const notificationWindow = new BrowserWindow({
      width: 350,
      height: 100,
      frame: false,
      alwaysOnTop: true,
      transparent: true,
      x: Math.floor(width - 20), // Centered horizontally
      y: 20, // At the top of the screen
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: false,
      },
    });

    notificationWindow.loadURL(
      "data:text/html;charset=utf-8," +
        encodeURIComponent(`
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body {
            background-color: rgba(0, 0, 0, 0);
            padding: 0;
            height:100%;
            width:100%;
            overflow: hidden;
            font-size: medium;
            margin: 10px;
            color: white;
          }
          .notification {
            display:flex;
            justify-content: space-between;
            padding-left: 20px;
            padding-right: 20px;
            text-align: center;
            background-color: #282828;
            border:1px solid #4d4d4d;
            align-items: center;
            border-radius: 8px;
            scale: .8;
            font-weight: lighter;
          }
          button, input, optgroup, select, textarea {
            font-family: inherit;
            font-size: inherit;
            font-weight: inherit;
            margin: 0;
            padding: 0;
            border: none;
            background: none;
            text-align: inherit;
          }
          button {
            color: black;
            background-color:  #c8edd2;
            border-radius: 8px;
            padding-left: 15px;
            padding-right: 15px;
            padding-top: 5px;
            padding-bottom: 5px;
            font-weight: bold;
          }
        </style>
      </head>
      <body>
        <div class="notification">
          <h3>Downloading: 0 Mb</h3>
          <button class="cancel">Cancel</button>
        </div>
        <script>
          const { ipcRenderer } = require('electron');
          const cancelButton = document.querySelector('.cancel');
          
          cancelButton.addEventListener('click', () => {
            ipcRenderer.send('cancel-download');
          });
          
          ipcRenderer.on('update-download', (event, progress) => {
            document.querySelector('h3').innerText = 'Downloading: ' + progress + ' Mb';
          });
        </script>
      </body>
      </html>
    `)
    );

    let downloadStarted = false;
    let progress = 0;

    item.on("updated", (event, state) => {
      if (state === "progressing") {
        if (!downloadStarted) {
          notificationWindow.show();
          downloadStarted = true;
        }
        progress = Math.round(item.getReceivedBytes() / 1024 / 1024);
        notificationWindow.webContents.send("update-download", progress);
      }
    });

    item.on("done", (event, state) => {
      if (state === "completed") {
        notificationWindow.webContents.send(
          "update-download",
          "Download completed!"
        );
      } else {
        notificationWindow.webContents.send(
          "update-download",
          "Download failed!"
        );
      }
      notificationWindow.close();
    });

    ipcMain.on("cancel-download", () => {
      item.cancel();
      notificationWindow.webContents.send(
        "update-download",
        "Download canceled!"
      );
      notificationWindow.close();
    });

    item.setSavePath(path.join(app.getPath("downloads"), item.getFilename()));
  });

  return mainWindow;
}

// Set a very high memory limit
app.commandLine.appendSwitch("js-flags", "--max-old-space-size=20480");
// Disables
app.disableHardwareAcceleration(); // Disable Hardware Acceleration
app.commandLine.appendSwitch("disable-gpu-rasterization");
//app.commandLine.appendSwitch('disable-extensions'); // Disable all extensions
app.commandLine.appendSwitch("disable-software-rasterizer"); // Disable software rasterizer
app.commandLine.appendSwitch("enable-logging"); // Enable logging for debugging

app.whenReady().then(() => {
  const args = process.argv.slice(2);
  let url = "http:/127.0.0.1:48000";

  args.forEach((arg) => {
    if (arg.startsWith("--app=")) {
      url = arg.split("=")[1];
    }
  });

  createWindow(url);

  app.on("activate", function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow(url);
  });
});

app.on("window-all-closed", function () {
  if (process.platform !== "darwin") app.quit();
});

// Enable garbage collection (only in --inspect mode)
if (process.argv.includes("--inspect")) {
  global.gc();
}
