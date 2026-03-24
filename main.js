import { SenseEngine } from './engine.js'
import { Dashboard } from './dashboard.js'

const video = document.getElementById('camera')
const overlay = document.getElementById('overlay')
const statusDot = document.getElementById('status-dot')

let currentEngine = null
let currentStream = null

async function enumerateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices()
  return devices.filter(d => d.kind === 'videoinput')
}

function populateSelect(selectEl, cameras, currentId) {
  selectEl.innerHTML = cameras.map((cam, i) =>
    `<option value="${cam.deviceId}" ${cam.deviceId === currentId ? 'selected' : ''}>${cam.label || `摄像头 ${i + 1}`}</option>`
  ).join('')
}

async function startCamera(deviceId) {
  // Stop previous
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop())
  }

  const constraints = {
    video: {
      width: 640, height: 480,
      ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' })
    }
  }

  const stream = await navigator.mediaDevices.getUserMedia(constraints)
  currentStream = stream
  video.srcObject = stream
  await video.play()

  overlay.width = video.videoWidth || 640
  overlay.height = video.videoHeight || 480

  return stream
}

async function init() {
  const startBtn = document.getElementById('start-btn')
  const startOverlay = document.getElementById('start-overlay')
  const app = document.getElementById('app')
  const startSelect = document.getElementById('camera-select')
  const hudSelect = document.getElementById('camera-switch')

  // Get camera list
  try {
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true })
    tempStream.getTracks().forEach(t => t.stop())

    const cameras = await enumerateCameras()
    populateSelect(startSelect, cameras)
    populateSelect(hudSelect, cameras)
  } catch (e) {
    startSelect.innerHTML = '<option value="">默认摄像头</option>'
  }

  // Wait for start
  await new Promise(resolve => {
    startBtn.addEventListener('click', resolve, { once: true })
  })

  const selectedDeviceId = startSelect.value
  startOverlay.style.display = 'none'
  app.style.display = 'block'

  try {
    await startCamera(selectedDeviceId)

    // Sync HUD select
    if (hudSelect.value !== selectedDeviceId) {
      hudSelect.value = selectedDeviceId
    }

    // Init engine
    currentEngine = new SenseEngine(video, overlay)
    await currentEngine.init()

    const dashboard = new Dashboard()

    // Loop
    function loop() {
      const result = currentEngine.detect()
      if (result) {
        dashboard.update(result)
        // Press 's' to dump sense frame to console
        if (window.__dumpSense && result.sense) {
          console.log('SenseFrame:', JSON.stringify(result.sense, null, 2))
          window.__dumpSense = false
        }
      }
      requestAnimationFrame(loop)
    }
    requestAnimationFrame(loop)

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.key === 's') window.__dumpSense = true
    })

    // HUD camera switch — runtime
    hudSelect.addEventListener('change', async () => {
      const newId = hudSelect.value
      try {
        await startCamera(newId)
        // Engine keeps working — it reads from the same video element
        console.log('Switched camera:', newId)
      } catch (e) {
        console.error('Camera switch failed:', e)
      }
    })

  } catch (e) {
    console.error('Init failed:', e)
    document.getElementById('synthesis-text').textContent = `错误: ${e.message}`
    statusDot.style.background = 'var(--red)'
  }
}

init()
