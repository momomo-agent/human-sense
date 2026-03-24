import { SenseEngine } from './engine.js'
import { Dashboard } from './dashboard.js'

const video = document.getElementById('camera')
const overlay = document.getElementById('overlay')
const statusDot = document.getElementById('status-dot')

async function init() {
  const startBtn = document.getElementById('start-btn')
  const startOverlay = document.getElementById('start-overlay')
  const app = document.getElementById('app')
  const cameraSelect = document.getElementById('camera-select')

  // Enumerate cameras — need a temp stream first to get labels
  try {
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true })
    tempStream.getTracks().forEach(t => t.stop())

    const devices = await navigator.mediaDevices.enumerateDevices()
    const cameras = devices.filter(d => d.kind === 'videoinput')

    cameraSelect.innerHTML = cameras.map((cam, i) =>
      `<option value="${cam.deviceId}">${cam.label || `摄像头 ${i + 1}`}</option>`
    ).join('')
  } catch (e) {
    cameraSelect.innerHTML = '<option value="">默认摄像头</option>'
  }

  // Wait for start
  await new Promise(resolve => {
    startBtn.addEventListener('click', resolve, { once: true })
  })

  const selectedDeviceId = cameraSelect.value
  startOverlay.style.display = 'none'
  app.style.display = 'block'

  try {
    const constraints = {
      video: {
        width: 640, height: 480,
        ...(selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : { facingMode: 'user' })
      }
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    video.srcObject = stream
    await video.play()

    overlay.width = video.videoWidth || 640
    overlay.height = video.videoHeight || 480

    const engine = new SenseEngine(video, overlay)
    await engine.init()

    const dashboard = new Dashboard()

    function loop() {
      const result = engine.detect()
      if (result) dashboard.update(result)
      requestAnimationFrame(loop)
    }

    requestAnimationFrame(loop)

  } catch (e) {
    console.error('Init failed:', e)
    document.getElementById('synthesis-text').textContent = `错误: ${e.message}`
    statusDot.style.background = 'var(--red)'
  }
}

init()
