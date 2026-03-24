/**
 * SenseEngine — face detection + feature extraction
 * 
 * Strategy: Start with Chrome FaceDetector API (zero deps, fast),
 * then layer MediaPipe Face Mesh on top for detailed landmarks.
 * 
 * Phase 1 (now): FaceDetector API for presence + basic orientation
 * Phase 2: MediaPipe Face Mesh for gaze, blink, expression
 * Phase 3: MediaPipe Hands for gesture
 */

// ---- EMA utility ----
class EMA {
  constructor(alpha = 0.3) {
    this.alpha = alpha
    this.value = null
  }
  update(v) {
    if (this.value === null) { this.value = v; return v }
    this.value = this.value * (1 - this.alpha) + v * this.alpha
    return this.value
  }
  get() { return this.value }
}

// ---- Blink detector ----
class BlinkDetector {
  constructor() {
    this.history = []       // timestamps of blinks
    this.wasOpen = true
    this.threshold = 0.2    // EAR threshold for "closed"
  }

  // Eye Aspect Ratio from landmarks
  // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
  computeEAR(eyeLandmarks) {
    if (!eyeLandmarks || eyeLandmarks.length < 6) return 1
    const [p1, p2, p3, p4, p5, p6] = eyeLandmarks
    const vertical1 = Math.hypot(p2.x - p6.x, p2.y - p6.y)
    const vertical2 = Math.hypot(p3.x - p5.x, p3.y - p5.y)
    const horizontal = Math.hypot(p1.x - p4.x, p1.y - p4.y)
    return (vertical1 + vertical2) / (2 * horizontal + 0.001)
  }

  update(ear) {
    const isOpen = ear > this.threshold
    if (this.wasOpen && !isOpen) {
      this.history.push(Date.now())
      // Keep last 60s
      const cutoff = Date.now() - 60000
      this.history = this.history.filter(t => t > cutoff)
    }
    this.wasOpen = isOpen
    return this.getRate()
  }

  getRate() {
    const cutoff = Date.now() - 60000
    const recent = this.history.filter(t => t > cutoff)
    return recent.length  // blinks per minute
  }
}

// ---- Expression classifier ----
class ExpressionClassifier {
  // From face mesh landmarks, compute basic expressions
  classify(landmarks) {
    if (!landmarks || landmarks.length < 468) return { expression: '未知', confidence: 0 }

    // Mouth openness: distance between upper and lower lip
    const upperLip = landmarks[13]   // upper lip center
    const lowerLip = landmarks[14]   // lower lip center
    const mouthLeft = landmarks[61]
    const mouthRight = landmarks[291]

    const mouthOpen = Math.abs(upperLip.y - lowerLip.y)
    const mouthWidth = Math.abs(mouthLeft.x - mouthRight.x)
    const mouthRatio = mouthOpen / (mouthWidth + 0.001)

    // Eyebrow raise: distance between eyebrow and eye
    const leftBrow = landmarks[66]    // left eyebrow inner
    const leftEye = landmarks[159]    // left eye upper
    const browEyeDist = Math.abs(leftBrow.y - leftEye.y)

    // Mouth corner vs center — smile detection
    const mouthCenter = (upperLip.y + lowerLip.y) / 2
    const leftCorner = landmarks[61]
    const rightCorner = landmarks[291]
    const cornerAvgY = (leftCorner.y + rightCorner.y) / 2
    const smileScore = mouthCenter - cornerAvgY  // positive = corners higher = smile

    if (mouthRatio > 0.4) return { expression: '😮 惊讶/说话', confidence: 0.7 }
    if (smileScore > 0.01) return { expression: '😊 微笑', confidence: 0.6 }
    if (browEyeDist > 0.045) return { expression: '🤔 思考/困惑', confidence: 0.5 }
    return { expression: '😐 平静', confidence: 0.5 }
  }
}

// ---- Head pose estimator ----
class HeadPoseEstimator {
  estimate(landmarks) {
    if (!landmarks || landmarks.length < 468) {
      return { yaw: 0, pitch: 0, roll: 0, facing: true, posture: '未知', tilt: '正' }
    }

    // Use nose tip (1), left face edge (234), right face edge (454)
    const nose = landmarks[1]
    const leftFace = landmarks[234]
    const rightFace = landmarks[454]
    const forehead = landmarks[10]
    const chin = landmarks[152]

    // Yaw: nose position relative to face center
    const faceCenter = (leftFace.x + rightFace.x) / 2
    const faceWidth = Math.abs(rightFace.x - leftFace.x)
    const yaw = (nose.x - faceCenter) / (faceWidth + 0.001)  // -1..1

    // Pitch: nose position relative to vertical face center
    const faceMidY = (forehead.y + chin.y) / 2
    const faceHeight = Math.abs(chin.y - forehead.y)
    const pitch = (nose.y - faceMidY) / (faceHeight + 0.001)  // -1..1

    // Roll: angle of eye line
    const leftEye = landmarks[33]
    const rightEye = landmarks[263]
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180 / Math.PI

    // Derived
    const facing = Math.abs(yaw) < 0.15 && Math.abs(pitch) < 0.2
    let posture = '正坐'
    if (pitch < -0.15) posture = '前倾'
    else if (pitch > 0.15) posture = '后仰'
    if (Math.abs(yaw) > 0.25) posture = '侧头'

    let tilt = '正'
    if (roll > 8) tilt = '右倾'
    else if (roll < -8) tilt = '左倾'

    return { yaw, pitch, roll, facing, posture, tilt }
  }
}

// ---- Gaze estimator (from head pose + eye landmarks) ----
class GazeEstimator {
  estimate(landmarks) {
    if (!landmarks || landmarks.length < 468) return { region: '未知', looking: false }

    // Iris landmarks (if available): 468-477
    // Left iris center: 468, Right iris center: 473
    const hasIris = landmarks.length >= 478

    let gazeX = 0, gazeY = 0

    if (hasIris) {
      // Left eye: corners 33, 133; iris center 468
      const le33 = landmarks[33], le133 = landmarks[133], liris = landmarks[468]
      const leWidth = Math.abs(le133.x - le33.x)
      const lGazeX = (liris.x - le33.x) / (leWidth + 0.001)

      // Right eye: corners 362, 263; iris center 473
      const re362 = landmarks[362], re263 = landmarks[263], riris = landmarks[473]
      const reWidth = Math.abs(re263.x - re362.x)
      const rGazeX = (riris.x - re362.x) / (reWidth + 0.001)

      gazeX = (lGazeX + rGazeX) / 2 - 0.5  // center at 0
      
      // Vertical
      const le159 = landmarks[159], le145 = landmarks[145]
      const leH = Math.abs(le145.y - le159.y)
      const lGazeY = (liris.y - le159.y) / (leH + 0.001) - 0.5
      
      const re386 = landmarks[386], re374 = landmarks[374]
      const reH = Math.abs(re374.y - re386.y)
      const rGazeY = (riris.y - re386.y) / (reH + 0.001) - 0.5

      gazeY = (lGazeY + rGazeY) / 2
    }

    // Map to screen region
    let region = '中央'
    if (gazeX < -0.15) region = '左'
    else if (gazeX > 0.15) region = '右'
    if (gazeY < -0.15) region = '上' + (region !== '中央' ? region : '')
    else if (gazeY > 0.15) region = '下' + (region !== '中央' ? region : '')

    const looking = Math.abs(gazeX) < 0.25 && Math.abs(gazeY) < 0.25

    return { region, looking, x: gazeX, y: gazeY }
  }
}

// ---- Focus scorer ----
class FocusScorer {
  constructor() {
    this.facingHistory = []  // boolean buffer
    this.bufferSize = 90     // ~3 seconds at 30fps
  }

  update(facing, looking, blinkRate) {
    this.facingHistory.push(facing ? 1 : 0)
    if (this.facingHistory.length > this.bufferSize) {
      this.facingHistory.shift()
    }

    const facingRatio = this.facingHistory.reduce((a, b) => a + b, 0) / this.facingHistory.length
    
    // Blink rate: 15-20/min is normal focused. <10 or >30 suggests distraction or fatigue
    let blinkScore = 1
    if (blinkRate < 8) blinkScore = 0.7   // staring / zoning out
    if (blinkRate > 30) blinkScore = 0.6  // fatigue

    const score = Math.round(facingRatio * 100 * blinkScore)
    
    let level = '高'
    if (score < 40) level = '低'
    else if (score < 70) level = '中'

    return { score, level }
  }
}

// ---- Synthesis ----
class SynthesisEngine {
  constructor() {
    this.lastState = ''
    this.stateStart = Date.now()
    this.events = []
  }

  synthesize(presence, attention, emotion) {
    let text = ''

    if (presence.count === 0) {
      text = '无人在屏幕前'
    } else if (presence.count > 1) {
      text = `${presence.count} 人在屏幕前`
      if (presence.facing) text += '，主用户面对屏幕'
    } else {
      // Single person
      const parts = []

      if (!presence.facing) {
        parts.push('没有面对屏幕')
      } else if (attention.focus.level === '高') {
        parts.push('专注地看着屏幕')
        if (attention.gaze.region !== '中央') {
          parts.push(`注视${attention.gaze.region}侧`)
        }
      } else if (attention.focus.level === '中') {
        parts.push('在看屏幕，但注意力一般')
      } else {
        parts.push('注意力涣散')
      }

      if (emotion.expression !== '😐 平静' && emotion.expression !== '未知') {
        parts.push(emotion.expression)
      }

      if (emotion.posture === '前倾') {
        parts.push('身体前倾（感兴趣）')
      } else if (emotion.posture === '后仰') {
        parts.push('身体后仰（放松/无聊）')
      }

      text = parts.join('，')
    }

    // State change tracking
    const stateKey = `${presence.count}-${presence.facing}-${attention.focus.level}`
    if (stateKey !== this.lastState) {
      const duration = ((Date.now() - this.stateStart) / 1000).toFixed(0)
      if (this.lastState && parseInt(duration) > 2) {
        this.events.push({
          time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
          text: `${this.lastState === '0-false-低' ? '离开' : '状态切换'} (${duration}s)`
        })
        if (this.events.length > 50) this.events.shift()
      }
      this.lastState = stateKey
      this.stateStart = Date.now()
    }

    return { text, events: this.events }
  }
}

// ---- Main engine ----
export class SenseEngine {
  constructor(video, overlayCanvas) {
    this.video = video
    this.overlay = overlayCanvas
    this.ctx = overlayCanvas.getContext('2d')

    this.detector = null
    this.faceMesh = null
    this.useFaceMesh = false

    this.blinkDetector = new BlinkDetector()
    this.expressionClassifier = new ExpressionClassifier()
    this.headPose = new HeadPoseEstimator()
    this.gazeEstimator = new GazeEstimator()
    this.focusScorer = new FocusScorer()
    this.synthesis = new SynthesisEngine()

    this.distanceEMA = new EMA(0.15)
    this.yawEMA = new EMA(0.2)
    this.pitchEMA = new EMA(0.2)

    this.lastDetection = null
    this.detecting = false
    this.frameCount = 0
  }

  async init() {
    // Try Chrome FaceDetector first
    if ('FaceDetector' in window) {
      try {
        this.detector = new window.FaceDetector({ maxDetectedFaces: 5, fastMode: true })
        console.log('Using Chrome FaceDetector API')
      } catch (e) {
        console.warn('FaceDetector failed:', e)
      }
    }

    // Try MediaPipe Face Mesh
    try {
      await this.initMediaPipe()
    } catch (e) {
      console.warn('MediaPipe not available, using FaceDetector only:', e)
    }

    if (!this.detector && !this.faceMesh) {
      throw new Error('No face detection available')
    }
  }

  async initMediaPipe() {
    // Dynamic import of MediaPipe Face Mesh
    // Self-hosted WASM to avoid CDN issues
    const { FaceMesh } = await import('https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.js')
    
    this.faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`
    })

    this.faceMesh.setOptions({
      maxNumFaces: 3,
      refineLandmarks: true,  // enables iris landmarks (468-477)
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })

    this.faceMeshResults = null
    this.faceMesh.onResults((results) => {
      this.faceMeshResults = results
    })

    // Warm up
    await this.faceMesh.send({ image: this.video })
    this.useFaceMesh = true
    console.log('MediaPipe Face Mesh loaded (with iris)')
  }

  detect() {
    this.frameCount++

    if (this.useFaceMesh) {
      return this.detectMediaPipe()
    } else {
      return this.detectBasic()
    }
  }

  detectMediaPipe() {
    // Send frame every 2 frames (15fps detection)
    if (this.frameCount % 2 === 0 && !this.detecting) {
      this.detecting = true
      this.faceMesh.send({ image: this.video }).then(() => {
        this.detecting = false
      }).catch(() => { this.detecting = false })
    }

    if (!this.faceMeshResults) return null

    const results = this.faceMeshResults
    const faceCount = results.multiFaceLandmarks ? results.multiFaceLandmarks.length : 0

    // Draw overlay
    this.drawOverlay(results)

    if (faceCount === 0) {
      return this.buildResult(0, false, 0, null, null, null)
    }

    // Primary face (first detected)
    const landmarks = results.multiFaceLandmarks[0]

    // Head pose
    const pose = this.headPose.estimate(landmarks)
    const smoothYaw = this.yawEMA.update(pose.yaw)
    const smoothPitch = this.pitchEMA.update(pose.pitch)
    pose.yaw = smoothYaw
    pose.pitch = smoothPitch
    pose.facing = Math.abs(smoothYaw) < 0.15 && Math.abs(smoothPitch) < 0.2

    // Distance from eye spacing
    const leftEye = landmarks[33]
    const rightEye = landmarks[263]
    const eyeDist = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y)
    const rawDist = 0.12 / (eyeDist + 0.001)
    const distance = this.distanceEMA.update(rawDist)

    // Gaze
    const gaze = this.gazeEstimator.estimate(landmarks)

    // Blink (EAR)
    // Left eye landmarks for EAR: 33, 160, 158, 133, 153, 144
    const leftEyePoints = [33, 160, 158, 133, 153, 144].map(i => landmarks[i])
    const rightEyePoints = [362, 385, 387, 263, 380, 373].map(i => landmarks[i])
    const leftEAR = this.blinkDetector.computeEAR(leftEyePoints)
    const rightEAR = this.blinkDetector.computeEAR(rightEyePoints)
    const avgEAR = (leftEAR + rightEAR) / 2
    const blinkRate = this.blinkDetector.update(avgEAR)

    // Focus
    const focus = this.focusScorer.update(pose.facing, gaze.looking, blinkRate)

    // Expression
    const expr = this.expressionClassifier.classify(landmarks)

    return this.buildResult(faceCount, pose.facing, distance, 
      { gaze, blinkRate, focus },
      { expression: expr.expression, posture: pose.posture, tilt: pose.tilt },
      pose
    )
  }

  detectBasic() {
    // Throttled FaceDetector API detection
    if (this.frameCount % 3 === 0 && !this.detecting && this.detector) {
      this.detecting = true
      
      this.detector.detect(this.video).then(faces => {
        this.detecting = false

        const faceCount = faces.length
        if (faceCount === 0) {
          this.lastDetection = this.buildResult(0, false, 0, null, null, null)
          return
        }

        const face = faces[0]
        const box = face.boundingBox

        // Rough facing: face width/height ratio (frontal ≈ 0.7-1.0)
        const aspectRatio = box.width / (box.height + 1)
        const facing = aspectRatio > 0.55

        // Distance from face width
        const rawDist = 200 / (box.width + 1)
        const distance = this.distanceEMA.update(rawDist)

        // Basic overlay
        this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height)
        this.ctx.strokeStyle = 'rgba(74, 222, 128, 0.6)'
        this.ctx.lineWidth = 2
        this.ctx.strokeRect(box.x, box.y, box.width, box.height)

        // No detailed attention/emotion without landmarks
        this.lastDetection = this.buildResult(faceCount, facing, distance, null, null, null)
      }).catch(() => { this.detecting = false })
    }

    return this.lastDetection
  }

  drawOverlay(results) {
    const ctx = this.ctx
    const w = this.overlay.width
    const h = this.overlay.height
    ctx.clearRect(0, 0, w, h)

    if (!results.multiFaceLandmarks) return

    for (const landmarks of results.multiFaceLandmarks) {
      // Face outline (silhouette)
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.3)'
      ctx.lineWidth = 1

      // Draw face oval
      const ovalIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 
        234, 127, 162, 21, 54, 103, 67, 109, 10]
      
      ctx.beginPath()
      const first = landmarks[ovalIndices[0]]
      ctx.moveTo(first.x * w, first.y * h)
      for (let i = 1; i < ovalIndices.length; i++) {
        const p = landmarks[ovalIndices[i]]
        ctx.lineTo(p.x * w, p.y * h)
      }
      ctx.stroke()

      // Eye landmarks
      ctx.fillStyle = 'rgba(96, 165, 250, 0.6)'
      const eyeIndices = [33, 133, 362, 263]  // eye corners
      for (const i of eyeIndices) {
        const p = landmarks[i]
        ctx.beginPath()
        ctx.arc(p.x * w, p.y * h, 2, 0, Math.PI * 2)
        ctx.fill()
      }

      // Iris (if available, landmarks 468-477)
      if (landmarks.length >= 478) {
        ctx.fillStyle = 'rgba(251, 191, 36, 0.8)'  // amber
        for (const i of [468, 473]) {  // left + right iris center
          const p = landmarks[i]
          ctx.beginPath()
          ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
  }

  buildResult(count, facing, distance, attention, emotion, pose) {
    const presence = {
      count,
      distance: distance ? distance.toFixed(1) : '-',
      facing
    }

    const attn = attention || {
      gaze: { region: '未知', looking: false },
      blinkRate: 0,
      focus: { score: 0, level: '-' }
    }

    const emo = emotion || {
      expression: '未知',
      posture: '未知',
      tilt: '-'
    }

    const synth = this.synthesis.synthesize(presence, attn, emo)

    return { presence, attention: attn, emotion: emo, synthesis: synth, pose }
  }
}
