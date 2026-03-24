/**
 * SenseEngine — face landmark detection + feature extraction
 * 
 * Uses @mediapipe/tasks-vision FaceLandmarker (self-hosted WASM + model)
 * Fallback: Chrome FaceDetector API (limited features)
 * 
 * Pipeline: Camera → FaceLandmarker (478 landmarks incl. iris) → Feature extractors → Synthesis
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
    this.history = []
    this.wasOpen = true
    this.threshold = 0.2
  }

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
      const cutoff = Date.now() - 60000
      this.history = this.history.filter(t => t > cutoff)
    }
    this.wasOpen = isOpen
    return this.getRate()
  }

  getRate() {
    const cutoff = Date.now() - 60000
    return this.history.filter(t => t > cutoff).length
  }
}

// ---- Expression classifier ----
class ExpressionClassifier {
  classify(landmarks) {
    if (!landmarks || landmarks.length < 468) return { expression: '未知', confidence: 0 }

    const upperLip = landmarks[13]
    const lowerLip = landmarks[14]
    const mouthLeft = landmarks[61]
    const mouthRight = landmarks[291]

    const mouthOpen = Math.abs(upperLip.y - lowerLip.y)
    const mouthWidth = Math.abs(mouthLeft.x - mouthRight.x)
    const mouthRatio = mouthOpen / (mouthWidth + 0.001)

    const leftBrow = landmarks[66]
    const leftEye = landmarks[159]
    const browEyeDist = Math.abs(leftBrow.y - leftEye.y)

    const mouthCenter = (upperLip.y + lowerLip.y) / 2
    const cornerAvgY = (mouthLeft.y + mouthRight.y) / 2
    const smileScore = mouthCenter - cornerAvgY

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

    const nose = landmarks[1]
    const leftFace = landmarks[234]
    const rightFace = landmarks[454]
    const forehead = landmarks[10]
    const chin = landmarks[152]

    const faceCenter = (leftFace.x + rightFace.x) / 2
    const faceWidth = Math.abs(rightFace.x - leftFace.x)
    const yaw = (nose.x - faceCenter) / (faceWidth + 0.001)

    const faceMidY = (forehead.y + chin.y) / 2
    const faceHeight = Math.abs(chin.y - forehead.y)
    const pitch = (nose.y - faceMidY) / (faceHeight + 0.001)

    const leftEye = landmarks[33]
    const rightEye = landmarks[263]
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180 / Math.PI

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

// ---- Gaze estimator ----
class GazeEstimator {
  estimate(landmarks) {
    if (!landmarks || landmarks.length < 468) return { region: '未知', looking: false }

    const hasIris = landmarks.length >= 478
    let gazeX = 0, gazeY = 0

    if (hasIris) {
      const le33 = landmarks[33], le133 = landmarks[133], liris = landmarks[468]
      const leWidth = Math.abs(le133.x - le33.x)
      const lGazeX = (liris.x - le33.x) / (leWidth + 0.001)

      const re362 = landmarks[362], re263 = landmarks[263], riris = landmarks[473]
      const reWidth = Math.abs(re263.x - re362.x)
      const rGazeX = (riris.x - re362.x) / (reWidth + 0.001)

      gazeX = (lGazeX + rGazeX) / 2 - 0.5

      const le159 = landmarks[159], le145 = landmarks[145]
      const leH = Math.abs(le145.y - le159.y)
      const lGazeY = (liris.y - le159.y) / (leH + 0.001) - 0.5

      const re386 = landmarks[386], re374 = landmarks[374]
      const reH = Math.abs(re374.y - re386.y)
      const rGazeY = (riris.y - re386.y) / (reH + 0.001) - 0.5

      gazeY = (lGazeY + rGazeY) / 2
    }

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
    this.facingHistory = []
    this.gazeHistory = []     // track gaze stability
    this.bufferSize = 90
    this.lastGazeX = 0
    this.lastGazeY = 0
    this.gazeJitter = 0       // how much gaze jumps around
  }

  update(facing, gaze, blinkRate) {
    this.facingHistory.push(facing ? 1 : 0)
    if (this.facingHistory.length > this.bufferSize) this.facingHistory.shift()

    // Gaze stability — rapid eye movement = less focused
    if (gaze && gaze.x !== undefined) {
      const dx = Math.abs(gaze.x - this.lastGazeX)
      const dy = Math.abs(gaze.y - this.lastGazeY)
      const jitter = dx + dy
      this.gazeJitter = this.gazeJitter * 0.95 + jitter * 0.05  // EMA
      this.lastGazeX = gaze.x
      this.lastGazeY = gaze.y
    }

    const facingRatio = this.facingHistory.reduce((a, b) => a + b, 0) / this.facingHistory.length

    // Blink rate factor: 12-20/min normal. Outside = less focused
    let blinkFactor = 1
    if (blinkRate < 5) blinkFactor = 0.75   // staring, zoning out
    if (blinkRate > 28) blinkFactor = 0.65  // fatigue

    // Gaze stability factor: stable gaze = focused
    let gazeFactor = 1
    if (this.gazeJitter > 0.02) gazeFactor = 0.8   // some wandering
    if (this.gazeJitter > 0.05) gazeFactor = 0.6   // very jumpy

    const score = Math.round(facingRatio * 100 * blinkFactor * gazeFactor)

    let level = '高'
    if (score < 35) level = '低'
    else if (score < 65) level = '中'

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
    } else {
      const parts = []
      if (!presence.facing) {
        parts.push('没看屏幕')
      } else {
        // Don't just say "专注" — describe what they're doing
        if (attention.focus.level === '高') {
          if (attention.gaze.region !== '中央' && attention.gaze.region !== '未知') {
            parts.push(`看着屏幕${attention.gaze.region}`)
          } else {
            parts.push('注视屏幕中')
          }
        } else if (attention.focus.level === '中') {
          parts.push('在看，有点走神')
        } else {
          parts.push('心不在焉')
        }
      }

      // Only add expression if it's NOT 平静 (that's the default, not interesting)
      if (emotion.expression && !emotion.expression.includes('平静')) {
        parts.push(emotion.expression)
      }

      if (emotion.posture === '前倾') parts.push('前倾')
      else if (emotion.posture === '后仰') parts.push('后仰')
      else if (emotion.posture === '侧头') parts.push('侧头')

      text = parts.join('，') || '面对屏幕'
    }

    const stateKey = `${presence.count}-${presence.facing}-${attention.focus.level}`
    if (stateKey !== this.lastState) {
      const duration = ((Date.now() - this.stateStart) / 1000).toFixed(0)
      if (this.lastState && parseInt(duration) > 2) {
        this.events.push({
          time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
          text: `${this.lastState.startsWith('0') ? '离开' : '状态切换'} (${duration}s)`
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

    this.faceLandmarker = null
    this.detector = null  // Chrome FaceDetector fallback

    this.blinkDetector = new BlinkDetector()
    this.expressionClassifier = new ExpressionClassifier()
    this.headPose = new HeadPoseEstimator()
    this.gazeEstimator = new GazeEstimator()
    this.focusScorer = new FocusScorer()
    this.synthesis = new SynthesisEngine()

    this.distanceEMA = new EMA(0.15)
    this.yawEMA = new EMA(0.2)
    this.pitchEMA = new EMA(0.2)

    this.lastResult = null
    this.frameCount = 0
  }

  async init() {
    // Try MediaPipe tasks-vision (self-hosted)
    try {
      await this.initMediaPipe()
      console.log('MediaPipe FaceLandmarker ready (478 landmarks + iris)')
      return
    } catch (e) {
      console.warn('MediaPipe init failed:', e)
    }

    // Fallback: Chrome FaceDetector
    if ('FaceDetector' in window) {
      try {
        this.detector = new window.FaceDetector({ maxDetectedFaces: 5, fastMode: true })
        console.log('Fallback: Chrome FaceDetector API')
        return
      } catch (e) {
        console.warn('FaceDetector failed:', e)
      }
    }

    throw new Error('No face detection available')
  }

  async initMediaPipe() {
    // Import the tasks-vision module
    const vision = await import('./mediapipe/vision_bundle.mjs')
    const { FaceLandmarker, FilesetResolver } = vision

    // Use self-hosted WASM files
    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')

    this.faceLandmarker = await FaceLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/face_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numFaces: 3,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false
    })
  }

  detect() {
    this.frameCount++

    if (this.faceLandmarker) {
      return this.detectMediaPipe()
    } else if (this.detector) {
      return this.detectBasic()
    }
    return null
  }

  detectMediaPipe() {
    if (this.video.readyState < 2) return this.lastResult

    const now = performance.now()
    let results
    try {
      results = this.faceLandmarker.detectForVideo(this.video, now)
    } catch (e) {
      return this.lastResult
    }

    const faceCount = results.faceLandmarks ? results.faceLandmarks.length : 0

    this.drawOverlay(results)

    if (faceCount === 0) {
      this.lastResult = this.buildResult(0, false, 0, null, null, null)
      return this.lastResult
    }

    // Primary face
    const landmarks = results.faceLandmarks[0]

    // Head pose
    const pose = this.headPose.estimate(landmarks)
    pose.yaw = this.yawEMA.update(pose.yaw)
    pose.pitch = this.pitchEMA.update(pose.pitch)
    pose.facing = Math.abs(pose.yaw) < 0.15 && Math.abs(pose.pitch) < 0.2

    // Distance
    const leftEye = landmarks[33], rightEye = landmarks[263]
    const eyeDist = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y)
    const distance = this.distanceEMA.update(0.12 / (eyeDist + 0.001))

    // Gaze
    const gaze = this.gazeEstimator.estimate(landmarks)

    // Blink
    const leftEyePoints = [33, 160, 158, 133, 153, 144].map(i => landmarks[i])
    const rightEyePoints = [362, 385, 387, 263, 380, 373].map(i => landmarks[i])
    const avgEAR = (this.blinkDetector.computeEAR(leftEyePoints) + this.blinkDetector.computeEAR(rightEyePoints)) / 2
    const blinkRate = this.blinkDetector.update(avgEAR)

    // Focus
    const focus = this.focusScorer.update(pose.facing, gaze, blinkRate)

    // Expression — use blendshapes if available, otherwise landmark-based
    let expression = '😐 平静'
    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
      expression = this.expressionFromBlendshapes(results.faceBlendshapes[0])
    } else {
      expression = this.expressionClassifier.classify(landmarks).expression
    }

    this.lastResult = this.buildResult(faceCount, pose.facing, distance,
      { gaze, blinkRate, focus },
      { expression, posture: pose.posture, tilt: pose.tilt },
      pose
    )
    return this.lastResult
  }

  expressionFromBlendshapes(blendshapes) {
    const bs = {}
    for (const cat of blendshapes.categories) {
      bs[cat.categoryName] = cat.score
    }

    const jawOpen = bs['jawOpen'] || 0
    const mouthSmileL = bs['mouthSmileLeft'] || 0
    const mouthSmileR = bs['mouthSmileRight'] || 0
    const browInnerUp = bs['browInnerUp'] || 0
    const browDownL = bs['browDownLeft'] || 0
    const browDownR = bs['browDownRight'] || 0
    const eyeSquintL = bs['eyeSquintLeft'] || 0
    const eyeSquintR = bs['eyeSquintRight'] || 0
    const mouthFrownL = bs['mouthFrownLeft'] || 0
    const mouthFrownR = bs['mouthFrownRight'] || 0
    const mouthPucker = bs['mouthPucker'] || 0
    const cheekPuff = bs['cheekPuff'] || 0
    const eyeWideL = bs['eyeWideLeft'] || 0
    const eyeWideR = bs['eyeWideRight'] || 0

    const smile = (mouthSmileL + mouthSmileR) / 2
    const frown = (mouthFrownL + mouthFrownR) / 2
    const browDown = (browDownL + browDownR) / 2
    const eyeWide = (eyeWideL + eyeWideR) / 2

    // Log top blendshapes for tuning (uncomment to debug)
    // const sorted = Object.entries(bs).filter(([,v]) => v > 0.1).sort((a,b) => b[1]-a[1]).slice(0,5)
    // if (this.frameCount % 30 === 0) console.log('BS:', sorted.map(([k,v]) => `${k}:${v.toFixed(2)}`).join(' '))

    // Thresholds tuned: jawOpen 0.4→0.6 (was triggering on slightly open mouth)
    // Need BOTH jawOpen AND eyeWide for true surprise vs just talking
    if (jawOpen > 0.6 && eyeWide > 0.3) return '😮 惊讶'
    if (jawOpen > 0.5) return '🗣️ 说话'
    if (smile > 0.5) return '😊 微笑'
    if (smile > 0.25 && jawOpen < 0.3) return '🙂 轻松'
    if (browInnerUp > 0.5 && browDown < 0.2) return '🤔 困惑'
    if (browDown > 0.4) return '🤨 皱眉'
    if (frown > 0.35) return '😕 不悦'
    if ((eyeSquintL + eyeSquintR) / 2 > 0.6) return '😑 眯眼'
    if (mouthPucker > 0.5) return '😗 嘟嘴'
    return '😐 平静'
  }

  detectBasic() {
    // Throttled FaceDetector API
    if (this.frameCount % 3 !== 0 || !this.detector) return this.lastResult
    if (this.video.readyState < 2) return this.lastResult

    this.detector.detect(this.video).then(faces => {
      const faceCount = faces.length
      if (faceCount === 0) {
        this.lastResult = this.buildResult(0, false, 0, null, null, null)
        return
      }

      const face = faces[0]
      const box = face.boundingBox
      const aspectRatio = box.width / (box.height + 1)
      const facing = aspectRatio > 0.55
      const distance = this.distanceEMA.update(200 / (box.width + 1))

      this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height)
      this.ctx.strokeStyle = 'rgba(74, 222, 128, 0.6)'
      this.ctx.lineWidth = 2
      this.ctx.strokeRect(box.x, box.y, box.width, box.height)

      this.lastResult = this.buildResult(faceCount, facing, distance, null, null, null)
    }).catch(() => {})

    return this.lastResult
  }

  drawOverlay(results) {
    const ctx = this.ctx
    const w = this.overlay.width
    const h = this.overlay.height
    ctx.clearRect(0, 0, w, h)

    if (!results.faceLandmarks) return

    for (const landmarks of results.faceLandmarks) {
      // Face oval
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.3)'
      ctx.lineWidth = 1

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

      // Eye corners
      ctx.fillStyle = 'rgba(96, 165, 250, 0.6)'
      for (const i of [33, 133, 362, 263]) {
        const p = landmarks[i]
        ctx.beginPath()
        ctx.arc(p.x * w, p.y * h, 2, 0, Math.PI * 2)
        ctx.fill()
      }

      // Iris
      if (landmarks.length >= 478) {
        ctx.fillStyle = 'rgba(251, 191, 36, 0.8)'
        for (const i of [468, 473]) {
          const p = landmarks[i]
          ctx.beginPath()
          ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
  }

  buildResult(count, facing, distance, attention, emotion, pose) {
    const presence = { count, distance: distance ? distance.toFixed(1) : '-', facing }

    const attn = attention || {
      gaze: { region: '未知', looking: false },
      blinkRate: 0,
      focus: { score: 0, level: '-' }
    }

    const emo = emotion || { expression: '未知', posture: '未知', tilt: '-' }
    const synth = this.synthesis.synthesize(presence, attn, emo)

    return { presence, attention: attn, emotion: emo, synthesis: synth, pose }
  }
}
