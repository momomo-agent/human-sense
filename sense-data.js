/**
 * agentic-sense/sense-data.js
 * 
 * 二次封装层：把 MediaPipe 原始 landmarks + blendshapes 
 * 转成结构化的、可直接消费的感知数据
 * 
 * 输入：MediaPipe FaceLandmarker 原始结果
 * 输出：HumanSenseFrame — 一帧完整的人类感知数据
 */

// MediaPipe landmark indices (468 + 10 iris)
const IDX = {
  // Face structure
  noseTip: 1,
  forehead: 10,
  chin: 152,
  leftFaceEdge: 234,
  rightFaceEdge: 454,

  // Eyes — corners
  leftEyeInner: 133,
  leftEyeOuter: 33,
  rightEyeInner: 362,
  rightEyeOuter: 263,

  // Eyes — EAR points (upper/lower lid)
  leftEyeUpper: [159, 160, 161],
  leftEyeLower: [144, 145, 153],
  rightEyeUpper: [386, 385, 384],
  rightEyeLower: [373, 374, 380],

  // EAR 6-point model
  leftEAR: [33, 160, 158, 133, 153, 144],
  rightEAR: [362, 385, 387, 263, 380, 373],

  // Iris centers (landmarks 468-477)
  leftIris: 468,
  rightIris: 473,

  // Mouth
  upperLipCenter: 13,
  lowerLipCenter: 14,
  mouthLeft: 61,
  mouthRight: 291,

  // Face oval (for overlay)
  faceOval: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
}

// Blendshape name mapping → clean keys
const BS_MAP = {
  jawOpen: 'jawOpen',
  mouthSmileLeft: 'smileL',
  mouthSmileRight: 'smileR',
  mouthFrownLeft: 'frownL',
  mouthFrownRight: 'frownR',
  mouthPucker: 'pucker',
  browInnerUp: 'browUp',
  browDownLeft: 'browDownL',
  browDownRight: 'browDownR',
  eyeSquintLeft: 'squintL',
  eyeSquintRight: 'squintR',
  eyeWideLeft: 'eyeWideL',
  eyeWideRight: 'eyeWideR',
  eyeBlinkLeft: 'blinkL',
  eyeBlinkRight: 'blinkR',
  cheekPuff: 'cheekPuff',
  mouthClose: 'mouthClose',
  mouthOpen: 'mouthOpen',
  noseSneerLeft: 'sneerL',
  noseSneerRight: 'sneerR',
  jawLeft: 'jawLeft',
  jawRight: 'jawRight',
  mouthLeft: 'mouthL',
  mouthRight: 'mouthR',
  mouthShrugUpper: 'shrugUpper',
  mouthShrugLower: 'shrugLower',
  mouthRollUpper: 'rollUpper',
  mouthRollLower: 'rollLower',
  mouthFunnel: 'funnel',
  mouthDimpleLeft: 'dimpleL',
  mouthDimpleRight: 'dimpleR',
  mouthStretchLeft: 'stretchL',
  mouthStretchRight: 'stretchR',
  mouthPressLeft: 'pressL',
  mouthPressRight: 'pressR',
  mouthLowerDownLeft: 'lowerDownL',
  mouthLowerDownRight: 'lowerDownR',
  mouthUpperUpLeft: 'upperUpL',
  mouthUpperUpRight: 'upperUpR',
}

/**
 * Extract structured sense data from one face's landmarks + blendshapes
 */
function extractFace(landmarks, blendshapeCategories) {
  const lm = landmarks
  const hasIris = lm.length >= 478

  // ---- Geometry ----
  const head = extractHead(lm)
  const eyes = extractEyes(lm, hasIris)
  const mouth = extractMouth(lm)

  // ---- Blendshapes ----
  const blendshapes = {}
  const rawBlendshapes = {}
  if (blendshapeCategories) {
    for (const cat of blendshapeCategories) {
      rawBlendshapes[cat.categoryName] = cat.score
      const key = BS_MAP[cat.categoryName]
      if (key) blendshapes[key] = Math.round(cat.score * 1000) / 1000
    }
  }

  return {
    head,
    eyes,
    mouth,
    blendshapes,
    rawBlendshapes,
    landmarkCount: lm.length,
  }
}

function extractHead(lm) {
  const nose = lm[IDX.noseTip]
  const forehead = lm[IDX.forehead]
  const chin = lm[IDX.chin]
  const leftEdge = lm[IDX.leftFaceEdge]
  const rightEdge = lm[IDX.rightFaceEdge]

  const faceWidth = Math.abs(rightEdge.x - leftEdge.x)
  const faceHeight = Math.abs(chin.y - forehead.y)
  const faceCenter = { x: (leftEdge.x + rightEdge.x) / 2, y: (forehead.y + chin.y) / 2 }

  // Yaw: nose offset from face center, normalized
  const yaw = (nose.x - faceCenter.x) / (faceWidth + 0.001)
  // Pitch: nose vertical offset
  const pitch = (nose.y - faceCenter.y) / (faceHeight + 0.001)
  // Roll: angle of eye line
  const leye = lm[IDX.leftEyeOuter]
  const reye = lm[IDX.rightEyeOuter]
  const roll = Math.atan2(reye.y - leye.y, reye.x - leye.x) * 180 / Math.PI

  return {
    position: { x: nose.x, y: nose.y, z: nose.z },
    yaw: round(yaw),
    pitch: round(pitch),
    roll: round(roll),
    faceWidth: round(faceWidth),
    faceHeight: round(faceHeight),
    faceCenter,
    facing: Math.abs(yaw) < 0.15 && Math.abs(pitch) < 0.2,
  }
}

function extractEyes(lm, hasIris) {
  // Eye Aspect Ratio
  const leftEAR = computeEAR(IDX.leftEAR.map(i => lm[i]))
  const rightEAR = computeEAR(IDX.rightEAR.map(i => lm[i]))

  // Eye centers
  const leftCenter = midpoint(lm[IDX.leftEyeInner], lm[IDX.leftEyeOuter])
  const rightCenter = midpoint(lm[IDX.rightEyeInner], lm[IDX.rightEyeOuter])

  // Inter-pupillary distance (proxy for screen distance)
  const ipd = Math.hypot(rightCenter.x - leftCenter.x, rightCenter.y - leftCenter.y)

  // Iris positions (if available)
  let iris = null
  if (hasIris) {
    const li = lm[IDX.leftIris]
    const ri = lm[IDX.rightIris]

    // Iris position within eye socket (0=inner corner, 1=outer corner)
    const leftSocket = Math.abs(lm[IDX.leftEyeOuter].x - lm[IDX.leftEyeInner].x)
    const rightSocket = Math.abs(lm[IDX.rightEyeOuter].x - lm[IDX.rightEyeInner].x)

    iris = {
      left: { x: li.x, y: li.y, ratioX: (li.x - lm[IDX.leftEyeInner].x) / (leftSocket + 0.001) },
      right: { x: ri.x, y: ri.y, ratioX: (ri.x - lm[IDX.rightEyeInner].x) / (rightSocket + 0.001) },
    }
  }

  return {
    left: { center: leftCenter, ear: round(leftEAR) },
    right: { center: rightCenter, ear: round(rightEAR) },
    avgEAR: round((leftEAR + rightEAR) / 2),
    ipd: round(ipd),
    iris,
  }
}

function extractMouth(lm) {
  const upper = lm[IDX.upperLipCenter]
  const lower = lm[IDX.lowerLipCenter]
  const left = lm[IDX.mouthLeft]
  const right = lm[IDX.mouthRight]

  const openness = Math.abs(upper.y - lower.y)
  const width = Math.abs(left.x - right.x)
  const ratio = openness / (width + 0.001)

  return {
    openness: round(openness),
    width: round(width),
    ratio: round(ratio),
    center: midpoint(upper, lower),
  }
}

// ---- Utilities ----
function computeEAR(pts) {
  if (pts.length < 6) return 1
  const [p1, p2, p3, p4, p5, p6] = pts
  const v1 = Math.hypot(p2.x - p6.x, p2.y - p6.y)
  const v2 = Math.hypot(p3.x - p5.x, p3.y - p5.y)
  const h = Math.hypot(p1.x - p4.x, p1.y - p4.y)
  return (v1 + v2) / (2 * h + 0.001)
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
}

function round(v, d = 4) {
  const f = Math.pow(10, d)
  return Math.round(v * f) / f
}

/**
 * Process full MediaPipe results into a HumanSenseFrame
 * 
 * @param {object} faceResult - FaceLandmarker result
 * @param {object} handResult - GestureRecognizer result (optional)
 * @param {object} poseResult - PoseLandmarker result (optional)
 * @param {object} segResult  - ImageSegmenter result (optional)
 * @param {object} objectResult - ObjectDetector result (optional)
 * @param {object} faceDetResult - FaceDetector result (optional)
 * @param {object} handLmResult - HandLandmarker result (optional, fallback if no gesture)
 */
export function extractFrame(faceResult, handResult, poseResult, segResult, objectResult, faceDetResult, handLmResult, timestamp) {
  // ---- Faces ----
  const faces = []
  const faceCount = faceResult?.faceLandmarks ? faceResult.faceLandmarks.length : 0

  for (let i = 0; i < faceCount; i++) {
    const landmarks = faceResult.faceLandmarks[i]
    const bs = faceResult.faceBlendshapes ? faceResult.faceBlendshapes[i]?.categories : null
    faces.push(extractFace(landmarks, bs))
  }

  // ---- Hands (from GestureRecognizer) ----
  const hands = []
  const handSource = handResult?.landmarks ? handResult : handLmResult
  if (handSource?.landmarks) {
    for (let h = 0; h < handSource.landmarks.length; h++) {
      const lm = handSource.landmarks[h]
      const handedness = handSource.handednesses?.[h]?.[0]
      const gesture = handResult?.gestures?.[h]?.[0]  // only from GestureRecognizer

      // Finger states: check if fingertip is above (lower y) its PIP joint
      const fingers = {
        thumb:  lm[4].y < lm[3].y,
        index:  lm[8].y < lm[6].y,
        middle: lm[12].y < lm[10].y,
        ring:   lm[16].y < lm[14].y,
        pinky:  lm[20].y < lm[18].y,
      }
      const extendedCount = Object.values(fingers).filter(Boolean).length

      hands.push({
        side: handedness?.categoryName || 'unknown',  // 'Left' or 'Right'
        confidence: round(handedness?.score || 0),
        gesture: gesture?.categoryName !== 'None' ? gesture?.categoryName : null,
        gestureConfidence: gesture?.categoryName !== 'None' ? round(gesture?.score || 0) : null,
        wrist: { x: round(lm[0].x), y: round(lm[0].y), z: round(lm[0].z) },
        fingers,
        extendedCount,
        landmarks: lm.map(p => ({ x: round(p.x), y: round(p.y), z: round(p.z) })),
      })
    }
  }

  // ---- Pose (body) ----
  let body = null
  if (poseResult?.landmarks && poseResult.landmarks.length > 0) {
    const pose = poseResult.landmarks[0]

    // Named body parts
    const POSE_NAMES = {
      0: 'nose', 
      11: 'leftShoulder', 12: 'rightShoulder',
      13: 'leftElbow', 14: 'rightElbow',
      15: 'leftWrist', 16: 'rightWrist',
      23: 'leftHip', 24: 'rightHip',
      25: 'leftKnee', 26: 'rightKnee',
      27: 'leftAnkle', 28: 'rightAnkle',
    }

    const joints = {}
    for (const [idx, name] of Object.entries(POSE_NAMES)) {
      const pt = pose[parseInt(idx)]
      if (pt) {
        joints[name] = { x: round(pt.x), y: round(pt.y), z: round(pt.z), visibility: round(pt.visibility || 0) }
      }
    }

    // Derived metrics
    const shoulderWidth = joints.leftShoulder && joints.rightShoulder
      ? round(Math.abs(joints.rightShoulder.x - joints.leftShoulder.x))
      : null

    const shoulderMidY = joints.leftShoulder && joints.rightShoulder
      ? (joints.leftShoulder.y + joints.rightShoulder.y) / 2
      : null
    const hipMidY = joints.leftHip && joints.rightHip
      ? (joints.leftHip.y + joints.rightHip.y) / 2
      : null
    const torsoLength = shoulderMidY !== null && hipMidY !== null
      ? round(Math.abs(hipMidY - shoulderMidY))
      : null

    body = {
      joints,
      shoulderWidth,
      torsoLength,
      landmarkCount: pose.length,
    }
  }

  // ---- Segmentation ----
  let segmentation = null
  if (segResult?.categoryMask) {
    const mask = segResult.categoryMask
    // Compute person pixel ratio (how much of frame is person)
    const data = mask.getAsUint8Array()
    let personPixels = 0
    for (let i = 0; i < data.length; i++) {
      if (data[i] > 0) personPixels++
    }
    const totalPixels = data.length
    segmentation = {
      personRatio: round(personPixels / totalPixels),
      width: mask.width,
      height: mask.height,
      // mask itself not included (too large for JSON) — access via engine.lastSegResult
    }
  }

  // ---- Objects ----
  const objects = []
  if (objectResult?.detections) {
    for (const det of objectResult.detections) {
      const cat = det.categories?.[0]
      const bb = det.boundingBox
      if (cat && bb) {
        objects.push({
          label: cat.categoryName,
          confidence: round(cat.score),
          box: {
            x: round(bb.originX),
            y: round(bb.originY),
            width: round(bb.width),
            height: round(bb.height),
          }
        })
      }
    }
  }

  // ---- Fast face detection ----
  let faceDetection = null
  if (faceDetResult?.detections) {
    faceDetection = {
      count: faceDetResult.detections.length,
      faces: faceDetResult.detections.map(det => {
        const bb = det.boundingBox
        const keypoints = det.keypoints || []
        return {
          box: bb ? {
            x: round(bb.originX),
            y: round(bb.originY),
            width: round(bb.width),
            height: round(bb.height),
          } : null,
          keypoints: keypoints.map(kp => ({
            x: round(kp.x),
            y: round(kp.y),
            label: kp.label || null,
          })),
          confidence: round(det.categories?.[0]?.score || 0),
        }
      })
    }
  }

  return {
    timestamp: timestamp || performance.now(),
    faceCount,
    faces,
    handCount: hands.length,
    hands,
    body,
    segmentation,
    objectCount: objects.length,
    objects,
    faceDetection,
  }
}

export { IDX }
