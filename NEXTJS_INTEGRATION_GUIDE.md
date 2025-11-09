# Next.js Integration Guide for ASL Real-time Recognition Backend

## Table of Contents
1. [Application Overview](#application-overview)
2. [Architecture Analysis](#architecture-analysis)
3. [Backend Endpoints & Events](#backend-endpoints--events)
4. [Next.js Integration Setup](#nextjs-integration-setup)
5. [Implementation Examples](#implementation-examples)
6. [Deployment Considerations](#deployment-considerations)
7. [Error Handling & Best Practices](#error-handling--best-practices)

---

## Application Overview

This is an **ASL (American Sign Language) Real-time Recognition System** that processes webcam video streams and recognizes sign language gestures using a TensorFlow/Keras model.

### Technology Stack (Backend)
- **Framework**: FastAPI + Socket.IO (python-socketio 5.8.0)
- **ML Framework**: TensorFlow 2.3.1 with Keras
- **Real-time Communication**: Socket.IO over ASGI
- **Image Processing**: OpenCV (cv2)
- **Model**: WLASL20c (20 classes) trained on 224x224x3 frames

### Key Features
- Real-time video frame processing (10-frame buffer)
- Socket.IO bidirectional communication
- Sliding window prediction with threshold-based filtering
- Support for 20 ASL words/signs
- Non-blocking async prediction with threading

---

## Architecture Analysis

### Backend Components

#### 1. **server.py** (Main Entry Point)
- **FastAPI App**: Handles HTTP endpoints (`/healthz`, `/labels`)
- **Socket.IO Server**: Manages WebSocket connections for real-time streaming
- **Client Session Management**: One `VideoInferenceService` instance per connected client

#### 2. **model_runtime.py** (Inference Engine)
- **VideoInferenceService**: Maintains a 10-frame sliding window buffer per client
- **Threading Model**: One prediction thread per client (non-blocking)
- **Prediction Logic**: Runs inference when buffer is full, returns latest result

#### 3. **utils.py** (Image Processing)
- `b64_to_bgr_image()`: Decodes base64 JPEG/PNG to OpenCV BGR array
- `preprocess_frame_bgr_exact()`: Resizes to 224x224 and normalizes to [0,1]

#### 4. **config.py** (Configuration)
```python
DIM = (224, 224)          # Frame dimensions
FRAMES = 10               # Buffer size
CHANNELS = 3              # RGB/BGR
THRESHOLD = 0.50          # Confidence threshold
MODEL_PATH = "./model/WLASL20c_model.h5"
```

#### 5. **labels.py** (Sign Classes)
20 supported signs: book, chair, clothes, computer, drink, drum, family, football, go, hat, hello, kiss, like, play, school, street, table, university, violin, wall

---

## Backend Endpoints & Events

### HTTP REST Endpoints

#### 1. `GET /healthz`
**Purpose**: Health check and model status  
**Response**:
```json
{
  "status": "ok",
  "model": "Sequential"
}
```
**When to call**: On app initialization to verify backend is running

---

#### 2. `GET /labels`
**Purpose**: Retrieve all supported ASL sign labels  
**Response**:
```json
{
  "0": "book",
  "1": "chair",
  "2": "clothes",
  ...
  "19": "wall"
}
```
**When to call**: On component mount to populate sign vocabulary

---

### Socket.IO Events

#### Server URL
```
git@github.com:DhruvJ2k4/ASL_liveWordsRecognition_webSockets.git
```

#### Client → Server Events

##### 1. **`connect`** (Automatic)
**Trigger**: When client establishes Socket.IO connection  
**Backend Action**: 
- Creates a `VideoInferenceService` instance for the client
- Sends `status` event with `{"message": "ready"}`

**Frontend Action**: Initialize webcam and start streaming

---

##### 2. **`frame`** (Manual Emission)
**Purpose**: Send video frame for prediction  
**Payload**:
```javascript
{
  image: "<base64-encoded-JPEG-or-PNG>"
}
```

**Example**:
```javascript
socket.emit('frame', { image: base64Frame });
```

**Backend Processing Flow**:
1. Decode base64 → BGR image
2. Resize to 224x224, normalize to [0, 1]
3. Append to client's 10-frame buffer
4. If buffer full & no active prediction thread → launch prediction
5. Return latest prediction result immediately

**Frequency**: Send every frame (e.g., 25 FPS)

---

##### 3. **`disconnect`** (Automatic)
**Trigger**: When client closes connection  
**Backend Action**: Cleanup client's `VideoInferenceService` instance

---

#### Server → Client Events

##### 1. **`status`** (Received on Connect)
**Purpose**: Connection confirmation  
**Payload**:
```javascript
{
  message: "ready"
}
```

**Frontend Action**: Start webcam stream

---

##### 2. **`prediction`** (Continuous Stream)
**Purpose**: Real-time prediction results  
**Payload**:
```javascript
{
  label: "hello",    // Recognized sign or "none"
  score: 0.92        // Confidence score (0.0 - 1.0)
}
```

**Emission Logic**:
- Sent **for every frame** received
- Returns latest cached prediction (non-blocking)
- `label = "none"` if confidence < 0.50

**Frontend Action**: Update UI with recognized sign

---

## Next.js Integration Setup

### 1. Install Dependencies

```bash
npm install socket.io-client
# or
yarn add socket.io-client
```

### 2. Environment Variables

Create `.env.local`:
```env
NEXT_PUBLIC_SOCKETIO_URL=http://localhost:8000
```

For production:
```env
NEXT_PUBLIC_SOCKETIO_URL=https://your-backend-domain.com
```

---

## Implementation Examples

### Example 1: React Component with Socket.IO (App Router)

```typescript
// app/components/ASLRecognizer.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface PredictionResult {
  label: string;
  score: number;
}

export default function ASLRecognizer() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult>({
    label: 'none',
    score: 0,
  });
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [isStreaming, setIsStreaming] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize Socket.IO connection
  useEffect(() => {
    const socketUrl = process.env.NEXT_PUBLIC_SOCKETIO_URL || 'http://localhost:8000';
    const newSocket = io(socketUrl, {
      transports: ['websocket', 'polling'],
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  // Socket event listeners
  useEffect(() => {
    if (!socket) return;

    // Connection established
    socket.on('connect', () => {
      console.log('[Socket.IO] Connected');
      setIsConnected(true);
    });

    // Disconnection
    socket.on('disconnect', () => {
      console.log('[Socket.IO] Disconnected');
      setIsConnected(false);
      stopStreaming();
    });

    // Server ready status
    socket.on('status', (data: { message: string }) => {
      console.log('[Server Status]', data.message);
    });

    // Prediction results
    socket.on('prediction', (data: PredictionResult) => {
      setPrediction(data);
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('status');
      socket.off('prediction');
    };
  }, [socket]);

  // Fetch labels on mount
  useEffect(() => {
    const fetchLabels = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_SOCKETIO_URL}/labels`);
        const data = await res.json();
        setLabels(data);
      } catch (error) {
        console.error('[Error] Failed to fetch labels:', error);
      }
    };

    fetchLabels();
  }, []);

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: 'user',
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('[Error] Failed to access webcam:', error);
      alert('Please allow camera access to use ASL recognition');
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  // Capture frame and send to backend
  const captureAndSendFrame = () => {
    if (!socket || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.readyState !== video.HAVE_ENOUGH_DATA) return;

    // Draw video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to base64 JPEG
    canvas.toBlob(
      (blob) => {
        if (!blob) return;

        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = (reader.result as string).split(',')[1];
          
          // Send frame to backend
          socket.emit('frame', { image: base64String });
        };
        reader.readAsDataURL(blob);
      },
      'image/jpeg',
      0.8 // JPEG quality
    );
  };

  // Start streaming frames
  const startStreaming = async () => {
    await startWebcam();
    setIsStreaming(true);

    // Send frames at ~25 FPS
    streamIntervalRef.current = setInterval(() => {
      captureAndSendFrame();
    }, 40); // 1000ms / 25fps = 40ms
  };

  // Stop streaming
  const stopStreaming = () => {
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    stopWebcam();
    setIsStreaming(false);
    setPrediction({ label: 'none', score: 0 });
  };

  return (
    <div className="flex flex-col items-center gap-6 p-8">
      <h1 className="text-3xl font-bold">ASL Real-time Recognition</h1>

      {/* Connection Status */}
      <div className="flex items-center gap-2">
        <div
          className={`w-3 h-3 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`}
        />
        <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {/* Video Feed */}
      <div className="relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-[640px] h-[480px] bg-black rounded-lg"
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Prediction Overlay */}
        {isStreaming && (
          <div className="absolute bottom-4 left-4 bg-black/70 text-white px-4 py-2 rounded-lg">
            <p className="text-xl font-bold">
              {prediction.label === 'none'
                ? 'No sign detected'
                : `${prediction.label.toUpperCase()}`}
            </p>
            {prediction.label !== 'none' && (
              <p className="text-sm">
                Confidence: {(prediction.score * 100).toFixed(1)}%
              </p>
            )}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-4">
        <button
          onClick={startStreaming}
          disabled={!isConnected || isStreaming}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition"
        >
          Start Recognition
        </button>
        <button
          onClick={stopStreaming}
          disabled={!isStreaming}
          className="px-6 py-2 bg-red-600 text-white rounded-lg disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-red-700 transition"
        >
          Stop
        </button>
      </div>

      {/* Supported Signs */}
      <div className="w-full max-w-2xl">
        <h2 className="text-xl font-semibold mb-3">Supported Signs ({Object.keys(labels).length})</h2>
        <div className="grid grid-cols-4 gap-2">
          {Object.values(labels).map((label) => (
            <div
              key={label}
              className={`px-3 py-2 rounded-lg text-center ${
                prediction.label === label
                  ? 'bg-green-500 text-white font-bold'
                  : 'bg-gray-200 text-gray-700'
              }`}
            >
              {label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

---

### Example 2: Pages Router Implementation

```typescript
// pages/asl-recognition.tsx
import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import type { NextPage } from 'next';

const ASLRecognition: NextPage = () => {
  // Same component code as above
  // ... (copy the component logic)
};

export default ASLRecognition;
```

---

### Example 3: Custom Hook for Socket.IO

```typescript
// hooks/useASLSocket.ts
import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface PredictionResult {
  label: string;
  score: number;
}

export function useASLSocket() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult>({
    label: 'none',
    score: 0,
  });

  useEffect(() => {
    const socketUrl = process.env.NEXT_PUBLIC_SOCKETIO_URL || 'http://localhost:8000';
    const newSocket = io(socketUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    newSocket.on('connect', () => {
      console.log('[Socket.IO] Connected');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('[Socket.IO] Disconnected');
      setIsConnected(false);
    });

    newSocket.on('status', (data: { message: string }) => {
      console.log('[Server]', data.message);
    });

    newSocket.on('prediction', (data: PredictionResult) => {
      setPrediction(data);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const sendFrame = (base64Image: string) => {
    if (socket && isConnected) {
      socket.emit('frame', { image: base64Image });
    }
  };

  return { socket, isConnected, prediction, sendFrame };
}
```

Usage:
```typescript
// components/ASLRecognizerSimple.tsx
'use client';

import { useASLSocket } from '@/hooks/useASLSocket';

export default function ASLRecognizerSimple() {
  const { isConnected, prediction, sendFrame } = useASLSocket();

  // ... rest of component logic
}
```

---

### Example 4: API Route for Health Check (Optional)

```typescript
// app/api/asl/health/route.ts
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_SOCKETIO_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/healthz`);
    const data = await response.json();

    return NextResponse.json({
      healthy: data.status === 'ok',
      ...data,
    });
  } catch (error) {
    return NextResponse.json(
      { healthy: false, error: 'Backend unreachable' },
      { status: 503 }
    );
  }
}
```

```typescript
// app/api/asl/labels/route.ts
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_SOCKETIO_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/labels`);
    const labels = await response.json();

    return NextResponse.json(labels);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch labels' },
      { status: 500 }
    );
  }
}
```

---

## Deployment Considerations

### Backend Deployment

#### 1. **Environment Variables**
```env
MODEL_PATH=./model/WLASL20c_model.h5
USE_DUMMY_MODEL=0
INFER_WORKERS=2
```

#### 2. **Docker Deployment**
```bash
# Build image
docker build -t asl-backend .

# Run container
docker run -p 8000:8000 asl-backend
```

#### 3. **Production Server**
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
```

**Important**: Use only 1 worker for Socket.IO (sticky sessions not needed with 1 worker)

---

### Frontend Deployment (Next.js)

#### 1. **Vercel/Netlify**
```env
NEXT_PUBLIC_SOCKETIO_URL=https://your-backend.railway.app
```

#### 2. **CORS Configuration**
Ensure backend `CORSMiddleware` allows your frontend domain:
```python
# app/server.py
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 3. **WebSocket Transport**
If deploying on platforms with WebSocket restrictions (e.g., some CDNs), ensure Socket.IO falls back to polling:
```typescript
const socket = io(socketUrl, {
  transports: ['websocket', 'polling'], // Try WebSocket first, fallback to polling
});
```

---

## Error Handling & Best Practices

### 1. **Connection Error Handling**

```typescript
useEffect(() => {
  if (!socket) return;

  socket.on('connect_error', (error) => {
    console.error('[Connection Error]', error.message);
    // Show user-friendly error
    setError('Failed to connect to ASL recognition server');
  });

  socket.on('reconnect_failed', () => {
    console.error('[Reconnection Failed]');
    setError('Unable to reconnect. Please refresh the page.');
  });

  return () => {
    socket.off('connect_error');
    socket.off('reconnect_failed');
  };
}, [socket]);
```

---

### 2. **Frame Rate Optimization**

```typescript
// Adaptive frame rate based on connection quality
const [frameRate, setFrameRate] = useState(25);

useEffect(() => {
  if (socket) {
    socket.on('pong', (latency) => {
      if (latency > 100) {
        setFrameRate(15); // Reduce frame rate if latency high
      } else {
        setFrameRate(25);
      }
    });

    // Send ping periodically
    const pingInterval = setInterval(() => {
      socket.emit('ping');
    }, 1000);

    return () => clearInterval(pingInterval);
  }
}, [socket]);
```

---

### 3. **Memory Leak Prevention**

```typescript
useEffect(() => {
  // Cleanup on unmount
  return () => {
    stopStreaming();
    if (socket) {
      socket.disconnect();
    }
  };
}, []);
```

---

### 4. **Camera Permission Handling**

```typescript
const checkCameraPermission = async () => {
  try {
    const permission = await navigator.permissions.query({ name: 'camera' as PermissionName });
    
    if (permission.state === 'denied') {
      alert('Camera access denied. Please enable it in browser settings.');
      return false;
    }
    return true;
  } catch (error) {
    // Fallback for browsers that don't support permission query
    return true;
  }
};
```

---

### 5. **Performance Monitoring**

```typescript
const [metrics, setMetrics] = useState({
  framesProcessed: 0,
  avgLatency: 0,
  fps: 0,
});

useEffect(() => {
  if (!isStreaming) return;

  const startTime = Date.now();
  let frameCount = 0;

  const metricsInterval = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    const fps = frameCount / elapsed;

    setMetrics({
      framesProcessed: frameCount,
      avgLatency: 0, // Calculate from ping/pong
      fps: Math.round(fps),
    });

    frameCount++;
  }, 1000);

  return () => clearInterval(metricsInterval);
}, [isStreaming]);
```

---

## Complete Integration Checklist

### Backend Setup
- [ ] Install Python dependencies (`pip install -r requirements.txt`)
- [ ] Place model file at `./model/WLASL20c_model.h5`
- [ ] Configure environment variables
- [ ] Test backend: `uvicorn app.server:app --reload`
- [ ] Verify endpoints: `GET /healthz`, `GET /labels`
- [ ] Test Socket.IO connection with `client_webcam.py`

### Frontend Setup
- [ ] Install Socket.IO client: `npm install socket.io-client`
- [ ] Set `NEXT_PUBLIC_SOCKETIO_URL` in `.env.local`
- [ ] Implement Socket.IO connection logic
- [ ] Request camera permissions
- [ ] Implement frame capture and base64 encoding
- [ ] Handle `prediction` events
- [ ] Add error handling for disconnections
- [ ] Test with backend running locally

### Testing
- [ ] Test on Chrome/Firefox/Safari
- [ ] Test on mobile browsers (iOS Safari, Chrome Mobile)
- [ ] Verify frame rate performance (should maintain ~25 FPS)
- [ ] Test with poor network conditions
- [ ] Verify all 20 signs are recognized correctly
- [ ] Test reconnection logic

### Production
- [ ] Deploy backend to cloud (Railway, Render, AWS, etc.)
- [ ] Update CORS settings with production frontend URL
- [ ] Configure HTTPS for both frontend and backend
- [ ] Test WebSocket connectivity on production
- [ ] Monitor latency and adjust frame rate if needed
- [ ] Set up error logging (Sentry, LogRocket, etc.)

---

## Troubleshooting

### Issue: "WebSocket connection failed"
**Solution**: Check if backend is running and CORS is configured correctly. Try using polling transport:
```typescript
const socket = io(url, { transports: ['polling'] });
```

### Issue: "Camera not accessible"
**Solution**: Ensure HTTPS in production (getUserMedia requires secure context). For localhost, HTTP is allowed.

### Issue: "Predictions always show 'none'"
**Solution**: 
- Verify model file exists at `./model/WLASL20c_model.h5`
- Check backend logs for prediction errors
- Ensure frames are being sent correctly (base64 JPEG format)

### Issue: High latency/lag
**Solution**: 
- Reduce frame rate (15 FPS instead of 25)
- Compress JPEG quality (0.6 instead of 0.8)
- Deploy backend closer to users (use CDN/edge functions)

---

## Summary of Triggers & Endpoints

| Type | Endpoint/Event | Direction | Purpose | Data Format |
|------|---------------|-----------|---------|-------------|
| HTTP | `GET /healthz` | Frontend → Backend | Check server health | `{status: "ok", model: "..."}` |
| HTTP | `GET /labels` | Frontend → Backend | Get supported signs | `{0: "book", 1: "chair", ...}` |
| Socket.IO | `connect` | Auto (bidirectional) | Establish connection | N/A |
| Socket.IO | `status` | Backend → Frontend | Server ready signal | `{message: "ready"}` |
| Socket.IO | `frame` | Frontend → Backend | Send video frame | `{image: "base64..."}` |
| Socket.IO | `prediction` | Backend → Frontend | Receive prediction | `{label: "hello", score: 0.92}` |
| Socket.IO | `disconnect` | Auto (bidirectional) | Close connection | N/A |

---

**End of Integration Guide**

For questions or issues, check:
- Backend logs: `uvicorn app.server:app --log-level debug`
- Browser console: Network tab for WebSocket activity
- Model predictions: Verify `WLASL20c_model.h5` is loaded correctly
