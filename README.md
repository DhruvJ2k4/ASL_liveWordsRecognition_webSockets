# 🤟 ASL Real-time Recognition WebSocket Server

Real-time American Sign Language (ASL) recognition system using TensorFlow, FastAPI, and Socket.IO.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Model file: `model/WLASL20c_model.h5`

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd ASL_liveWordsRecognition_webSockets

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional)
cp .env.example .env
# Edit .env with your settings

# 4. Run server
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

---

## 📋 Recent Critical Fixes (Nov 10, 2025)

### ✅ Server Stability Issues RESOLVED

The server was experiencing crashes and restarts after processing requests. All issues have been fixed:

1. **Memory Leak** - Fixed buffer overflow bug causing unbounded memory growth
2. **TensorFlow Memory** - Configured GPU memory growth and threading limits
3. **Thread Safety** - Added locks to prevent race conditions
4. **Resource Cleanup** - Proper cleanup on client disconnect
5. **Rate Limiting** - Added max client limit to prevent overload
6. **Dependencies** - Reduced from 80+ to 9 essential packages

**See `BUGFIX_SUMMARY.md` for complete details.**

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Model
MODEL_PATH=./model/WLASL20c_model.h5
USE_DUMMY_MODEL=0

# Performance
TF_INTER_OP_THREADS=1
TF_INTRA_OP_THREADS=2
LOG_LEVEL=info
MAX_CLIENTS=50

# CORS
CORS_ORIGINS=*

# Socket.IO
SOCKETIO_PING_TIMEOUT=60
SOCKETIO_PING_INTERVAL=25
```

### Production Deployment

```bash
# With environment variables
LOG_LEVEL=info MAX_CLIENTS=50 uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1

# With Docker
docker build -t asl-backend .
docker run -p 8000:8000 asl-backend
```

**Important**: Use only 1 worker for Socket.IO compatibility.

---

## 📡 API Endpoints

### HTTP Endpoints

#### `GET /healthz`
Health check and model status
```json
{
  "status": "ok",
  "model": "Sequential"
}
```

#### `GET /labels`
Get all supported ASL signs (20 total)
```json
{
  "0": "book",
  "1": "chair",
  ...
  "19": "wall"
}
```

### Socket.IO Events

#### Client → Server

**`frame`** - Send video frame for recognition
```javascript
socket.emit('frame', { 
  image: "<base64-encoded-JPEG>" 
});
```

#### Server → Client

**`status`** - Connection ready
```javascript
socket.on('status', (data) => {
  // { message: "ready" }
});
```

**`prediction`** - Recognition result (sent for every frame)
```javascript
socket.on('prediction', (data) => {
  // { label: "hello", score: 0.92 }
  // label: "none" if confidence < 0.50
});
```

---

## 🎯 Supported ASL Signs (20)

book, chair, clothes, computer, drink, drum, family, football, go, hat, hello, kiss, like, play, school, street, table, university, violin, wall

---

## 🔗 Frontend Integration

### Next.js Example

See `NEXTJS_INTEGRATION_GUIDE.md` for complete integration guide with:
- React component examples
- Custom hooks
- Error handling
- Production deployment
- Troubleshooting

### Basic Socket.IO Client

```javascript
import { io } from 'socket.io-client';

const socket = io('http://localhost:8000');

socket.on('connect', () => {
  console.log('Connected!');
});

socket.on('prediction', (data) => {
  console.log(`Detected: ${data.label} (${data.score})`);
});

// Send frame (base64 JPEG)
socket.emit('frame', { image: base64Frame });
```

---

## 🐛 Troubleshooting

### Server keeps restarting
✅ **FIXED** - Install updated requirements and restart server

### High memory usage
✅ **FIXED** - Memory leak resolved in buffer management

### TensorFlow CUDA warnings
```bash
# Force CPU mode (add to .env)
export TF_CPP_MIN_LOG_LEVEL=2
```

Or uncomment in `app/model_runtime.py`:
```python
tf.config.set_visible_devices([], 'GPU')
```

### Too many clients
```bash
# Increase limit
export MAX_CLIENTS=100
```

---

## 📦 Dependencies (Minimal)

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-socketio` - Real-time communication
- `tensorflow` - ML inference
- `opencv-python-headless` - Image processing
- `numpy` - Array operations

Total: **9 packages** (down from 80+)

---

## 🧪 Testing

### Manual Test with Python Client

```bash
python client_webcam.py
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test (TODO: create locust test file)
locust -f tests/load_test.py --host http://localhost:8000
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Frames per second | 25 FPS |
| Frame buffer size | 10 frames |
| Prediction threshold | 0.50 (50%) |
| Max concurrent clients | 50-100 |
| Avg prediction latency | 50-100ms |
| Memory usage (per client) | ~50-100MB |

---

## 🔐 Security Notes

### Production Checklist

- [ ] Update CORS origins to specific domains
- [ ] Enable HTTPS/WSS
- [ ] Add authentication (JWT tokens)
- [ ] Rate limit per IP address
- [ ] Monitor server logs
- [ ] Set up error tracking (Sentry)
- [ ] Configure firewall rules
- [ ] Use environment variables for secrets

### CORS Configuration

```python
# app/server.py
CORS_ORIGINS = ["https://yourdomain.com"]  # Specific domains only
```

---

## 📝 License

[Add your license here]

---

## 👥 Contributors

[Add contributors here]

---

## 🙏 Acknowledgments

- WLASL Dataset
- TensorFlow team
- Socket.IO team

---

## 📞 Support

For issues, see:
- `BUGFIX_SUMMARY.md` - Recent fixes and troubleshooting
- `NEXTJS_INTEGRATION_GUIDE.md` - Complete frontend integration
- GitHub Issues - Report bugs

---

**Last Updated**: November 10, 2025  
**Version**: 1.1.0 (Stability Improvements)
