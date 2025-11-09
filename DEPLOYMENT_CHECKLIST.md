# 🚀 Deployment Checklist

**Date**: November 10, 2025  
**Critical Fixes Applied**: ✅ All 7 major issues resolved

---

## ⚡ Quick Start (30 seconds)

```bash
# 1. Reinstall dependencies (REQUIRED)
pip install -r requirements.txt

# 2. Run server
uvicorn app.server:app --host 0.0.0.0 --port 8000

# 3. Test in another terminal
python client_webcam.py
```

---

## 📋 Pre-Deployment Checklist

### ☑️ Required Steps

- [ ] **Reinstall dependencies** (CRITICAL - old packages had bugs)
  ```bash
  pip uninstall -y -r requirements.txt
  pip install -r requirements.txt
  ```

- [ ] **Verify model file exists**
  ```bash
  ls -lh model/WLASL20c_model.h5
  # Should show file size (e.g., 50MB+)
  ```

- [ ] **Test server starts without errors**
  ```bash
  uvicorn app.server:app --reload
  # Should see: "Model loaded: Sequential"
  # Should see: "Uvicorn running on http://0.0.0.0:8000"
  ```

- [ ] **Test health endpoint**
  ```bash
  curl http://localhost:8000/healthz
  # Should return: {"status":"ok","model":"Sequential"}
  ```

- [ ] **Test labels endpoint**
  ```bash
  curl http://localhost:8000/labels
  # Should return: {"0":"book","1":"chair",...}
  ```

### ☑️ Optional (Recommended)

- [ ] **Create .env file for configuration**
  ```bash
  cp .env.example .env
  # Edit .env with your settings
  ```

- [ ] **Set production CORS origins**
  ```bash
  # In .env
  CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
  ```

- [ ] **Configure logging level**
  ```bash
  # In .env
  LOG_LEVEL=info  # Use 'debug' for troubleshooting
  ```

- [ ] **Set client limit**
  ```bash
  # In .env
  MAX_CLIENTS=50  # Adjust based on server capacity
  ```

---

## 🧪 Testing Checklist

### Basic Tests (5 minutes)

- [ ] **Single client test**
  ```bash
  # Terminal 1: Start server
  uvicorn app.server:app --port 8000
  
  # Terminal 2: Run client
  python client_webcam.py
  
  # Expected: See "Connected" and predictions appearing
  ```

- [ ] **Check logs for errors**
  ```bash
  # Should see:
  # [INFO] [CONNECT] Client <id> connected
  # [INFO] [PREDICTION] Client <id>: hello (0.85)
  # Should NOT see excessive logging or errors
  ```

- [ ] **Disconnect/reconnect test**
  ```bash
  # Stop client (Ctrl+C)
  # Check logs: [INFO] [DISCONNECT] Client <id> disconnected and cleaned up
  # Restart client - should reconnect without issues
  ```

### Stress Tests (15 minutes)

- [ ] **Multiple concurrent clients** (if possible)
  ```bash
  # Run 3-5 client_webcam.py instances simultaneously
  # Monitor server CPU/memory
  # All should work without crashes
  ```

- [ ] **Extended run test**
  ```bash
  # Let server run for 15+ minutes with active clients
  # Monitor memory usage - should remain stable
  watch -n 5 'ps aux | grep uvicorn | grep -v grep'
  ```

- [ ] **Memory stability check**
  ```bash
  # Memory should NOT increase continuously
  # Some fluctuation is normal, but no unbounded growth
  ```

### Performance Tests

- [ ] **Check prediction latency**
  ```bash
  # Predictions should appear within 100-200ms
  # No multi-second delays
  ```

- [ ] **Verify frame rate**
  ```bash
  # Client should maintain ~25 FPS
  # No freezing or stuttering
  ```

---

## 🐛 What to Look For

### ✅ Good Signs
- ✅ Steady memory usage (slight fluctuations OK)
- ✅ CPU usage 40-60% during active predictions
- ✅ Predictions appearing smoothly
- ✅ Clean logs with minimal spam
- ✅ Clients can reconnect without issues
- ✅ Server uptime > 30 minutes without restart

### 🔴 Red Flags (If you see these, report immediately)
- 🔴 Memory continuously increasing
- 🔴 CPU at 100% constantly
- 🔴 Server crashes or restarts
- 🔴 Predictions stop coming
- 🔴 "Out of memory" errors
- 🔴 Excessive TensorFlow errors in logs

---

## 🔧 If Something Goes Wrong

### Issue: Server won't start

**Check**:
```bash
# 1. Dependencies installed?
pip list | grep -E "tensorflow|fastapi|socketio|opencv"

# 2. Model file exists?
ls -lh model/WLASL20c_model.h5

# 3. Port already in use?
lsof -i :8000  # On Linux/Mac
netstat -ano | findstr :8000  # On Windows

# 4. Python version?
python --version  # Should be 3.7+
```

**Fix**:
```bash
# Kill existing process
kill -9 $(lsof -ti:8000)  # Linux/Mac
# Or just use different port
uvicorn app.server:app --port 8001
```

---

### Issue: TensorFlow CUDA warnings/errors

**Fix**: Force CPU mode
```bash
# Option 1: Environment variable
export TF_CPP_MIN_LOG_LEVEL=2

# Option 2: Edit app/model_runtime.py (line 39)
# Uncomment: tf.config.set_visible_devices([], 'GPU')
```

---

### Issue: "Server at capacity" message

**Cause**: Too many clients connected

**Fix**:
```bash
# Increase limit
export MAX_CLIENTS=100
uvicorn app.server:app --port 8000
```

---

### Issue: Memory still growing

**Check**:
```bash
# Monitor for 10 minutes
watch -n 10 'ps aux | grep uvicorn | grep -v grep | awk "{print \$4, \$6}"'
```

**If growing**:
```bash
# 1. Verify you reinstalled requirements.txt
pip install -r requirements.txt --force-reinstall

# 2. Check if old code is running
git status  # Should show clean working tree

# 3. Restart server completely
pkill -f uvicorn
uvicorn app.server:app --port 8000
```

---

## 🚀 Production Deployment

### Environment Setup

```bash
# Create .env file
cat > .env << EOF
MODEL_PATH=./model/WLASL20c_model.h5
USE_DUMMY_MODEL=0
TF_INTER_OP_THREADS=1
TF_INTRA_OP_THREADS=2
TF_CPP_MIN_LOG_LEVEL=2
LOG_LEVEL=info
MAX_CLIENTS=50
CORS_ORIGINS=https://yourdomain.com
SOCKETIO_PING_TIMEOUT=60
SOCKETIO_PING_INTERVAL=25
EOF
```

### Run Command

```bash
# Production run (with env vars)
source .env  # Linux/Mac
# Or: export $(cat .env | xargs)  # Alternative

uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
```

### Docker Deployment

```bash
# Build
docker build -t asl-backend:latest .

# Run
docker run -d \
  --name asl-server \
  -p 8000:8000 \
  --restart unless-stopped \
  -e LOG_LEVEL=info \
  -e MAX_CLIENTS=50 \
  -e CORS_ORIGINS=https://yourdomain.com \
  asl-backend:latest

# Check logs
docker logs -f asl-server

# Check status
docker stats asl-server
```

---

## 📊 Monitoring

### Real-time Monitoring

```bash
# Terminal 1: Server logs
uvicorn app.server:app --log-level info

# Terminal 2: Memory usage
watch -n 5 'ps aux | grep uvicorn | grep -v grep'

# Terminal 3: CPU usage
top -p $(pgrep -f uvicorn)

# Terminal 4: Connection count
watch -n 1 'netstat -an | grep :8000 | grep ESTABLISHED | wc -l'
```

### Log Monitoring

```bash
# Follow logs for errors
tail -f /path/to/logs | grep -E "ERROR|WARNING|DISCONNECT"

# Count predictions per minute
tail -f /path/to/logs | grep PREDICTION | wc -l

# Watch for memory issues
tail -f /path/to/logs | grep -E "memory|OOM|killed"
```

---

## ✅ Success Criteria

Your deployment is successful if:

1. ✅ Server starts without errors
2. ✅ Clients can connect and get predictions
3. ✅ Memory usage is stable for 30+ minutes
4. ✅ CPU usage is reasonable (40-60% during activity)
5. ✅ No crashes or restarts
6. ✅ Predictions are accurate
7. ✅ Logs are clean (no excessive errors)
8. ✅ Multiple clients work simultaneously

---

## 📞 Support & Documentation

- **Detailed bug fixes**: See `BUGFIX_SUMMARY.md`
- **Implementation details**: See `IMPLEMENTATION_SUMMARY.md`
- **Frontend integration**: See `NEXTJS_INTEGRATION_GUIDE.md`
- **Environment config**: See `.env.example`
- **General info**: See `README.md`

---

## 🎯 Next Steps

After successful deployment:

1. Monitor for 24 hours in production
2. Set up error tracking (Sentry, LogRocket)
3. Configure automated health checks
4. Set up backup/failover servers
5. Implement authentication if needed
6. Add rate limiting per IP if needed
7. Set up monitoring dashboard (Grafana, etc.)

---

**Status**: Ready for deployment ✅  
**Last Updated**: November 10, 2025  
**Version**: 1.1.0 (Stability Update)
