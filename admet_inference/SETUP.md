# ADMET Inference System - Complete Setup Guide

**Version**: 2.0.0 (Inference Only - CPU-Optimized with Multi-Threading)  
**Last Updated**: 2026-04-20

This guide provides complete instructions for running the ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) **inference** system both locally and in Docker. 

⚠️ **IMPORTANT**: This system is **INFERENCE ONLY**. Model training is handled separately using Jupyter notebooks (.ipynb files).

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Local Setup & Run](#local-setup--run)
5. [Docker Setup & Run](#docker-setup--run)
6. [API Usage](#api-usage)
7. [Performance & Benchmarks](#performance--benchmarks)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### What This System Does
- ✅ Loads pre-trained ADMET models
- ✅ Performs inference on molecules (SMILES strings)
- ✅ Multi-threaded batch processing for speed
- ✅ CPU-optimized (no GPU required)
- ❌ Does NOT train models (use Jupyter notebooks for training)

### Architecture
```
Docker Container (or Local Python)
    ↓
FastAPI Server (http://localhost:8000)
    ↓
Async/Multi-threaded Batch Processor
    ↓
5 Pre-trained MPNN Models (Absorption, Distribution, Metabolism, Excretion, Toxicity)
    ↓
REST API Responses (JSON)
```

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Local Setup & Run](#local-setup--run)
4. [Docker Setup & Run](#docker-setup--run)
5. [API Usage](#api-usage)
6. [Troubleshooting](#troubleshooting)
7. [Performance Notes](#performance-notes)

---

## System Requirements

### Minimum Requirements
- **CPU**: 1 core (2+ cores recommended)
- **RAM**: 2 GB
- **Disk**: 3 GB free space
- **OS**: Windows, macOS, or Linux

### Optional
- **Docker**: For containerized deployment
- **Git**: For version control

---

## Quick Start

### Option A: Quick Run with Docker (Recommended)
```bash
cd admet_inference
docker-compose up -d
```
Then visit: **http://localhost:8000/docs**

### Option B: Quick Run Locally
```bash
cd admet_inference
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Then visit: **http://localhost:8000/docs**

---

## Local Setup & Run

### Step 1: Setup Python Virtual Environment

**Windows (PowerShell):**
```powershell
cd admet_inference
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux (Bash):**
```bash
cd admet_inference
python -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected installation time**: 3-5 minutes

### Step 3: Verify Installation

```bash
python -c "import torch; import chemprop; print('✓ All dependencies installed')"
```

### Step 4: Start the API Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Output** (expected):
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 5: Test the API

**In a new terminal/PowerShell:**

```bash
# Single molecule prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'

# Or using PowerShell:
$body = @{"smiles"="CC(=O)OC1=CC=CC=C1C(=O)O"} | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:8000/predict" `
  -Method POST -Body $body -ContentType "application/json"
```

**Response** (example):
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "valid": true,
  "predictions": {
    "Absorption": -5.15,
    "Distribution": 0.45,
    "Metabolism": 0.72,
    "Excretion": 0.50,
    "Toxicity": 0.25
  },
  "status": {
    "Absorption": "Good",
    "Distribution": "BBB-",
    "Metabolism": "Substrate",
    "Excretion": "Stable",
    "Toxicity": "Safe"
  }
}
```

### Step 6: Stop the Server

Press `Ctrl+C` in the terminal running the server.

---

## Docker Setup & Run

### Prerequisites

- **Docker**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Docker Compose**: Included in Docker Desktop

**Verify Installation:**
```bash
docker --version
docker-compose --version
```

### Option 1: Using Docker Compose (Simple)

#### Build Image
```bash
cd admet_inference
docker-compose build
```

#### Start Service
```bash
docker-compose up -d
```

#### View Logs
```bash
docker-compose logs -f admet-api
```

#### Stop Service
```bash
docker-compose down
```

#### Test the API
```bash
# Same curl/PowerShell commands as local setup
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

### Option 2: Using Docker Commands Directly

#### Build Image
```bash
cd admet_inference
docker build -t admet-inference:latest .
```

**Expected output:**
```
[+] Building 120.5s (14/14) FINISHED
```

#### Run Container
```bash
docker run -d \
  --name admet-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --restart unless-stopped \
  admet-inference:latest
```

**Windows PowerShell** (use this instead):
```powershell
docker run -d `
  --name admet-api `
  -p 8000:8000 `
  -v "${PWD}/models:/app/models:ro" `
  --restart unless-stopped `
  admet-inference:latest
```

#### View Container Logs
```bash
docker logs -f admet-api
```

#### Stop Container
```bash
docker stop admet-api
docker rm admet-api
```

#### Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccccc1"}'
```

---

## API Usage

### Web Interface (Swagger UI)
- **URL**: http://localhost:8000/docs
- Browse and test endpoints interactively

### ReDoc Documentation
- **URL**: http://localhost:8000/redoc

### Endpoints

#### 1. Single Prediction
```bash
POST /predict
```

**Request:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "return_probabilities": false
}
```

**Response:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "valid": true,
  "predictions": {
    "Absorption": -5.15,
    "Distribution": 0.45,
    "Metabolism": 0.72,
    "Excretion": 0.50,
    "Toxicity": 0.25
  },
  "status": {
    "Absorption": "Good",
    "Distribution": "BBB-",
    "Metabolism": "Substrate",
    "Excretion": "Stable",
    "Toxicity": "Safe"
  }
}
```

#### 2. Batch Prediction
```bash
POST /predict_batch
```

**Request:**
```json
{
  "smiles_list": [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CCO",
    "c1ccccc1"
  ]
}
```

**Response:**
```json
{
  "results": [
    { "smiles": "...", "predictions": {...}, "status": {...} },
    { "smiles": "...", "predictions": {...}, "status": {...} }
  ],
  "count": 3
}
```

#### 3. Health Check
```bash
GET /health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "models_loaded": 5,
  "timestamp": "2026-04-20T10:30:00"
}
```

#### 4. Model Status
```bash
GET /status
```

**Response:**
```json
{
  "status": "ready",
  "models": {
    "Absorption": true,
    "Distribution": true,
    "Metabolism": true,
    "Excretion": true,
    "Toxicity": true
  }
}
```

### cURL Examples

**Single prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles":"c1ccccc1"}'
```

**Batch prediction:**
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"smiles_list":["CCO","c1ccccc1","CC(C)C"]}'
```

**Health check:**
```bash
curl "http://localhost:8000/health"
```

### Python Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
)
print(response.json())

# Batch prediction
response = requests.post(
    f"{BASE_URL}/predict_batch",
    json={"smiles_list": ["CCO", "c1ccccc1", "CC(C)C"]}
)
print(response.json())

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())
```

---

## Troubleshooting

### Local Setup Issues

#### Issue: `ModuleNotFoundError: No module named 'torch'`
**Solution:**
```bash
pip install --upgrade torch
pip install -r requirements.txt
```

#### Issue: `Port 8000 already in use`
**Solution (Windows PowerShell):**
```powershell
# Find process using port 8000
Get-NetTCPConnection -LocalPort 8000

# Kill the process
Stop-Process -Id <PID> -Force
```

**Solution (Linux/macOS):**
```bash
lsof -i :8000
kill -9 <PID>
```

**Or use a different port:**
```bash
python -m uvicorn app.main:app --port 8001
```

#### Issue: `torch.cuda.OutOfMemoryError` or `RuntimeError: CUDA out of memory`
**Note:** This system is CPU-only. If you see GPU errors:
```bash
# Verify CPU mode
python -c "import torch; print(torch.device('cpu'))"
```

#### Issue: Models not found or not loading
**Solution:**
```bash
# Check model directory structure
ls admet_inference/models/

# Should show:
# Absorption/best_model.ckpt
# Distribution/best_model.ckpt
# Metabolism/best_model.ckpt
# Excretion/best_model.ckpt
# Toxicity/best_model.ckpt
```

If models are missing, ensure they're in the correct paths.

### Docker Issues

#### Issue: `docker: command not found`
**Solution:** Install Docker Desktop from https://www.docker.com

#### Issue: `Cannot connect to Docker daemon`
**Solution:** Start Docker Desktop

#### Issue: `docker-compose: command not found`
**Solution:** Use `docker compose` (newer syntax) instead:
```bash
docker compose up -d
```

#### Issue: Port 8000 already in use
**Solution:**
```bash
# Kill container using the port
docker stop $(docker ps -q -f ancestor=admet-inference:latest)

# Or use a different port in docker-compose.yml:
# Change ports: - "8001:8000"
```

#### Issue: `Out of memory` or slow predictions
**Solution:** Check resource allocation:
```bash
docker stats admet-api
```

Increase memory in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

#### Issue: API returns 500 error
**Solution:** Check logs:
```bash
# Local
tail -f logs/*.log

# Docker
docker logs -f admet-api
```

---

## Performance Notes

### CPU Performance

**Expected prediction times (CPU):**
- **Single molecule**: 0.5-1.0 seconds
- **Batch (10 molecules)**: 5-10 seconds
- **Batch (100 molecules)**: 50-100 seconds

**Tips for better performance:**
1. **Use batch predictions** (more efficient than single calls)
2. **Increase workers** in docker-compose.yml if needed:
   ```bash
   # Change uvicorn command:
   CMD ["python", "-m", "uvicorn", "app.main:app", "--workers", "4"]
   ```
3. **Monitor resource usage**:
   ```bash
   docker stats
   ```

## Performance Notes

### CPU Performance with Multi-Threading

**Expected prediction times (CPU, with async multi-threading):**
- **Single molecule**: 0.5-1.0 seconds
- **Batch (10 molecules)**: 2-3 seconds (3-5x faster with async!)
- **Batch (100 molecules)**: 15-25 seconds (4-5x faster with async!)
- **Batch (1000 molecules)**: 150-250 seconds (4-5x faster with async!)

### Multi-Threading Benefits

```
Sequential Processing:     Async Multi-Threading:
[Mol1] ← 0.5s            [Mol1, Mol2, Mol3, Mol4] ← Running in parallel
[Mol2] ← 0.5s            (4 threads) ← 0.5-1.0s total
[Mol3] ← 0.5s
[Mol4] ← 0.5s
Total: 2.0s              Total: 1.0s (2x faster!)
```

### Enabling Multi-Threading

**Asyncmulti-threaded processing is ENABLED by default** in batch predictions:

```bash
# Auto-uses async for 10+ molecules:
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles_list": ["CCO", "c1ccccc1", "CC(C)C"],
    "use_async": true
  }'
```

### Image Size

Current Docker image size: **~800-900 MB**

Breakdown:
- Python runtime: ~200 MB
- Dependencies: ~300 MB  
- Models: ~300-400 MB

---

## Advanced Configuration

### Environment Variables (Docker)

Edit `docker-compose.yml`:
```yaml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
  - WORKERS=2               # Number of uvicorn workers
  - ASYNC_WORKERS=4         # Number of threads for async processing
```

### Custom Settings (Local)

Edit `app/inference.py` to adjust thread pool size:
```python
predictor = ADMETPredictor(num_workers=4)  # Change number of threads
```

Then run:
```bash
export LOG_LEVEL=DEBUG
python -m uvicorn app.main:app
```

### Multi-Threading Configuration

In `app/inference.py`:
```python
# Number of worker threads for parallel inference
predictor = ADMETPredictor(num_workers=4)  # Default: 4 threads
```

Recommended values:
- **CPU cores = 2**: `num_workers=2`
- **CPU cores = 4**: `num_workers=4` (default)
- **CPU cores = 8**: `num_workers=8`
- **CPU cores = 16+**: `num_workers=12`

### Persistent Logs

**Docker:** Already configured in `docker-compose.yml`
- Logs written to: `./logs/`
- Retention: 10 MB per file, 3 files max

**Local:**
```bash
python -m uvicorn app.main:app >> logs/api.log 2>&1
```

---

## Training vs Inference

### Training (Separate Process)
- Use `.ipynb` Jupyter notebook files
- Requires GPU (optional but recommended)
- Outputs trained models as `.ckpt` files
- Takes hours/days depending on data size

### Inference (This System)
- Load pre-trained `.ckpt` models
- Make predictions on new molecules
- CPU-only, multi-threaded
- Real-time responses (milliseconds to seconds)

## Architecture Notes

This system is optimized for inference:
- ✅ **Removed PyTorch Lightning** (training framework)
- ✅ **Removed scikit-learn** (not needed for inference)
- ✅ **Added asyncio** (multi-threaded processing)
- ✅ **Added ThreadPoolExecutor** (parallel predictions)
- ✅ **CPU-only mode** (no GPU memory overhead)

---

## Deployment Checklist

- [ ] System requirements verified
- [ ] Python/Docker installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models present in `models/` directory
- [ ] Port 8000 available (or changed in config)
- [ ] API responds to `/health` endpoint
- [ ] Test prediction successful
- [ ] Logs accessible and monitored
- [ ] Resource limits set appropriately

---

## Support & Issues

### Quick Diagnostics

```bash
# Check system info
python --version
pip list | grep -E "torch|chemprop|fastapi"

# Test API connectivity
curl -v http://localhost:8000/health

# Check Docker
docker ps
docker logs admet-api

# View resource usage
docker stats admet-api
```

### Common Commands Reference

| Task | Command |
|------|---------|
| Start (Docker) | `docker-compose up -d` |
| Stop (Docker) | `docker-compose down` |
| View logs (Docker) | `docker-compose logs -f` |
| Start (Local) | `python -m uvicorn app.main:app` |
| Test API | `curl http://localhost:8000/health` |
| Rebuild image | `docker-compose build --no-cache` |
| Remove container | `docker rm admet-api` |

---

## File Structure

```
admet_inference/
├── SETUP.md                 ← YOU ARE HERE
├── Dockerfile              # Docker build configuration (CPU-optimized)
├── docker-compose.yml      # Docker Compose setup (CPU-optimized)
├── requirements.txt        # Python dependencies
├── README.md              # Original documentation
├── app/
│   ├── main.py            # FastAPI application
│   ├── inference.py       # Core prediction engine (CPU-only)
│   ├── utils.py           # Utility functions
│   └── __init__.py
├── models/                # Pre-trained model checkpoints
│   ├── Absorption/
│   ├── Distribution/
│   ├── Metabolism/
│   ├── Excretion/
│   └── Toxicity/
├── config/
│   └── nginx.conf         # (Optional reverse proxy config)
└── logs/                  # Output logs directory
```

---

**Last Updated**: 2026-04-20  
**System**: ADMET Prediction v1.0 (CPU-Optimized)
