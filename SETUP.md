# 🚀 Time-Series Forecasting System - Local Setup Guide

## 📋 What You'll Build

A **complete real-time server monitoring and forecasting system** with:
- 📊 **Beautiful Streamlit Dashboard** for interactive forecasting
- 🤖 **SARIMA ML Models** for CPU/Memory usage prediction with seasonal patterns
- 📈 **Grafana Monitoring** for real-time system metrics
- 🔬 **MLflow Tracking** for experiment management
- 🐳 **Docker Stack** with 7 integrated services

## 🎯 Quick Start (5 Minutes)

```bash
# 1. Setup directories (CRITICAL - prevents permission errors)
./setup-directories.sh

# 2. Start all services
docker-compose up -d

# 3. Run ML pipeline
python src/main.py

# 4. Open dashboard
# Visit: http://localhost:8501
```

## 🔧 Prerequisites

### System Requirements
- **Linux/macOS** (Ubuntu/Debian recommended)
- **Docker** and **Docker Compose** 
- **Python 3.8+** with pip
- **4GB RAM minimum** (8GB recommended)

### Install Docker (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER
# Log out and back in for group changes
```

### Install Docker (macOS)
```bash
# Install Docker Desktop from: https://docker.com/products/docker-desktop
# Or using Homebrew:
brew install --cask docker
```

## 📁 Step 1: Get the Project

```bash
# Clone the repository
git clone <https://github.com/akashdv25/Time-Series-Forecasting>
cd Time-series

# Verify you have all files
ls -la
# Should see: docker-compose.yml, app.py, src/, requirements.txt, etc.
```

## 🐍 Step 2: Python Environment

```bash
# Create virtual environment
python3 -m venv myvenv
source myvenv/bin/activate  # Linux/macOS
# myvenv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## 🐳 Step 3: Docker Setup (CRITICAL)

```bash
# IMPORTANT: Run this BEFORE docker-compose up
./setup-directories.sh

# This creates directories with proper permissions:
# - artifacts/ (ML models)
# - mlruns/ (experiment tracking)  
# - mlflow_db/ (database)
```

**Why this matters:** Without proper permissions, MLflow will fail with "Permission Denied" errors.

## 🚀 Step 4: Start Services

```bash
# Start all 7 services in background
docker-compose up -d

# Verify all containers are running
docker-compose ps

# Should show 7 services: streamlit, grafana, prometheus, influxdb, mlflow, telegraf, node-exporter
```

### Service Startup Time
- **Wait 30-60 seconds** for all services to initialize
- InfluxDB and MLflow need time to set up databases

## 📊 Step 5: Run ML Pipeline

```bash
# Activate Python environment (if not already)
source myvenv/bin/activate

# Run complete pipeline
python src/main.py
```

**What this does:**
1. 📥 **Ingestion**: Collects system metrics from InfluxDB
2. 🧹 **Preprocessing**: Cleans and prepares data
3. 🤖 **Training**: Trains SARIMA models for CPU/Memory forecasting
4. 🔮 **Inference**: Generates next 4-hour predictions

## 🌐 Step 6: Access Your Dashboard

| Service | URL | Purpose |
|---------|-----|---------|
| **🎯 Main Dashboard** | http://localhost:8501 | **Interactive Forecasting UI** |
| **📈 Grafana** | http://localhost:3000 | Real-time monitoring |
| **🔬 MLflow** | http://localhost:5000 | ML experiment tracking |
| **📊 Prometheus** | http://localhost:9090 | Metrics collection |
| **💾 InfluxDB** | http://localhost:8086 | Time-series database |

### Default Credentials
- **Grafana**: `admin` / `StrongAdminPassword123!`
- **InfluxDB**: `admin` / `StrongAdminPassword123!`

## ✅ Verification Checklist

### 1. Check Docker Services
```bash
docker-compose ps
# All services should show "Up" status
```

### 2. Check Data Pipeline
```bash
# Should exist after running pipeline
ls -la data/preprocessed/system_metrics_preprocessed.csv
ls -la artifacts/*_forecast.csv
ls -la artifacts/*_arima_model.pkl
```

### 3. Check Dashboard
- Visit http://localhost:8501
- Should show current CPU/Memory usage
- Should display forecast charts

### 4. Check Logs
```bash
# View recent pipeline logs
tail -f logging-info/logs.log

# Check for successful completion
grep "Pipeline completed successfully" logging-info/logs.log
```

## 🎨 Dashboard Features

Your Streamlit dashboard includes:

### 📊 **Metrics Cards**
- Current CPU/Memory usage
- Next 4-hour forecasts (showing first prediction step)
- Beautiful gradient styling

### 📈 **Interactive Charts**
- **CPU Analysis**: Historical vs forecasted CPU usage
- **Memory Analysis**: Historical vs forecasted memory usage
- **Forecasts Only**: Side-by-side comparison

### 🔮 **Smart Error Handling**
- Shows helpful messages if data/models missing
- Guides you through setup process
- Graceful fallbacks for missing components

## 🔧 Common Issues & Solutions

### Issue: Permission Denied (MLflow)
```bash
# Solution: Fix directory permissions
sudo chown -R $(id -u):$(id -g) artifacts mlruns mlflow_db
chmod -R 755 artifacts mlruns mlflow_db
```

### Issue: Docker Containers Won't Start
```bash
# Solution: Restart Docker services
docker-compose down
docker-compose up -d

# Check logs for specific errors
docker-compose logs <service-name>
```

### Issue: No Data in Dashboard
```bash
# Solution: Run the pipeline first
python src/main.py

# Check if data was created
ls -la data/preprocessed/
ls -la artifacts/
```

### Issue: InfluxDB Connection Failed
```bash
# Solution: Wait longer for InfluxDB to start
sleep 60
python src/main.py

# Or restart InfluxDB
docker-compose restart influxdb
```

### Issue: MLflow UI Shows "No Experiments"
```bash
# Solution: Run training to create experiments
python src/model_train.py

# Check MLflow is accessible
curl http://localhost:5000/health
```

## 🔄 Development Workflow

### Daily Usage
```bash
# 1. Start services (if not running)
docker-compose up -d

# 2. Activate Python environment
source myvenv/bin/activate

# 3. Run pipeline to get fresh forecasts
python src/main.py

# 4. View results in dashboard
# http://localhost:8501
```

### Making Changes
```bash
# Edit source code
vim src/model_train.py

# Test individual components
python src/ingestion.py
python src/model_train.py

# Run full pipeline
python src/main.py
```

### Viewing Experiments
- **MLflow UI**: http://localhost:5000
- **Artifacts**: Check `artifacts/` folder
- **Logs**: `tail -f logging-info/logs.log`

## 📁 Project Structure

```
Time-series/
├── 🚀 src/main.py              # Main pipeline runner
├── 📊 app.py                   # Streamlit dashboard
├── 🐳 docker-compose.yml       # Service orchestration
├── 📋 requirements.txt         # Python dependencies
├── ⚙️ setup-directories.sh     # Permission setup
├── 🐳 Dockerfile.streamlit     # Streamlit container config
├── 📄 setup.py                 # Package installation
├── 📄 example.env / .env       # Environment variables
├── 📁 src/                     # Source code
│   ├── __init__.py            # Package initialization
│   ├── ingestion.py           # Data collection
│   ├── pre_processing.py      # Data cleaning
│   ├── model_train.py         # ML training
│   ├── model_inference.py     # Predictions
│   └── logger_setup.py        # Logging configuration
├── 📁 data/                    # Data storage
│   ├── raw/                   # Raw metrics
│   └── preprocessed/          # Cleaned data
├── 📁 assets/                  # Static assets
│   └── images/                # UI screenshots & diagrams
│       ├── ui.png             # Dashboard screenshot
│       └── flow-1.png         # System flow diagram
├── 📁 artifacts/               # ML models & forecasts
├── 📁 mlruns/                  # MLflow experiment tracking
├── 📁 mlflow_db/               # MLflow database storage
├── 📁 logging-info/            # Application logs
├── 📁 grafana/                 # Grafana configuration
│   └── provisioning/          # Auto-provisioning configs
│       ├── dashboards/        # Dashboard definitions
│       │   ├── dashboard.yaml # Dashboard config
│       │   └── node-exporter-full.json # Node exporter dashboard
│       └── datasources/       # Data source configs
│           └── datasources.yml # InfluxDB connection
├── 📁 prometheus/              # Prometheus configuration
│   └── prometheus.yml         # Metrics collection config
├── 📁 telegraf/                # Telegraf configuration
│   └── telegraf.conf          # Metrics forwarding config
├── 📁 myvenv/                  # Python virtual environment
└── 📁 Time_series_forecasting.egg-info/ # Package metadata
```

## 🛑 Shutdown

```bash
# Stop all services
docker-compose down

# Remove volumes (optional - deletes data)
docker-compose down -v

# Deactivate Python environment
deactivate
```

## 🚀 Next Steps

### Automation
```bash
# Set up automated hourly retraining
crontab -e
# Add: 0 * * * * cd /path/to/Time-series && python src/main.py
```


### Model Improvements
- Experiment with different ARIMA parameters
- Try LSTM or Prophet models
- Add more metrics (disk, network)

## 🆘 Getting Help

### Check Service Status
```bash
docker-compose ps
docker-compose logs <service-name>
```

### Debug Pipeline
```bash
# Run individual steps
python src/ingestion.py
python src/pre_processing.py
python src/model_train.py
```

### Monitor Resources
```bash
# Check system resources
htop
df -h
docker stats
```

### View Logs
```bash
# Application logs
tail -f logging-info/logs.log

# Docker logs
docker-compose logs -f
```

---

## 🎉 Success!

If everything works correctly, you should have:
- ✅ **7 Docker services running**
- ✅ **Beautiful dashboard at http://localhost:8501**
- ✅ **ML models generating forecasts**
- ✅ **Real-time monitoring in Grafana**
- ✅ **Experiment tracking in MLflow**

**Your Time-Series Forecasting System is now running locally!** 🚀


