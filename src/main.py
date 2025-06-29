"""
Main Pipeline Runner

Simple script that runs the complete Time-Series Forecasting pipeline in sequence:
1. Data Ingestion (from InfluxDB)
2. Data Preprocessing (cleaning and preparation)
3. Model Training (ARIMA models with MLflow tracking)
4. Model Inference (demonstration of latest model usage)

Usage:
    python src/main.py
"""

import subprocess
import sys
from src.logger_setup import Log

def run_script(script_name):
    """
    Run a Python script and return success/failure
    """
    logger = Log.setup_logging()
    
    try:
        logger.info(f"Running {script_name}...", stacklevel=2)
        result = subprocess.run([sys.executable, f"src/{script_name}"], 
                              check=True, 
                              capture_output=True, 
                              text=True)
        logger.info(f"{script_name} completed successfully", stacklevel=2)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} failed with exit code {e.returncode}", stacklevel=2)
        logger.error(f"Error output: {e.stderr}", stacklevel=2)
        return False

def main():
    """
    Run the complete pipeline
    """
    logger = Log.setup_logging()
    logger.info("Starting Time-Series Forecasting Pipeline", stacklevel=2)
    
    scripts = [
        "ingestion.py",
        "pre_processing.py", 
        "model_train.py",
        "model_inference.py"
    ]
    
    for script in scripts:
        if not run_script(script):
            logger.error(f"Pipeline failed at {script}", stacklevel=2)
            sys.exit(1)
    
    logger.info("Pipeline completed successfully", stacklevel=2)

if __name__ == "__main__":
    main() 