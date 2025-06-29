"""
Model Inference Script

This script loads and uses the trained ARIMA models that were saved as pickle files.
Simple, fast, and automatically gets the latest trained models.
"""

import os
import pickle
import pandas as pd
import warnings
from datetime import datetime
from src.logger_setup import Log

# Suppress all warnings for cleaner output during inference
warnings.simplefilter('ignore')

class ModelInference:
    def __init__(self):
        self.logger = Log.setup_logging()
        
    def load_model(self, model_name):
        """
        Load the latest trained model from pickle file
        """
        try:
            model_path = f"artifacts/{model_name}_arima_model.pkl"
            
            # Check if file exists and get its modification time
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            mod_time = os.path.getmtime(model_path)
            mod_datetime = datetime.fromtimestamp(mod_time)
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                
            self.logger.info(f"Loaded model: {model_path} (trained: {mod_datetime})", stacklevel=2)
            return model, mod_datetime
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}", stacklevel=2)
            raise
    
    def predict(self, model, steps=4):
        """
        Make predictions using the ARIMA model
        """
        try:
            forecast = model.forecast(steps=steps)
            self.logger.info(f"Generated forecast for {steps} steps", stacklevel=2)
            return forecast
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", stacklevel=2)
            raise
    
    def list_available_models(self):
        """
        List all available model files with their timestamps
        """
        try:
            artifacts_dir = "artifacts"
            if not os.path.exists(artifacts_dir):
                self.logger.warning("No artifacts directory found. Run model training first.", stacklevel=2)
                return []
            
            model_files = [f for f in os.listdir(artifacts_dir) if f.endswith('_arima_model.pkl')]
            
            if not model_files:
                self.logger.warning("No trained models found. Run model training first.", stacklevel=2)
                return []
            
            self.logger.info("Available Trained Models:", stacklevel=2)
            models_info = []
            
            for model_file in sorted(model_files):
                model_path = os.path.join(artifacts_dir, model_file)
                mod_time = os.path.getmtime(model_path)
                mod_datetime = datetime.fromtimestamp(mod_time)
                
                model_name = model_file.replace('_arima_model.pkl', '')
                self.logger.info(f"  {model_name} - Trained: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')} - File: {model_file}", stacklevel=2)
                
                models_info.append({
                    'name': model_name,
                    'file': model_file,
                    'trained_at': mod_datetime
                })
            
            return models_info
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}", stacklevel=2)
            return []
    
    def forecast_metric(self, metric_name, steps=4):
        """
        Generic method to forecast any metric (removes code duplication)
        """
        try:
            display_name = metric_name.replace('_', ' ').title()
            self.logger.info(f"Starting {display_name.lower()} forecasting", stacklevel=2)
            
            model, trained_at = self.load_model(metric_name)
            forecast = self.predict(model, steps=steps)
            
            self.logger.info(f"{display_name} model trained: {trained_at.strftime('%Y-%m-%d %H:%M:%S')}", stacklevel=2)
            self.logger.info(f"{display_name} forecast for next {steps * 5} minutes:", stacklevel=2)
            
            for i, value in enumerate(forecast):
                self.logger.info(f"  Step {i+1} (+{(i+1)*5} min): {value:.2f}%", stacklevel=2)
                
            return forecast, trained_at
            
        except Exception as e:
            self.logger.error(f"{metric_name} model failed: {e}", stacklevel=2)
            return None, None

    def run_inference(self):
        """
        Run inference on both CPU and memory models (optimized version)
        """
        self.logger.info("Starting model inference demonstration", stacklevel=2)
        
        # Simple existence check instead of full model listing
        if not os.path.exists("artifacts"):
            self.logger.error("No artifacts directory found. Please run model training first: python src/model_train.py", stacklevel=2)
            return
        
        # Show available models for information
        self.logger.info("Checking available models...", stacklevel=2)
        self.list_available_models()
        
        self.logger.info("Running predictions with latest models", stacklevel=2)
        
        # Use generic method for both metrics (eliminates code duplication)
        metrics = ["cpu_usage_percent", "memory_usage_percent"]
        
        for metric in metrics:
            self.forecast_metric(metric, steps=4)
        
        self.logger.info("Inference completed successfully", stacklevel=2)
        self.logger.info("To retrain models with new data, run: python src/model_train.py", stacklevel=2)

if __name__ == "__main__":
    inference = ModelInference()
    inference.run_inference() 