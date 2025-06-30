'''
SARIMA Model Training Script

This script:
- Loads preprocessed data
- Fits SARIMA (Seasonal ARIMA) models for CPU and memory usage separately
- Forecasts the next 4 hours (48 steps at 5-min intervals)
- Logs metrics and artifacts to MLflow for tracking and visualization
- Saves models as pickle files for inference

'''

import os
import pandas as pd
import pickle
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
from itertools import product
from src.logger_setup import Log
import mlflow
import mlflow.pyfunc

# Suppress statsmodels and MLflow warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')
warnings.filterwarnings('ignore', message='No supported index is available')
warnings.filterwarnings('ignore', message='.*index.*', module='statsmodels')
warnings.filterwarnings('ignore', message='.*frequency.*', module='statsmodels')

class SARIMAForecaster:
    def __init__(self, file_path, forecast_steps=48):
        self.file_path = file_path
        self.forecast_steps = forecast_steps
        self.logger = Log.setup_logging()
    
    def check_stationarity(self, series):
        """
        Check if series is stationary using Augmented Dickey-Fuller test
        """
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            is_stationary = p_value < 0.05
            self.logger.info(f"Stationarity test p-value: {p_value:.4f}, Stationary: {is_stationary}", stacklevel=2)
            return is_stationary, p_value
        except Exception as e:
            self.logger.warning(f"Stationarity test failed: {e}", stacklevel=2)
            return False, 1.0
    
    def find_best_sarima_params(self, series, max_p=2, max_d=1, max_q=2, seasonal_periods=12):
        """
        Find best SARIMA parameters using grid search with AIC and forecast validation
        """
        try:
            self.logger.info("Searching for optimal SARIMA parameters...", stacklevel=2)
            
            # Check if differencing is needed
            is_stationary, p_value = self.check_stationarity(series)
            d_range = [0] if is_stationary else [0, 1]
            
            best_score = float('inf')
            best_params = ((1, 1, 1), (1, 1, 1, seasonal_periods))  # Force seasonal components
            
            # Expanded parameter ranges for better pattern detection
            # Non-seasonal: (p, d, q)
            # Seasonal: (P, D, Q, s) where s is seasonal period
            non_seasonal_combinations = list(product(range(max_p + 1), d_range, range(max_q + 1)))
            seasonal_combinations = list(product([1, 2], [0, 1], [1, 2]))  # Force seasonal components (no 0s)
            
            self.logger.info(f"Testing SARIMA models with seasonal period = {seasonal_periods} (12 * 5min = 1 hour)", stacklevel=2)
            
            for (p, d, q) in non_seasonal_combinations:
                if p == 0 and d == 0 and q == 0:
                    continue  # Skip (0,0,0)
                
                for (P, D, Q) in seasonal_combinations:
                    try:
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, seasonal_periods)
                        
                        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
                        fitted_model = model.fit(disp=False)
                        
                        # Test forecast to ensure it's reasonable
                        test_forecast = fitted_model.forecast(steps=5)
                        
                        # Check if forecast is reasonable (within 3x the data range for more flexibility)
                        data_min, data_max = series.min(), series.max()
                        data_range = data_max - data_min
                        forecast_min, forecast_max = test_forecast.min(), test_forecast.max()
                        
                        # Reject if forecast is way outside reasonable bounds (more lenient)
                        if (forecast_min < data_min - 2*data_range or 
                            forecast_max > data_max + 2*data_range):
                            self.logger.info(f"Rejecting SARIMA{order}x{seasonal_order} - unrealistic forecast range: {forecast_min:.1f} to {forecast_max:.1f}", stacklevel=2)
                            continue
                        
                        # Reject flat forecasts (all values too similar)
                        forecast_std = test_forecast.std()
                        if forecast_std < 0.1:  # If standard deviation is too low, it's essentially flat
                            self.logger.info(f"Rejecting SARIMA{order}x{seasonal_order} - flat forecast (std: {forecast_std:.3f})", stacklevel=2)
                            continue
                        
                        # Use AIC with reduced penalty to allow more complex seasonal models
                        penalty = (p + q + P + Q) * 2  # Reduced penalty to allow seasonal patterns
                        
                        # Bonus for models with seasonal components (encourage non-flat forecasts)
                        seasonal_bonus = 0
                        if P > 0 and Q > 0:  # Has both seasonal AR and MA components
                            seasonal_bonus = -15  # Strong bonus for full seasonal models
                        elif P > 0 or Q > 0:  # Has at least one seasonal component
                            seasonal_bonus = -10  # Moderate bonus
                        
                        score = fitted_model.aic + penalty + seasonal_bonus
                        
                        if score < best_score:
                            best_score = score
                            best_params = (order, seasonal_order)
                            
                    except Exception as e:
                        continue  # Skip problematic parameter combinations
            
            order, seasonal_order = best_params
            self.logger.info(f"Best SARIMA parameters: {order}x{seasonal_order} (Score: {best_score:.2f})", stacklevel=2)
            return best_params
            
        except Exception as e:
            self.logger.warning(f"Parameter search failed: {e}, using seasonal default", stacklevel=2)
            return ((1, 1, 1), (1, 1, 1, seasonal_periods))

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, index_col='timestamp', parse_dates=True)
            # Try to infer frequency, fallback to None if not regular
            try:
                self.df.index.freq = pd.infer_freq(self.df.index)
                if self.df.index.freq is None:
                    self.df.index.freq = '5min'
            except:
                self.df.index.freq = None
            self.logger.info(f"Loaded preprocessed data from {self.file_path}", stacklevel=2)
        except Exception as e:
            self.logger.critical(f"Failed to load data: {e}", stacklevel=2)
            raise

    def train_and_forecast(self, column_name):
        """
        Train SARIMA and forecast for a specific column, log to MLflow.
        """
        try:
            y = self.df[column_name].dropna()
            # Try to set frequency for statsmodels, but don't fail if it can't be inferred
            try:
                if y.index.freq is None:
                    inferred_freq = pd.infer_freq(y.index)
                    if inferred_freq:
                        y.index.freq = inferred_freq
            except:
                # If frequency setting fails, continue without it
                pass
            self.logger.info(f"Training SARIMA for {column_name}...", stacklevel=2)

            with mlflow.start_run(run_name=f"SARIMA_{column_name}") as run:
                # Find optimal SARIMA parameters
                # Use 12 as seasonal period (12 * 5min = 1 hour cycle)
                best_params = self.find_best_sarima_params(y, seasonal_periods=12)
                order, seasonal_order = best_params
                p, d, q = order
                P, D, Q, s = seasonal_order
                
                # Log SARIMA hyperparameters
                mlflow.log_params({
                    "model": "SARIMA",
                    "order_p": p,
                    "order_d": d,
                    "order_q": q,
                    "seasonal_P": P,
                    "seasonal_D": D,
                    "seasonal_Q": Q,
                    "seasonal_period": s,
                    "forecast_steps": self.forecast_steps
                })

                # Fit SARIMA with optimized parameters
                model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
                try:
                    model_fit = model.fit(disp=False)
                    self.logger.info(f"SARIMA{order}x{seasonal_order} model fitted successfully for {column_name}", stacklevel=2)
                    
                    # Log model quality metrics
                    mlflow.log_metric("aic", model_fit.aic)
                    mlflow.log_metric("bic", model_fit.bic)
                    
                except Exception as fit_error:
                    self.logger.error(f"SARIMA{order}x{seasonal_order} fitting failed for {column_name}: {fit_error}", stacklevel=2)
                    # Try seasonal fallback (never pure ARIMA)
                    self.logger.info(f"Trying seasonal fallback SARIMA(1,1,1)x(1,0,1,12) for {column_name}", stacklevel=2)
                    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
                    model_fit = model.fit(disp=False)
                    mlflow.log_params({"fallback_used": True})

                # Log model summary as artifact
                model_summary = model_fit.summary().as_text()
                
                # Use local artifacts directory that matches Docker volume mapping
                artifact_dir = "./artifacts"
                os.makedirs(artifact_dir, exist_ok=True)
                summary_path = os.path.join(artifact_dir, f"{column_name}_sarima_summary.txt")

                with open(summary_path, "w") as f:
                    f.write(model_summary)
                mlflow.log_artifact(summary_path)

                # Save the trained model as pickle
                model_path = os.path.join(artifact_dir, f"{column_name}_sarima_model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model_fit, f)
                mlflow.log_artifact(model_path)

                # Also log the model using MLflow's generic model logging
                class SARIMAWrapper(mlflow.pyfunc.PythonModel):
                    def __init__(self, model):
                        self.model = model
                    
                    def predict(self, context, model_input):
                        # model_input should be number of steps to forecast
                        steps = model_input.iloc[0, 0] if hasattr(model_input, 'iloc') else model_input
                        return self.model.forecast(steps=int(steps))
                
                wrapped_model = SARIMAWrapper(model_fit)
                mlflow.pyfunc.log_model(
                    artifact_path=f"{column_name}_model",
                    python_model=wrapped_model,
                    registered_model_name=f"SARIMA_{column_name}"
                )

                # Forecast with confidence intervals for better predictions
                forecast_result = model_fit.get_forecast(steps=self.forecast_steps)
                forecast = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int()
                
                # Debug: Print forecast values
                self.logger.info(f"Raw forecast values: {forecast}", stacklevel=2)
                self.logger.info(f"Forecast has {len(forecast)} values, range: {forecast.min():.2f} to {forecast.max():.2f}", stacklevel=2)
                
                # Create forecast index
                forecast_index = pd.date_range(
                    start=y.index[-1] + pd.Timedelta(minutes=5),
                    periods=self.forecast_steps,
                    freq='5min'
                )
                
                # Fix: Extract values properly to avoid index mismatch
                if hasattr(forecast, 'values'):
                    forecast_values = forecast.values
                else:
                    forecast_values = np.array(forecast)
                forecast_series = pd.Series(forecast_values, index=forecast_index)

                hours = (self.forecast_steps * 5) / 60
                self.logger.info(f"Forecast for {column_name} for next {hours:.1f} hours ({self.forecast_steps * 5} minutes):", stacklevel=2)
                for i, (timestamp, value) in enumerate(forecast_series.items()):
                    if pd.isna(value):
                        self.logger.warning(f"  Step {i+1}: {timestamp} -> NaN (model failed to predict)", stacklevel=2)
                    else:
                        self.logger.info(f"  Step {i+1}: {timestamp} -> {value:.2f}%", stacklevel=2)

                # Log forecast CSV as artifact with confidence intervals
                forecast_df = forecast_series.reset_index()
                forecast_df.columns = ["timestamp", f"{column_name}_forecast"]
                
                # Add confidence intervals to the CSV
                forecast_df[f"{column_name}_forecast_lower"] = forecast_ci.iloc[:, 0].values
                forecast_df[f"{column_name}_forecast_upper"] = forecast_ci.iloc[:, 1].values

                csv_path = os.path.join(artifact_dir, f"{column_name}_forecast.csv")
                forecast_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path)

                # Log final forecasted values as metrics (skip NaN values)
                for idx, val in enumerate(forecast_series.values):
                    if not pd.isna(val):  # Only log non-NaN values
                        mlflow.log_metric(f"{column_name}_forecast_step_{idx+1}", val)
                    else:
                        self.logger.warning(f"Skipping NaN forecast value for {column_name} step {idx+1}", stacklevel=2)

                self.logger.info(f"Forecasting and logging complete for {column_name}", stacklevel=2)

                return forecast_series

        except Exception as e:
            self.logger.critical(f"Failed ARIMA forecast for {column_name}: {e}", stacklevel=2)
            raise

if __name__ == "__main__":
    # Set MLflow tracking URI to use local file system instead of remote server
    # This avoids permission issues with containerized MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "SARIMA_Forecasting"
    
    # Create experiment with local artifact location
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name, 
                artifact_location="./artifacts"
            )
            Log.setup_logging().info(f"Created new experiment: {experiment_name}", stacklevel=2)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        Log.setup_logging().warning(f"Using default experiment due to: {e}", stacklevel=2)
        mlflow.set_experiment(experiment_name)

    forecaster = SARIMAForecaster(
        file_path="data/preprocessed/system_metrics_preprocessed.csv",
        forecast_steps=48  # forecasting next 4 hours at 5-min intervals
    )

    forecaster.load_data()

    cpu_forecast = forecaster.train_and_forecast('cpu_usage_percent')
    mem_forecast = forecaster.train_and_forecast('memory_usage_percent')

    Log.setup_logging().info("Forecasting completed successfully", stacklevel=2)