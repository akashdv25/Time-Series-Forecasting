'''
This script:
 Loads preprocessed data.
 Fits non-seasonal ARIMA for CPU and memory usage separately.
 Forecasts the next 20 minutes (4 steps at 5-min intervals).
 Logs metrics and artifacts to MLflow for tracking and visualization.
 Prints forecasts to the console.

'''

import os
import pandas as pd
import pickle
import warnings
from statsmodels.tsa.arima.model import ARIMA
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

class ARIMAForecaster:
    def __init__(self, file_path, forecast_steps=4):
        self.file_path = file_path
        self.forecast_steps = forecast_steps
        self.logger = Log.setup_logging()

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
        Train ARIMA and forecast for a specific column, log to MLflow.
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
            self.logger.info(f"Training ARIMA for {column_name}...", stacklevel=2)

            with mlflow.start_run(run_name=f"ARIMA_{column_name}") as run:
                # Log ARIMA hyperparameters
                mlflow.log_params({
                    "model": "ARIMA",
                    "order_p": 1,
                    "order_d": 1,
                    "order_q": 1,
                    "forecast_steps": self.forecast_steps
                })

                # Fit ARIMA with better error handling
                model = ARIMA(y, order=(1, 1, 1))
                try:
                    model_fit = model.fit()
                    self.logger.info(f"ARIMA model fitted successfully for {column_name}", stacklevel=2)
                except Exception as fit_error:
                    self.logger.error(f"ARIMA fitting failed for {column_name}: {fit_error}", stacklevel=2)
                    # Try simpler model
                    self.logger.info(f"Trying simpler ARIMA(1,0,0) for {column_name}", stacklevel=2)
                    model = ARIMA(y, order=(1, 0, 0))
                    model_fit = model.fit()

                # Log model summary as artifact
                model_summary = model_fit.summary().as_text()
                
                # Use local artifacts directory that matches Docker volume mapping
                artifact_dir = "./artifacts"
                os.makedirs(artifact_dir, exist_ok=True)
                summary_path = os.path.join(artifact_dir, f"{column_name}_arima_summary.txt")

                with open(summary_path, "w") as f:
                    f.write(model_summary)
                mlflow.log_artifact(summary_path)

                # Save the trained model as pickle
                model_path = os.path.join(artifact_dir, f"{column_name}_arima_model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model_fit, f)
                mlflow.log_artifact(model_path)

                # Also log the model using MLflow's generic model logging
                class ARIMAWrapper(mlflow.pyfunc.PythonModel):
                    def __init__(self, model):
                        self.model = model
                    
                    def predict(self, context, model_input):
                        # model_input should be number of steps to forecast
                        steps = model_input.iloc[0, 0] if hasattr(model_input, 'iloc') else model_input
                        return self.model.forecast(steps=int(steps))
                
                wrapped_model = ARIMAWrapper(model_fit)
                mlflow.pyfunc.log_model(
                    artifact_path=f"{column_name}_model",
                    python_model=wrapped_model,
                    registered_model_name=f"ARIMA_{column_name}"
                )

                # Forecast - use same method as inference for consistency
                forecast = model_fit.forecast(steps=self.forecast_steps)
                
                # Debug: Print forecast values
                self.logger.info(f"Raw forecast values: {forecast}", stacklevel=2)
                
                # Create forecast index
                forecast_index = pd.date_range(
                    start=y.index[-1] + pd.Timedelta(minutes=5),
                    periods=self.forecast_steps,
                    freq='5min'
                )
                forecast_series = pd.Series(forecast, index=forecast_index)

                self.logger.info(f"Forecast for {column_name} for next {self.forecast_steps * 5} minutes:", stacklevel=2)
                for i, (timestamp, value) in enumerate(forecast_series.items()):
                    if pd.isna(value):
                        self.logger.warning(f"  Step {i+1}: {timestamp} -> NaN (model failed to predict)", stacklevel=2)
                    else:
                        self.logger.info(f"  Step {i+1}: {timestamp} -> {value:.2f}%", stacklevel=2)

                # Log forecast CSV as artifact
                forecast_df = forecast_series.reset_index()
                forecast_df.columns = ["timestamp", f"{column_name}_forecast"]

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
    experiment_name = "ARIMA_Forecasting"
    
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

    forecaster = ARIMAForecaster(
        file_path="data/preprocessed/system_metrics_preprocessed.csv",
        forecast_steps=4  # forecasting next 20 minutes at 5-min intervals
    )

    forecaster.load_data()

    cpu_forecast = forecaster.train_and_forecast('cpu_usage_percent')
    mem_forecast = forecaster.train_and_forecast('memory_usage_percent')

    Log.setup_logging().info("Forecasting completed successfully", stacklevel=2)