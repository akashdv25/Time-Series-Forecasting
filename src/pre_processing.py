"""
This module implements simple pre-processing for time series data:

Handle NaNs (forward fill + backfill if needed).
Convert timestamp to datetime index.
Save cleaned data to data/preprocessed/ folder.
"""

import pandas as pd
import os
from src.logger_setup import Log

class PreProcessor:
    def __init__(self, input_file, output_file):
        """
        Initializes PreProcessor with paths and logger.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.logger = Log.setup_logging()
        self.df = None

    def load_data(self):
        """
        Load CSV into DataFrame.
        """
        try:
            self.df = pd.read_csv(self.input_file)
            self.logger.info(f"Loaded data from {self.input_file}", stacklevel=2)
        except FileNotFoundError:
            self.logger.critical(f"Input file {self.input_file} not found.", stacklevel=2)
            raise

    def process(self):
        """
        Pre-process:
        - Parse timestamp
        - Set as index
        - Handle NaNs
        """
        try:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.set_index('timestamp', inplace=True)
            self.df.sort_index(inplace=True)

            # Handle NaNs with forward fill, then backfill for initial NaNs
            self.df.ffill(inplace=True)
            self.df.bfill(inplace=True)

            self.logger.info("Pre-processing completed: timestamp indexed and NaNs handled.", stacklevel=2)
        except Exception as e:
            self.logger.critical(f"Pre-processing failed: {e}", stacklevel=2)
            raise

    def save(self):
        """
        Save the preprocessed DataFrame to the output file.
        """
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            self.df.to_csv(self.output_file)
            self.logger.info(f"Preprocessed data saved to {self.output_file}", stacklevel=2)
        except Exception as e:
            self.logger.critical(f"Failed to save preprocessed data: {e}", stacklevel=2)
            raise

if __name__ == "__main__":
    preprocessor = PreProcessor(
        input_file="data/raw/system_metrics.csv",
        output_file="data/preprocessed/system_metrics_preprocessed.csv"
    )
    preprocessor.load_data()
    preprocessor.process()
    preprocessor.save()
