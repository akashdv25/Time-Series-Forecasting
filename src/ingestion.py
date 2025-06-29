import pandas as pd
from influxdb_client import InfluxDBClient
from src.logger_setup import Log
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Ingestion:
    def __init__(self):
        """
        Initializes the Ingestion class with InfluxDB details and sets up logging.
        """
        try:
            self.logger = Log.setup_logging()

            self.url = "http://localhost:8086"
            self.token = os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN")
            self.org = os.getenv("DOCKER_INFLUXDB_INIT_ORG")
            self.bucket = os.getenv("DOCKER_INFLUXDB_INIT_BUCKET")
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

            self.logger.info("Ingestion class initialized successfully", stacklevel=2)

        except Exception as e:
            self.logger.critical(f"Ingestion class failed to initialize: {e}", stacklevel=2)
            raise

    def load_csv(self):
        """
        Checks if system_metrics.csv exists inside data/raw.
        If yes, uses the latest timestamp as the starting point for new data.
        If not, fetches the last 1 hour of data.
        """
        try:
            self.df_existing = pd.read_csv("data/raw/system_metrics.csv", parse_dates=["timestamp"])
            last_timestamp = self.df_existing["timestamp"].max()

            if pd.isna(last_timestamp):
                self.logger.warning("CSV exists but no valid timestamp found. Pulling last 1 hour.", stacklevel=2)
                self.last_timestamp = "-1h"
            else:
                self.logger.info(f"Last saved timestamp: {last_timestamp}", stacklevel=2)
                self.last_timestamp = last_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

        except FileNotFoundError:
            self.logger.warning("No existing CSV found, pulling last 1 hour of data.", stacklevel=2)
            self.last_timestamp = "-1h"
            self.df_existing = pd.DataFrame()

    def create_csv(self):
        """
        Queries InfluxDB for CPU and RAM usage data,
        appends new records to system_metrics.csv or creates it if not present.
        """
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: {self.last_timestamp})
        |> filter(fn: (r) => r._measurement == "cpu" or r._measurement == "mem")
        |> filter(fn: (r) => r._field == "usage_active" or r._field == "used_percent")
        |> keep(columns: ["_time", "_field", "_value"])
        '''

        try:
            query_api = self.client.query_api()
            tables = query_api.query(query, org=self.org)

            data = []
            for table in tables:
                for record in table.records:
                    data.append({
                        "timestamp": record.get_time(),
                        "metric": "cpu_usage_percent" if record.get_field() == "usage_active" else "memory_usage_percent",
                        "value": record.get_value()
                    })

            df_new = pd.DataFrame(data)

            if not df_new.empty:
                df_pivot = df_new.pivot(index='timestamp', columns='metric', values='value').reset_index()
                df_pivot.sort_values('timestamp', inplace=True)

                try:
                    df_existing = pd.read_csv("data/raw/system_metrics.csv", parse_dates=["timestamp"])
                    df_combined = pd.concat([df_existing, df_pivot], ignore_index=True)
                    df_combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                    df_combined.sort_values("timestamp", inplace=True)
                except FileNotFoundError:
                    df_combined = df_pivot

                df_combined.to_csv("data/raw/system_metrics.csv", index=False)
                self.logger.info(f"{len(df_pivot)} new rows appended to system_metrics.csv", stacklevel=2)

            else:
                self.logger.info("No new data found in InfluxDB.", stacklevel=2)

        except Exception as e:
            self.logger.error(f"Failed during create_csv execution: {e}", stacklevel=2)
            raise

        finally:
            self.client.close()
            self.logger.info("InfluxDB client connection closed.", stacklevel=2)

if __name__ == '__main__':
    ingestion_obj = Ingestion()
    ingestion_obj.load_csv()
    ingestion_obj.create_csv()
