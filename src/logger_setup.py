import logging
import os

class Log:
    @staticmethod
    def setup_logging(log_dir="logging-info"):
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger("central_logger")
        logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if already added
        if not logger.hasHandlers():
            file_handler = logging.FileHandler(os.path.join(log_dir, "logs.log"), encoding="utf-8")
            
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


if __name__ == "__main__":
    logger = Log.setup_logging()
    # Use stacklevel=2 so it shows the caller's info, not inside this function
    logger.info("Logging setup completed successfully", stacklevel=2)