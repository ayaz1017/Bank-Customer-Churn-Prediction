import logging
import os
from datetime import datetime

# Create a unique log filename based on current date and time
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Define logs directory inside your project root
logs_path = os.path.join(os.getcwd(), "logs")

# Create the logs folder if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(levelname)s in %(name)s [Line: %(lineno)d] - %(message)s",
    level=logging.INFO,
)

# Optional: Get a logger instance for your project
logger = logging.getLogger("BankCustomerChurn")

if __name__ == "__main__":
    logger.info("✅ Logging setup complete for Bank Customer Churn Prediction project.")
