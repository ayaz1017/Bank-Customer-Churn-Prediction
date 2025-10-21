import sys
from src.logger import logger

def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information including file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number: [{exc_tb.tb_lineno}] "
        f"with error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    Custom Exception class for detailed error tracking in the
    Bank Customer Churn Prediction project.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        # Log the exception when it occurs
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        # Example to test error handling
        1 / 0
    except Exception as e:
        raise CustomException(e, sys)
