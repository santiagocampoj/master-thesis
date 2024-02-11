import logging

def setup_file_logging(log_file_path):
    # Create a custom logger
    logger = logging.getLogger("audio_processing")
    logger.setLevel(logging.INFO)  # Or whatever level you want

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    # Avoid logging to the terminal by setting propagate to False
    logger.propagate = False

    return logger
