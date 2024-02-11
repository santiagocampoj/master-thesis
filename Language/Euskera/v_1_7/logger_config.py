import logging

def setup_file_logging(log_file_path):
    logger = logging.getLogger("audio_processing")
    logger.setLevel(logging.INFO)  # whatever level you want

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
