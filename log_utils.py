import logging
import logging.handlers

log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

# create logger and set logging level
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create console handler and set formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))

# create file handler, set formatter and add to logger
file_handler = logging.handlers.RotatingFileHandler(
    filename="log.txt", backupCount=3, maxBytes=1024 * 1024 * 10)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

# add console handler to logger
logger.addHandler(console_handler)

