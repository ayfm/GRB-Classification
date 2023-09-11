import os
from typing import List
import coloredlogs
import logging

# get current directory of the package
DIR_PACKAGE = os.path.dirname(os.path.realpath(__file__))

# path of the catalogs
DIR_CATALOGS = os.path.join("..", "catalogs")

# path of the datasets
DIR_DATASETS = os.path.join("..", "datasets")

# path of the reports
DIR_REPORTS = os.path.join("..", "reports")

# path of the figures
DIR_FIGURES = os.path.join("..", "figures")

# path of the models
DIR_MODELS = os.path.join("..", "models")


# default log level
_LOG_LEVEL = "INFO"

# log formats
_LOG_FORMAT = "[%(levelname)-.1s]: %(message)s"

# the logger map;
# keys are name of the loggers, values are the logger objects
_loggers = dict()


# get logger
def get_logger():
    # return if logger is already created
    if "root" in _loggers:
        return _loggers["root"]

    # get root logger
    logger = logging.getLogger("root")
    # create stream handler
    stream_handler = logging.StreamHandler()
    # set log level
    stream_handler.setLevel(_LOG_LEVEL)
    # create formatter
    formatter = coloredlogs.ColoredFormatter(_LOG_FORMAT)
    # add formatter
    stream_handler.setFormatter(formatter)
    # add handler to the logger
    logger.addHandler(stream_handler)

    # set log level
    logger.setLevel(_LOG_LEVEL)

    # set root logger
    _loggers["root"] = logger

    return logger
