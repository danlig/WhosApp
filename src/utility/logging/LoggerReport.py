import os
import logging
import datetime
from collections.abc import Hashable
from ..config_path import LOGS_PATH, LOGS_ALIASES_PATH, LOGS_DATASET_PATH, LOGS_MODELS_PATH, LOGS_PIPELINE_PATH
from ..clean_coding.decorator import check_path_exists
from ..clean_coding.ensure import *


@check_path_exists("../logs/", create=True)
@check_path_exists("../logs/dataset/", create=True)
@check_path_exists("../logs/training", create=True)
@check_path_exists("../logs/pipeline", create=True)
@validation(
    "name",
    "Nome del file di log",
    ensure_valid_file_extension(".log"), ensure_not_none("unknown")
)
@validation(
    "start_message",
    "Messaggio di inizio log",
    ensure_valid_type(Hashable), ensure_not_none("!! START NEW LOG !!")
)
@validation(
    "path",
    "Percorso del file di log",
    ensure_into_allowed_options(
        options=[LOGS_PATH, LOGS_ALIASES_PATH, LOGS_DATASET_PATH, LOGS_MODELS_PATH, LOGS_PIPELINE_PATH],
        default_value=LOGS_PATH
    )
)
class LoggerReport:
    """
    Classe che gestisce i logger statici del programma
    """

    DEFAULT_LOGGING_PATH = "../logs/"
    DATASET_LOGGING_PATH = DEFAULT_LOGGING_PATH + "dataset/"
    TRAINING_LOGGING_PATH = DEFAULT_LOGGING_PATH + "training/"
    PIPELINE_LOGGING_PATH = DEFAULT_LOGGING_PATH + "pipeline/"
    __logging_paths = (DATASET_LOGGING_PATH, TRAINING_LOGGING_PATH, PIPELINE_LOGGING_PATH)

    def __init__(
            self,
            name: str = "unknown",
            start_message: str = "!! START NEW LOG !!",
            path: str = LOGS_PATH
    ):
        self.__name = name
        self.__start_message = start_message
        self.__path = path
        self.__filelog = self.__format_name_filelog()

    def __format_name_filelog(self):
        if self.__path == self.DATASET_LOGGING_PATH:
            name = f"report-dataset_{self.__name}"
        elif self.__path == self.TRAINING_LOGGING_PATH:
            name = f"report-training_{self.__name}"
        else:
            name = f"report-{self.__name}"

        return self.__path + name

    def run(self):
        logging.basicConfig(
            filename=self.__filelog,
            level=logging.INFO,
            format='%(message)s'
        )

        logging.info(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
        logging.info(self.__start_message)
