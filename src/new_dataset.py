import time
import calendar
import argparse
from utility.logging import LoggerReport, LoggerUser
from utility.dataset.datasetCreation import DatasetCreation
from utility.cmdlineManagement.PlaceholderUserManager import PlaceholderUserManager


# HOW TO USE
# py new_dataset.py -dN <*datasetName> -c <*configFile> -a <*aliases> -r
#   if datasetName exists
#       if refactor is specified then create dataset with said name
#       else return already made dataset
#   else create dataset based on rawdata with that name

# [W I P] you can use config.json to choose which function to run...


if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    timestamp = "dataset_" + str(calendar.timegm(time.gmtime()))

    # Optional arguments
    parser.add_argument("-dN", "--datasetName", help="Nome dataset", required=False, default=timestamp)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-a", "--aliases", help="File per gli alias in chat", required=False)
    parser.add_argument("-r", "--refactor", help="Opzione di refactor", action="store_true", required=False)

    args = parser.parse_args()
    dataset_name, config, aliases_file, refactor = args.datasetName, args.config, args.aliases, args.refactor

    # Selezione opzioni per l'utente "other"
    placeholder_user, remove_generic = PlaceholderUserManager(aliases_file).selection()

    # LOGGING:: Inizializza il logging
    LoggerReport(
        name=dataset_name,
        path=LoggerReport.DATASET_LOGGING_PATH,
        start_message="!! NEW DATASET CREATION !!"
    ).run()

    LoggerUser.open(dataset_name)

    # Creazione del dataset con i parametri passati da linea di comando
    DatasetCreation(dataset_name, config, aliases_file, placeholder_user, remove_generic, refactor).run()

    LoggerUser.close()
