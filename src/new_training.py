import time
import calendar
import argparse
from utility.logging import LoggerReport, LoggerUserModelHistory
from utility.model.modelTraining import ModelTraining
from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.modelSelection import ModelSelection

# HOW TO USE:
# py new_training.py -oN <*outputName> -c <*configFile> -r <*retrain>

# CHECKS IF SPECIFIED DATASET EXIST
# (dataCreation.py return already existing DF)

# ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from new_dataset.py

# ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT


if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    timestamp = "model_" + str(calendar.timegm(time.gmtime()))

    # Optional arguments
    parser.add_argument("-oN", "--outputName", help="Nome file del modello salvato", required=False, default=timestamp)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)

    args = parser.parse_args()
    output_name, config, retrain = args.outputName, args.config, args.retrain

    # LOGGING:: Inizializza il logging
    LoggerReport(
        name=output_name,
        start_message="!! NEW TRAINING !!",
        path=LoggerReport.TRAINING_LOGGING_PATH
    ).run()

    # Select dataset
    dataset_selection = DatasetSelection()
    dataset = dataset_selection.dataset
    dataset_name = dataset_selection.dataset_name

    # Select model
    model = ModelSelection().model

    LoggerUserModelHistory.append_model_user(dataset_name, output_name)

    # Training del modello con i parametri passati da linea di comando
    model_training = ModelTraining(output_name, model, dataset, config, retrain)
    model_training.run()
