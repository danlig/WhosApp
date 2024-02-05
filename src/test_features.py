from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.modelSelection import ModelSelection
from utility.model.modelTraining import ModelTraining
import json
from utility.logging import LoggerReport as Logger
import time
import calendar


def feature_testing():
    """
        Testa la potenza predittiva di ogni feature: 
            per ogni feature, testa il modello con solo e senza la feature.
    """
    # Select dataset
    dataset_sel = DatasetSelection()
    dataset_name = dataset_sel.dataset_name
    dataset = dataset_sel.dataset

    # Select model
    model = ModelSelection().model

    # Carica il file di configurazione
    config_name = "config.json"
    config_path = ModelTraining.CONFIG_PATH + config_name

    with open(config_path, "r") as f:
        config = json.load(f)

    def train(feature, disable_feature: bool = True):
        # Imposta la configurazione di test
        test_config = {key: disable_feature for key in config}
        if feature != "all": test_config[feature] = not(disable_feature)

        # Determina il nome del file di output
        output_file_name = dataset_name + "_"
        if disable_feature:
            output_file_name += "no_"
        output_file_name += feature

        print(f"\n\nTesting della feature: {output_file_name}...")

        # Sovrascrivi il file di configurazione con la nuova configurazione
        with open(config_path, "w") as f:
            json.dump(test_config, f, indent=4)

        # Training del modello
        return round(ModelTraining(output_file_name, model, dataset, config_name, False).run(), 6)
    
    # Determina accuracy con tutte le feature
    base_accuracy = train("all")
    
    # Determina accuracy con solo e senza una feature
    feature_results = {key: [] for key in config}
    for feature in config:
        feature_results[feature] = (train(feature, False), round(base_accuracy - train(feature, True), 5))

    # Ordina le feature secondo l'accuracy "con solo" in ordine decrescente
    feature_results = sorted(feature_results.items(), key=lambda x: x[1][0], reverse=True)

    # Stampa classifica feature più predittive
    print("\n\nRiepilogo della potenza predittiva delle feature")
    print("Formato: 'feature X': (<test solo con X>, <test tutte feature - test senza X>)")
   
    for i, feature in enumerate(feature_results):
        print(f"{i+1}) {feature[0]}: ({feature[1][0]}, {feature[1][1]})")  


if __name__ == "__main__":

    # Avvia il logger
    Logger(
        name=str(calendar.timegm(time.gmtime())),
        start_message="!! NEW TRAINING !!",
        path=Logger.TRAINING_LOGGING_PATH
    ).run()
        

    feature_testing()