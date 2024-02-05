import os
import logging
import pandas as pd
import inquirer
from ..exceptions import ExtensionError
from ..model.modelTraining import ModelTraining
from joblib import load


class TrainedModelSelection:
    MODEL_PATH = "../models/"

    def __init__(self):
        self.__model = self.__select_model()

    @classmethod
    def __show_models(cls):
        # Seleziona solo i file
        all_file = [file for file in os.listdir(cls.MODEL_PATH) if os.path.isfile(os.path.join(cls.MODEL_PATH, file))]

        # Seleziona solo i modelli
        models = [model for model in all_file if model.endswith(".joblib") or model.endswith(".onnx")]

        # Ordina i modelli in base alla data di creazione
        models = sorted(models, key=lambda x: os.path.getctime(os.path.join(cls.MODEL_PATH, x)), reverse=True)
        
        return models

    @classmethod
    def __load_model(cls, index, models):
        if 0 <= index < len(models):
            selected_model = models[index]

            if not selected_model.endswith(".joblib"):
                raise ExtensionError("Il modello deve essere in formato .joblib")

            path = f"{cls.MODEL_PATH}{selected_model}"

            # Carica il model
            return selected_model, load(path)
        else:
            raise ValueError("ID del modello non valido.")

    @classmethod
    def __select_model(cls):
        models = cls.__show_models()

        model_selection = [
            inquirer.List('model',
                message="Seleziona il modello da usare",
                choices=models
            ),
        ]

        model = inquirer.prompt(model_selection)

        # Ottieni model
        model_name = model["model"]
        menu_entry_index = models.index(model_name)

        print(f"Modello selezionato: {model_name}")
        # LOGGING:: Stampa il modello (e lo scaler) selezionato
        logging.info(f"Modello usato: {model_name}")

        return cls.__load_model(menu_entry_index, models)

    @property
    def model(self):
        return self.__model