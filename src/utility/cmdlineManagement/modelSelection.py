import logging
#from simple_term_menu import TerminalMenu
import inquirer
from ..model.model_list import models


class ModelSelection:
    MODEL_PATH = "../models/"
    model_names = [name for name in models.keys()]

    def __init__(self):
        self.__model = self.__select_model()

    @classmethod
    def __load_model(cls, model_name):
        if model_name not in models.keys():
            raise ValueError("Modello non valido.")

        selected_model_name = models[model_name]

        return selected_model_name

    @classmethod
    def __select_model(cls):
        # print("Elenco dei modelli disponibili:")
        # menu = TerminalMenu(cls.model_names)
        # menu_entry_index = menu.show()

        # model_name = cls.model_names[menu_entry_index]

        model_selection = [
            inquirer.List('model',
                message="Elenco dei modelli disponibili:",
                choices= cls.model_names
            ),
        ]

        model = inquirer.prompt(model_selection)

        model_name = model["model"]

        model_selected = cls.__load_model(model_name)

        print(f"Modello selezionato: {model_name}")

        # LOGGING:: Stampa il modello selezionato
        logging.info(f"Modello usato per il training: {model_name}")

        return model_selected

    @property
    def model(self):
        return self.__model

