import json
import time
import logging
import calendar
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from ..dataset.featureConstruction import FeatureConstruction
from sklearn.model_selection import train_test_split, cross_validate
from ..exceptions import DatasetNotFoundError, ModelNotFoundError
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from ..clean_coding.ensure import *
from ..clean_coding.decorator import check_path_exists
from ..config_path import MODELS_PATH, CONFIG_PATH


@check_path_exists(path="../models/", create=True)
@validation(
    "output_name",
    "Nome del modello da salvare",
    ensure_not_none("model_" + str(calendar.timegm(time.gmtime()))), ensure_valid_file_extension(".joblib")
)
@validation(
    "config_file",
    "File di configurazione",
    ensure_not_none("config.json"), ensure_valid_file_extension(".json"), ensure_file_exists("../configs/", "")
)
@validation(
    "model",
    "Modello da addestrare",
    ensure_not_none(exception_type=ModelNotFoundError)
)
@validation(
    "data_frame",
    "Dataset da usare per l'addestramento",
    ensure_not_none(exception_type=DatasetNotFoundError)
)
@validation(
    "retrain",
    "Opzione di retraining",
    ensure_not_none(False), ensure_valid_type(bool) 
)
class ModelTraining:
    __YES_CHOICES = ["yes", "y", ""]
    __NO_CHOICES = ["no", "n"]

    def __init__(
            self, 
            output_name: str = None, 
            model=None, 
            data_frame: pd.DataFrame = None,
            config_file: str = None,
            retrain: bool = None
    ):
        self._output_name = output_name
        self._model = model
        self._data_frame = data_frame
        self._config_file = config_file
        self._retrain = retrain

    def run(self):
        """Avvia il training del modello."""

        self.__init_configs()

        self.__check_duplicates()

        print("[INFO] Training del modello in corso...")
        return self._model_training()

    def __init_configs(self):
        """Inizializza i parametri di configurazione."""
        # Leggi file di configurazione
        with open(CONFIG_PATH + self._config_file, 'r') as f:
            features = json.load(f)

        # Estrai i nomi delle feature con valore falso (cioè le feature da filtrare)
        self.__features_disabled = [k for k, v in features.items() if not v]

        # LOGGING: Stampa le feature usate in fase di training
        logging.info(
            f"Feature usate in fase di training: \n" +
            "\n".join(f"\t {k}" for k, v in features.items() if v)
        )

        if "message_composition" in self.__features_disabled:
            self.__features_disabled.remove("message_composition")
            self.__features_disabled.extend(FeatureConstruction.POS_LIST)

    def __check_duplicates(self):
        """Controlla se il modello esiste già e se si vuole sovrascriverlo."""

        # Controllo in caso si voglia sovrascrivere comunque
        while os.path.exists(MODELS_PATH + self._output_name) and not self._retrain:

            user_input = input("Il modello '{}' già esiste, sovrascriverlo? [Y/N]\n".format(self._output_name))
            user_input = user_input.lower()

            if user_input in self.__YES_CHOICES:
                break
            elif user_input in self.__NO_CHOICES:
                print("Operazione di Training annullata")
                return 1
            else:
                print('Inserire \"Y\" o \"N\"')
                continue

    def _model_training(self, kfold: int = 0):
        """Applica random forest sul dataframe."""

        # LOGGING:: Stampa il nome del modello trainato
        logging.info(f"Modello trainato: {self._output_name}")

        # Definisci le features (X) e il target (Y) cioè la variabile da prevedere
        X = self._data_frame.drop(['user'], axis=1)
        y = self._data_frame["user"]

        # Rimuovi le feature da filtrare (specificate nel file di configurazione)
        if self.__features_disabled:
            # Droppa tutte le colonne generate da bag of words (cioè con nomi numerici)
            if "bag_of_words" in self.__features_disabled:
                self.__features_disabled.extend([col for col in X.columns if col.isdigit()])

            X = X.drop(self.__features_disabled, axis=1, errors='ignore')

        # Applica feature scaling
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # TRAINING CON CROSS VALIDATION
        # (numero di fold di solito 5 o 10)
        if kfold > 0:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            scores = cross_validate(self._model, X, y, cv=kfold, scoring=scoring)

            # Stampa un report sulle metriche di valutazione del modello
            print(f"[INFO] Media delle metriche di valutazione dopo {kfold}-fold cross validation:")
            # LOGGING:: Stampa un report sulle metriche di valutazione del modello
            logging.info(f"Media delle metriche di valutazione dopo {kfold}-fold cross validation:")

            indexes = list(scores.keys())

            for index in indexes:
                # LOGGING:: Stampa le metriche di valutazione del modello
                logging.info(f"\t{index}: %0.2f (+/- %0.2f)" % (scores[index].mean(), scores[index].std() * 2))
                print(f"\t{index}: %0.2f (+/- %0.2f)" % (scores[index].mean(), scores[index].std() * 2))

        # TRAINING CON SPLIT CLASSICO
        test_size = 0.2  # percentuale del dataset di test dopo lo split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self._model.fit(X_train, y_train)
        predictions = self._model.predict(X_test)

        # Genera un report per il modello addestrato
        print(f"\n[INFO] Report con {int((1 - test_size) * 100)}% training set e {int(test_size * 100)}% test set:")

        # Calcola l'accuratezza del modello
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {round(accuracy, 2)}\n')
        # LOGGING:: Stampa l'accuratezza del modello
        logging.info(f"Accuracy: {round(accuracy, 2)}\n")

        # Stampa il classification report
        report = classification_report(y_test, predictions)
        print(report)
        # LOGGING:: Stampa il classification report
        logging.info(report)

        # Stampa le feature più predittive
        n = 20  # numero di feature per cmdlin
        m = 100 # numero di feature per logging

        feature_names = X.columns.tolist()  # Estrai i nomi di tutte le feature

        try:
            importances = self._model.feature_importances_
            important_features = np.argsort(importances)[::-1]
            top_n_features_cmdline = important_features[:n]
            top_n_features_logging = important_features[:m]

            print("\n[INFO] Top {} feature più predittive:".format(n))

            # LOGGING:: Didascalia per le feature più predittive
            logging.info(f"Top {m} feature più predittive:")

            for i in top_n_features_cmdline:
                print(f"{feature_names[i]}: %0.5f" % importances[i])

            for ranking_index, i in enumerate(top_n_features_logging):
                # LOGGING:: Stampa le feature più predittive
                logging.info(f"\t {ranking_index+1}) {feature_names[i]}: %0.5f" % importances[i])

        except Exception:
            print("Il modello non verifica importanza delle features")


        # Crea una pipeline con lo scaler e il classificatore
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', self._model)
        ])

        # Salva la pipeline (scaler e modello)
        dump(pipeline, MODELS_PATH + self._output_name)

        # CREA CONFUSION MATRIX E ROC CURVE
        cm = confusion_matrix(y_test, predictions, labels=self._model.classes_)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._model.classes_)
        disp.plot()
        plt.show()

        # Convert labels to binary format
        y_test_bin = label_binarize(y_test, classes=self._model.classes_)

        # Compute probabilities for each class
        y_score = self._model.predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(self._model.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        for i in range(len(self._model.classes_)):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        return accuracy