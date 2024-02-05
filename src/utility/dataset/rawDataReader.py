import os
import logging
from tqdm import tqdm
from ..config_path import RAWDATA_PATH


class RawDataReader:

    # Secondo me non serve il costruttore.
    # Alla fine i file da leggere sono sempre all'interno della cartella RAWDATA_PATH.
    # Se si vuole cambiare la cartella basta cambiare il valore di RAWDATA_PATH.

    # P.S Se si cambia, si può rendere la classe un singleton oppure una classe statica.
    def __init__(self, data_path=RAWDATA_PATH):
        self.DATA_PATH = data_path

    def read_all_files(self):
        """Restituisce una lista di stringhe contenente il contenuto di tutti i file nella cartella DATA_PATH."""
        rawdata = []

        # Ottieni lista dei file nella cartella
        files = [f for f in os.listdir(self.DATA_PATH) if os.path.isfile(os.path.join(self.DATA_PATH, f))]

        if len(files) < 1:
            raise Exception("Nessun file di testo trovato per l'estrazione dati.")

        # Concatena contenuto di ogni file nella cartella
        for file_name in tqdm(files):
            f = open(os.path.join(self.DATA_PATH, file_name), 'r', encoding='utf-8')
            rawdata.append(f.read())
            f.close()

        # LOGGING:: Stampa i file utilizzati per l'estrazione dati
        logging.info("File grezzi usati: \n" + "\n".join(f"\t{file}" for file in files))

        return rawdata
