import re
import logging
import pandas as pd
from sklearn.utils import resample
from ..logging import LoggerUser


class DataFrameProcessor:

    BLACKLIST = "./utility/dataset/blacklist.txt"

    def __init__(self, dates=None, users=None, messages=None, other_user: str = None, remove_other: bool = False):
        self.__dates = dates
        self.__users = users
        self.__unique_users = sorted(set(users))
        self.__messages = messages
        self.__other_user = other_user
        self.__remove_other = remove_other

    def __responsiveness(self):
        """
        Calcola responsiveness (differenza fra il timestamp di un messaggio e il precedente).
        Quantifica quanto una persona ci mette a rispondere e anche quanti messaggi "di fila" scrive.
        """
        if self.__dates is not None:
            responsiveness = [max(self.__dates[i] - self.__dates[i - 1], 0) for i in range(1, len(self.__dates))]
            responsiveness.insert(0, 0)
            return responsiveness
        return [0]

    def __cleaning_info_record(self, df: pd.DataFrame):
        """
        Elimina messaggi "informativi"
        """

        if "info" in self.__unique_users:
            self.__unique_users.remove("info")
            return df.loc[df['user'] != "info"]
        return df

    def __indexing_users(self, df: pd.DataFrame):
        """
        Rimpiazza gli utenti con i propri indici per favorire l'elaborazione dei dati
        """

        df['user'].replace(self.__unique_users, range(len(self.__unique_users)), inplace=True)

    def __print_users(self):
        """
        Stampa gli utenti con i propri indici
        """

        print("[INFO] Utenti:", end=" ")
        print(", ".join(f"{i}:{u}" for i, u in enumerate(self.__unique_users)))
        
        # LOGGING:: Stampa gli utenti trovati
        logging.info(
            f"Utenti trovati: \n" +
            "\n".join(f"\t{user} -> {i}" for i, user in enumerate(self.__unique_users))
        )
        LoggerUser.write_user(self.__unique_users, range(len(self.__unique_users)))

    def __print_instances_count(self, df: pd.DataFrame, message=None):
        """
        Stampa il numero di istanze per utente.
        """

        print(f"[INFO] {message}:", end=" ")
        print(", ".join(
            f"{i}: {len(df[df['user'] == i])}" for i in range(len(self.__unique_users))
        ))

        # LOGGING:: Stampa il numero di istanze per utente
        logging.info(f"{message}: \n" + "\n".join(f"\t{i}: {len(df[df['user'] == i])}" for i in range(len(self.__unique_users))))

    @classmethod
    def __cleaning_blacklist(cls, df: pd.DataFrame):
        """
           Rimuovere le righe con altri messaggi informativi
           """

        # Leggi le regex dal file "blacklist.txt"
        with open(cls.BLACKLIST, 'r') as file:
            blacklist_patterns = [re.compile(line.strip()) for line in file]

        # Funzione per controllare se un messaggio matcha una regex nella blacklist
        def matches_blacklist(message: str):
            for pattern in blacklist_patterns:
                if pattern.fullmatch(message):
                    # Il carattere "‎" è presente nei messaggi informativi di IOS
                    return True
            return False

        # Rimuovere le stringhe "<This message was edited>" o "<Questo messaggio è stato modificato>"
        df['message'] = (df['message'].str
                         .replace(r"<This message was edited>|<Questo messaggio è stato modificato>", "", regex=True))

        # Applica la funzione matches_blacklist a ogni riga del DataFrame
        df = df[~df['message'].apply(matches_blacklist)]

        # Rimuove anche i messaggi che contengono "(file attached)"
        df = df[~df['message'].str.contains("\\(file attached\\)")]

        return df

    def __undersampling(self, df: pd.DataFrame):
        """
        Rimuove le righe in eccesso (casualmente) per bilanciare il dataset
        """

        user_class_list = [df[df['user'] == i] for i in range(len(self.__unique_users))]

        min_class = min([len(c) for c in user_class_list])

        user_class_list_downsampled = []
        for c in user_class_list:
            user_class_list_downsampled.append(resample(c, replace=False, n_samples=min_class, random_state=42))

        return pd.concat(user_class_list_downsampled)

    def __cleaning_remove_other(self, df: pd.DataFrame):
        """
        Rimuove i messaggi dell'utente "other"
        """

        logging.info(f"Rimozione degli utenti non prensenti nel alias file")
        # LOGGING:: Stampa la rimozione degli utenti non presenti in aliases
        print(f"[INFO] Rimozione degli utenti non prensenti nel alias file")
        return df.loc[df['user'] != self.__unique_users.index(self.__other_user)]

    def run(self) -> pd.DataFrame:
        """
        Crea il dataframe e effettua le operazioni di pulizia e bilanciamento
        """
        
        df = pd.DataFrame({
            "responsiveness": self.__responsiveness(),
            "user": self.__users,
            "message": self.__messages
        })

        df = self.__cleaning_info_record(df)
        self.__indexing_users(df)

        self.__print_instances_count(df, "Numero di istanze per utente")

        df = self.__cleaning_blacklist(df)
        df = self.__undersampling(df)

        if self.__remove_other:
            df = self.__cleaning_remove_other(df)

        self.__print_users()
        self.__print_instances_count(df, "Numero di istanze per utente dopo cleaning eundersampling")
        return df.reset_index(drop=True)
