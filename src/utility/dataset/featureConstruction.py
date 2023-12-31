import json
import spacy
import emojis
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from feel_it import EmotionClassifier, SentimentClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


class featureConstruction():

    def __init__(self, dataFrame: pd.DataFrame, datasetPath: str, config="../configs/config.json"):
        self.DATASET_PATH = datasetPath
        self.__dataFrame = dataFrame
        self.__config = config

        self.__init_configs()
        self.__feature_construction()
        self.__write_dataFrame()

    def __init_configs(self):
        """Inizializza variabili in base al file di configurazione."""

        # Leggi file di configurazione
        with open(self.__config, 'r') as f:
            features = json.load(f)

        # Estrai i nomi delle feature con valore vero (cioè feature abilitate)
        self.__features_enabled = [k for k, v in features.items() if v]

        # Inizializza analizzatore NLP
        # Bisogna prima scaricare i moduli (sm -> small, md -> medium, lg -> large):
        #   python3 -m spacy download en_core_web_sm
        #   python3 -m spacy download it_core_news_lg
        # ATTENZIONE: Il vocabolario inglese contiene anche parole italiane e viceversa
        self.__nlp_it = spacy.load("it_core_news_lg")
        self.__nlp_en = spacy.load("en_core_web_sm")

        # Lista tag POS (Part-of-speech)
        self.__POS_LIST = [
            "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
            "SYM", "VERB", "X", "SPACE"
        ]

        # Lista di vocaboli italiani e inglesi in minuscolo
        if "italianness" in self.__features_enabled:
            self.__italian_words = {w.lower() for w in self.__nlp_it.vocab.strings}

        if "englishness" in self.__features_enabled:
            self.__english_words = {w.lower() for w in self.__nlp_en.vocab.strings}

        # Inizializza sentiment/emotion classifier
        if "sentiment" in self.__features_enabled:
            self.__sentiment_classifier = SentimentClassifier()
            self.__sentiment_mapping = {'negative': 0, 'positive': 1}

        if "emotion" in self.__features_enabled:
            self.__emotion_classifier = EmotionClassifier()
            self.__emotion_mapping = {"anger": 0, "fear": 1, "joy": 2, "sadness": 3}

    def __feature_construction(self):
        """Crea nuove feature: un insieme di metadati relativo ad ogni messaggio."""
        # Applica bag of words solo se abilitato
        if "bag_of_words" in self.__features_enabled:
            print("[LOADING] Applicazione tecnica bag of words con HashingVectorizer...")
            self.bag_of_words()
            self.__features_enabled.remove("bag_of_words")

        # Colonne (features) che si aggiungeranno al dataframe
        features = {key: [] for key in self.__features_enabled}

        # LOGGING:: Inserire le feature usate per la predizione
        logging.info(
            "Feature usate: \n" +
            "\n".join(f"\t{feature_name}" for feature_name in self.__features_enabled)
        )

        for message in tqdm(self.__dataFrame['message']):
            # Resetta analisi nlp per nuovo messaggio
            self.__nlp_it_message = None

            for feature_name in self.__features_enabled:
                # Ottieni funzione dal nome
                featureFunction = getattr(self, feature_name)

                # Chiama la funzione e aggiunge alla lista il valore di ritorno
                features[feature_name].append(featureFunction(message))

        # Aggiungi nuove colonne al dataframe
        for feature_name in self.__features_enabled:
            self.__dataFrame[feature_name] = features[feature_name]

        for pos in self.__POS_LIST:
            self.__dataFrame[pos] = [d[pos] for d in features["message_composition"]]

    def __get_nlp_it_message(self, m):
        """Metodo che serve per non ricalcolare nlp_it_message in feature diverse."""
        if self.__nlp_it_message is None:
            self.__nlp_it_message = self.__nlp_it(m)

        return self.__nlp_it_message

    def bag_of_words(self, max_accuracy=False):
        """
            Trasforma il messaggio in una matrice di frequenza delle parole (bag of words).
            In questo modo, il modello capisce le parole più utilizzate da un utente
        """
        # Numero di feature di default (compromesso fra velocità e accuratezza)
        n = 2 ** 10

        # Se si vuole usare il numero di feature ottimale per non avere collisioni
        # ma si vuole sacrificare la velocità di esecuzione
        if max_accuracy:
            # Tokenizza il testo e conta il numero di parole uniche
            count_vec = CountVectorizer()
            count_vec.fit(self.__dataFrame['message'])
            n_unique_words = len(count_vec.vocabulary_)

            # Imposta n_features come la potenza di 2 successiva che è maggiore di n_unique_words
            n = int(2 ** np.ceil(np.log2(n_unique_words)))

        # Inizializza l'HashingVectorizer con il numero di features calcolato
        hashing_vec = HashingVectorizer(n_features=n)
        hashed_text = hashing_vec.fit_transform(self.__dataFrame['message'])

        # Unisci la matrice al dataframe
        df_hashed_text = pd.DataFrame(hashed_text.toarray())
        self.__dataFrame = pd.concat([self.__dataFrame, df_hashed_text], axis=1)

    def __write_dataFrame(self):
        """Salva il dataframe aggiornato in formato parquet."""

        # Rimuovi features inutili in fase di training
        df_to_export = self.__dataFrame.drop(['date', 'message_composition', 'message'], axis=1)

        # Assicurati che le nuove colonne siano stringhe
        df_to_export.columns = df_to_export.columns.astype(str)

        # Esporta file
        df_to_export.to_parquet(self.DATASET_PATH)

    @staticmethod
    def uppercase_count(m):
        """Conta il numero di caratteri in maiuscolo in un messaggio."""
        return sum(1 for c in m if c.isupper())

    @staticmethod
    def char_count(m):
        """Conta il numero di caratteri in un message."""
        return len(m)

    @staticmethod
    def word_length(m):
        """Conta la lunghezza media delle parole in un messaggio."""
        parole = m.split()

        if not parole:
            return 0  # Restituisce 0 se la stringa è vuota

        lunghezza_totale = sum(len(parola) for parola in parole)
        lunghezza_media = lunghezza_totale / len(parole)

        return round(lunghezza_media, 2)

    @staticmethod
    def emoji_count(m):
        """Conta il numero di emoji in un messaggio."""
        return emojis.count(m)

    @staticmethod
    def unique_emoji_count(m):
        """Conta il numero di emoji uniche in un messaggio."""
        return emojis.count(m, unique=True)

    @staticmethod
    def vocabulary_count(nlp_message, vocabulary):
        """
            Indica quanto una persona parla italiano "pulito"
                oppure quanti inglesismi usa (in base al vocabolario in input)

            cioè il rapporto fra parole presenti nel vocabolario
                e il numero totale di parole in un messaggio.
        """
        count = 0
        total_count = 0

        for token in nlp_message:
            if token.is_alpha:  # per filtrare numeri e simboli
                total_count += 1

                if token.text.lower() in vocabulary:
                    count += 1

        # Frase senza parole
        if total_count == 0:
            return 0

        return round(count / total_count, 2)

    def englishness(self, m):
        """
            Indica quanto un utente usa inglesismi all'interno di un messaggio.
        """
        return self.vocabulary_count(self.__nlp_en(m), self.__english_words)

    def italianness(self, m):
        """
            Indica quanto un utente parla italiano "pulito" all'interno di un messaggio.
        """
        return self.vocabulary_count(self.__get_nlp_it_message(m), self.__italian_words)

    def message_composition(self, m):
        """
            Calcola la percentuale di tag POS (Part-of-speech) presenti in un messaggio.
            Es. "Mario mangia la pasta"
                conta il numero di pronomi, verbi, articoli, sostantivi... -> {'PROPN': 1, 'VERB': 1, 'DET': 1, 'NOUN': 1}
                calcola la percentuale di ognuno -> {'VERB': 0.2, 'SCONJ': 0.2, 'PROPN': 0.2, 'PUNCT': 0.2, 'DET': 0.2, 'NOUN': 0.2}

            Lista di tag POS:
            ADJ: Adjective, e.g., big, old, green, incomprehensible, first.
            ADP: Adposition, e.g., in, to, during.
            ADV: Adverb, e.g., very, tomorrow, down, where, there1.
            AUX: Auxiliary verb, e.g., is, has (done), will (do), should (do)1.
            CONJ: Conjunction.
            CCONJ: Coordinating conjunction, e.g., and, or, but.
            DET: Determiner, e.g., a, an, the.
            INTJ: Interjection, e.g., psst, ouch, bravo, hello.
            NOUN: Noun, e.g., girl, cat, tree, air, beauty.
            NUM: Numeral, e.g., 1, 2017, one, seventy-seven, IV, MMXIV1.
            PART: Particle.
            PRON: Pronoun, e.g., I, you, he, she, myself, themselves, somebody1.
            PROPN: Proper noun, e.g., Mary, John, London, NATO, HBO1.
            PUNCT: Punctuation, e.g., ., (, ), ?.
            SCONJ: Subordinating conjunction, e.g., if, while, that.
            SYM: Symbol, e.g., *$, %, §, ©, +, −, ×, ÷, =, :).
            VERB: Verb, e.g., run, runs, running, eat, ate, eating.
            X: Other.
            SPACE: Space.
        """
        # Conta token per tipo
        c = Counter(([token.pos_ for token in self.__get_nlp_it_message(m)]))

        # Numero totale di token
        sbase = sum(c.values())

        # Calcola percentuale di quel tipo di token nel messaggio
        d = defaultdict(int)  # dizionario con valori di default 0
        for el, cnt in c.items():
            d[el] = round(cnt / sbase, 2)

        return d

    def first_word_type(self, m):
        """Restituisce l'id del tag POS della prima parola di un messaggio."""
        self.__get_nlp_it_message(m)

        if self.__nlp_it_message.text == "":
            return -1

        return self.__POS_LIST.index(self.__nlp_it_message[0].pos_)

    def sentiment(self, m):
        """Restituisce l'id del sentiment del messaggio descritto in sentiment_mapping."""
        return self.__sentiment_mapping[self.__sentiment_classifier.predict([m])[0]]

    def emotion(self, m):
        """Restituisce l'id dell'emotion del messaggio descritto in emotion_mapping."""
        return self.__emotion_mapping[self.__emotion_classifier.predict([m])[0]]