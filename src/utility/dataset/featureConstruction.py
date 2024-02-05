import re
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
from ..clean_coding.decorator import check_path_exists
from ..clean_coding.ensure import validation, ensure_not_none, ensure_valid_file_extension, ensure_file_exists
from ..config_path import CONFIG_PATH, DATASET_PATH, WORDLIST_PATH


@check_path_exists(path=CONFIG_PATH, create=True)
@validation(
    "dataset_name",
    "Nome del dataset",
    ensure_not_none())
@validation(
    "config",
    "File di configurazione",
    ensure_valid_file_extension(".json"), ensure_file_exists(CONFIG_PATH, ""))
class FeatureConstruction:
    # Lista tag POS (Part-of-speech)
    POS_LIST = [
        "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
        "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
        "SYM", "VERB", "X", "SPACE"
    ]

    def __init__(self, data_frame: pd.DataFrame, dataset_name: str, config_path="config.json", save_dataFrame:bool = True):
        self.__dataFrame = data_frame
        self._dataset_name = dataset_name
        self._config = config_path if config_path is not None else "config.json"
        self._columns_to_drop = ['message_composition', 'message']

        self.__init_configs()
        self.__feature_construction()
        
        if save_dataFrame:
            self.__write_dataFrame()

    def get_dataframe(self):
        return self.__dataFrame

    def __init_configs(self):
        """Inizializza variabili in base al file di configurazione."""

        # Leggi file di configurazione
        with open(CONFIG_PATH + self._config, 'r') as f:
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

        # Lista di vocaboli italiani e inglesi in minuscolo
        if "italianness" in self.__features_enabled:
            self.__italian_words = {w.lower() for w in self.__nlp_it.vocab.strings}

        if "englishness" in self.__features_enabled:
            self.__english_words = {w.lower() for w in self.__nlp_en.vocab.strings}

        # Car   ica wordlist
        if "swear_words" in self.__features_enabled:
            with open(WORDLIST_PATH + "swear_words.txt", 'r') as f:
                self.__swear_words = set(f.read().splitlines())

                # rimuovi l'ultimo carattere di ogni parola (per parole in dialetto)
                self.__swear_words.update([word[:-1] for word in self.__swear_words])

        if "common_words" in self.__features_enabled:
            with open(WORDLIST_PATH + "common_words.txt", 'r') as f:
                self.__common_words = set(f.read().splitlines())

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

        if "responsiveness" in self.__features_enabled:
            # Rimuovi dalla lista perché è già calcolata in fase di training
            self.__features_enabled.remove("responsiveness")
        else:
            # Rimuovi dal dataframe se non è abilitata
            # (perché non ha senso per il messaggio in input in demo.py)
            self._columns_to_drop.append("responsiveness")

        # Colonne (features) che si aggiungeranno al dataframe
        features = {key: [] for key in self.__features_enabled}

        # LOGGING:: Inserire le feature usate per la predizione
        logging.info(
            "Feature selezionate in Dataset Creation: \n" +
            "\n".join(f"\t{feature_name}" for feature_name in self.__features_enabled)
        )

        for message in tqdm(self._dataFrame['message']):
            # Resetta analisi nlp per nuovo messaggio
            self.__nlp_it_message = None

            for feature_name in self.__features_enabled:
                # Ottieni funzione dal nome
                featureFunction = getattr(self, feature_name)

                # Chiama la funzione e aggiunge alla lista il valore di ritorno
                features[feature_name].append(featureFunction(message))

        # Aggiungi nuove colonne al dataframe
        for feature_name in self.__features_enabled:
            self._dataFrame[feature_name] = features[feature_name]

        if "message_composition" in self.__features_enabled:
            for pos in self.POS_LIST:
                self._dataFrame[pos] = [d[pos] for d in features["message_composition"]]

        # Rimuovi features inutili in fase di training
        # errors='ignore' per evitare errori se la colonna non esiste
        self.__dataFrame = self._dataFrame.drop(self._columns_to_drop, axis=1, errors='ignore')

        # Assicurati che il nome delle colonne siano stringhe
        # (altrimenti ci sono problemi in fase di esportazione del file parquet)
        self.__dataFrame.columns = self.__dataFrame.columns.astype(str)

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
        n = 2 ** 12

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
        self._dataFrame = pd.concat([self.__dataFrame, df_hashed_text], axis=1)

    def __write_dataFrame(self):
        """Salva il dataframe aggiornato in formato parquet."""
        self.__dataFrame.to_parquet(DATASET_PATH + self._dataset_name)

    def type_token_ratio(self, m):
        """Calcola la ricchezza lessicale di un messaggio."""
        message_nlp = self.__get_nlp_it_message(m)
        lemmas = [token.lemma_ for token in message_nlp if token.is_alpha]
        if len(lemmas) == 0:
            return 0
        return len(set(lemmas)) / len(lemmas)

    def simpsons_index(self, m):
        """Calcola l'indice di Simpson di un messaggio."""
        message_nlp = self.__get_nlp_it_message(m)
        lemmas = [token.lemma_ for token in message_nlp if token.is_alpha]

        lemma_counts = Counter(lemmas)
        try:
            N = len(lemmas)
            D = sum(n * (n - 1) for n in lemma_counts.values())
            simpson_index = D / (N * (N - 1))

            return simpson_index
        except ZeroDivisionError:
            return 0

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

        return self.POS_LIST.index(self.__nlp_it_message[0].pos_)

    def sentiment(self, m):
        """Restituisce l'id del sentiment del messaggio descritto in sentiment_mapping."""
        return self.__sentiment_mapping[self.__sentiment_classifier.predict([m])[0]]

    def emotion(self, m):
        """Restituisce l'id dell'emotion del messaggio descritto in emotion_mapping."""
        return self.__emotion_mapping[self.__emotion_classifier.predict([m])[0]]
    
    def swear_words(self, m):
        """Conta il numero di parolacce in un messaggio."""
        matches = re.compile(r"\b(" + "|".join(self.__swear_words) + r")\b", re.IGNORECASE).findall(m)
        return len(matches)
    
    def common_words(self, m):
        """Conta il numero di parole comuni in un messaggio."""
        lemmas = [token.lemma_ for token in self.__get_nlp_it_message(m)]
        
        count = 0
        for lemma in lemmas:
            if lemma in self.__common_words:
                count += 1
 
        if (len(lemmas) == 0):
            return 0
        
        return count/len(lemmas)
    
    def readability(self, m):
        """
            Gulpease index: Questa metrica calcola la facilità di lettura di un testo su una scala da 100 punti. 
 
            I risultati sono compresi tra 0 (leggibilità più bassa) e 100 (leggibilità più alta)

            inferiore a 80 sono difficili da leggere per chi ha la licenza elementare
            inferiore a 60 sono difficili da leggere per chi ha la licenza media
            inferiore a 40 sono difficili da leggere per chi ha un diploma superiore

            https://it.wikipedia.org/wiki/Indice_Gulpease
        """

        nlp = self.__get_nlp_it_message(m)

        words = [token.text for token in nlp if not token.is_punct]
        num_words = len(words)
        num_letters = sum(len(word) for word in words)
        num_sentences = len(list(nlp.sents))

        if num_words == 0 or num_sentences == 0:
            return 100

        # Formula per calcolare l'indice di Gulpease
        score = 89 + (300 * num_sentences - 10 * num_letters) / num_words

        # Normalizza outliers (Se non si normalizza esce come predittivo)
        # if score > 100:
        #     score = 100
        # elif score < 0:
        #     score = 0

        return score

    def functional_words(self, m):
        """
            Conta il numero di parole funzionali (stop words) in un messaggio.
            
            Le parole funzionali sono parole che hanno poco significato lessicale o hanno un significato ambiguo, 
                ma servono invece a esprimere relazioni grammaticali con altre parole all'interno di una frase o
                a specificare l'atteggiamento o l'umore del parlante
            
            L'idea è che queste parole siano meno influenzate dall'argomento e più riflettano lo stile personale di un autore.
        """
        words = [token for token in self.__get_nlp_it_message(m) if not token.is_punct]
        # functional_words = len([token.text.lower() for token in self.__get_nlp_it_message(m) if token.is_stop])
        
        # Define a list of functional word types (POS tags)
        functional_pos_tags = ['ADP', 'AUX', 'CONJ', 'DET', 'PART', 'PRON', 'SCONJ']

        # Count the functional words
        functional_words = sum(token.pos_ in functional_pos_tags for token in self.__get_nlp_it_message(m))

        if len(words) == 0:
            return 0

        return functional_words/len(words)