import re

import pymorphy3
from nltk.corpus import stopwords


class TextPreprocessor:
    """
    A utility class for text preprocessing, including tasks such as normalization,
    punctuation cleaning, and stopword removal. This is particularly useful for
    preparing text data for NLP tasks.

    Attributes:
        morph: A morphological analyzer for Russian text.
        stopwords_set: A set of Russian stopwords with additional manually added words.
        morph_exceptions: Words that are excluded from morphological normalization.
        manual_corrections: A dictionary for correcting specific token errors.
        phone_pattern: A regex pattern for identifying phone numbers.
        name_pattern: A regex pattern for identifying full names.
        sim_card_pattern: A regex pattern for identifying SIM card references.
    """

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stopwords_set = set(stopwords.words("russian"))
        self.stopwords_set.update(["который", "всё", "добрый", "время", "сутки", "--", "00мск"])
        self.morph_exceptions = {"сим"}
        self.manual_corrections = {
            "удали": "удалить",
            "использьзовать": "использовать",
            "теле": "теле2",
        }
        self.phone_pattern = re.compile(r"^((8|\+7)[\- ]?)?(\(?\d{3}\)?[\- ]?)?[\d\- ]{7,10}$")
        self.name_pattern = re.compile(r"\b[А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+\b")
        self.sim_card_pattern = re.compile(r"\bсим-карт\w*\b")
        self.w2v_model = None

    def clean_key(self, key):
        """
        Cleans and normalizes a given key by replacing phone numbers and names,
        removing standalone digits, and normalizing punctuation.

        Args:
            key (str): The input string to clean.

        Returns:
            str: The cleaned and normalized string, or an empty string if the input is not a string.
        """
        if isinstance(key, str):
            key = self.phone_pattern.sub("номер телефона", key)
            key = self.name_pattern.sub("фио", key)
            key = re.sub(r"(?<![\w])\d+(?![\w])", "", key)
            key = re.sub(r"[!?]", ".", key)
            key = re.sub(r"\s-\s", " ", key)
            key = re.sub(r"(?<!\S)-|-(?!\S)", "", key)
            return key
        return ""

    def split_key(self, key_mod):
        """
        Splits a normalized string into tokens while preserving punctuation as separate tokens.

        Args:
            key_mod (str): The cleaned and normalized string to split.

        Returns:
            list: A list of tokens derived from the input string.
        """
        result = []
        temp_word = []

        for char in key_mod:
            if char.isalpha() or char.isdigit() or char == "-":
                temp_word.append(char)
            elif char == "." or char.isspace():
                if temp_word:
                    result.append("".join(temp_word))
                    temp_word = []
                if char == ".":
                    result.append(".")

        if temp_word:
            result.append("".join(temp_word))

        return result

    def clean_punctuation(self, list_elements):
        """
        Removes punctuation tokens from a list of elements.

        Args:
            list_elements (list): A list of tokens to clean.

        Returns:
            list: A list of tokens with punctuation removed.
        """
        return [element for element in list_elements if element != "."]

    def preprocess_text(self, text):
        """
        Processes input text by cleaning, splitting, normalizing, and filtering tokens.

        Args:
            text (str): The input text to preprocess.

        Returns:
            list: A list of preprocessed tokens.
        """
        cleaned_list = []
        key_mod = self.clean_key(text)
        words = self.split_key(key_mod.lower())

        for raw_token in words:
            token = raw_token
            if self.sim_card_pattern.search(token):
                token = self.sim_card_pattern.sub("сим-карта", token)

            token = self.manual_corrections.get(token, token)

            if token not in self.morph_exceptions:
                token = self.morph.parse(token)[0].normal_form

            if token not in self.stopwords_set and token != ".":
                cleaned_list.append(token)

        return cleaned_list
