import logging

import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_nltk():
    """
    NLTK data organization:
    - tokenizers/punkt: For sentence tokenization
    - corpora/stopwords: For stopwords lists
    """
    packages = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }

    for package, path in packages.items():
        try:
            nltk.data.find(path)
            logger.info(f"Found NLTK package: {path}")
        except LookupError:
            logger.info(f"Downloading NLTK package: {package}")
            nltk.download(package)


def main():
    logger.info("Initializing data...")
    initialize_nltk()
    logger.info("Data initialization completed")


if __name__ == "__main__":
    main()
