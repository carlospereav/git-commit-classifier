import logging
import os

def setup_logging(level=logging.INFO):
    """
    Configures the logging system with a standard format and the specified level.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # logging.getLogger("transformers").setLevel(logging.WARNING)
    # logging.getLogger("datasets").setLevel(logging.WARNING)
