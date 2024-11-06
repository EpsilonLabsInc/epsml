import logging


def configure_logger(logger_file_name):
    logging.basicConfig(
        filename=logger_file_name,
        filemode="w",
        format="%(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        level=logging.INFO)
