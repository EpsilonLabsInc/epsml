import logging


def configure_logger(logger_file_name, show_logging_level=True):
    logging.basicConfig(
        filename=logger_file_name,
        filemode="w",
        format="%(levelname)s: %(message)s" if show_logging_level else "%(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        level=logging.INFO)
