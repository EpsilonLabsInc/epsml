import logging


def configure_logger(logger_file_name, logging_level=logging.INFO, show_logging_level=True, append_to_existing_log_file=False):
    logging.basicConfig(
        filename=logger_file_name,
        filemode="a" if append_to_existing_log_file else "w",
        format="%(levelname)s: %(message)s" if show_logging_level else "%(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        level=logging_level)
