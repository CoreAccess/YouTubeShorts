import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler

class Logger:
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger('youtube_shorts')
        self.logger.setLevel(logging.DEBUG)

        # Truncate the log file at startup
        with open('app.log', 'w'):
            pass

        self.handler = ConcurrentRotatingFileHandler(
            'app.log',
            maxBytes=10240,
            backupCount=1,
            use_gzip=False
        )
        # Include level name in format for better debugging
        self.formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

        # Prevent Flask's default logger from propagating to the root logger
        app.logger.propagate = False

    def get_logger(self):
        return self.logger