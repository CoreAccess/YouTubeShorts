import logging

class Logger:
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger('youtube_shorts')
        self.logger.setLevel(logging.DEBUG)

        # Truncate the log file at startup
        with open('app.log', 'w'):
            pass

        # Use standard FileHandler instead of ConcurrentRotatingFileHandler
        self.handler = logging.FileHandler(
            'app.log',
            mode='a',
            encoding='utf-8'
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