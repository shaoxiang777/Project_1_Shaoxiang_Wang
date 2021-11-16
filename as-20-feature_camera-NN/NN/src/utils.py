import json
import logging

class Params:
    """
    Class that loads hyperparameters from a json file

    Example:
    '''
    params = Params(json_path)
    print(params.learning_rate) 
    '''

    """

    def __init__(self, json_path):
        self.name = 'Ka-RaceIng'  # Here automatically generate a self.__dict__
        self.update(json_path)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """ Load parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate'] or params.learning_rate`"""
        return self.__dict__

def set_logger(log_path):
    """Sets the logger to log info in terminal and file 'log_path'.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file.

    Exapmle:
    '''
    logging.info('Start training...')
    '''

    Args:
         log_path: (string) where to log
    """

    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)  # # Here level is configured as INFO information, that is, only INFO and information above its level are output.

    if not logger.handlers:
        # logging to a file
        file_handler = logging.FileHandler(log_path)  
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))  # It can be seen that the content displayed on the console terminal is less, so asctime and levelname are no longer stored
        logger.addHandler(stream_handler)



