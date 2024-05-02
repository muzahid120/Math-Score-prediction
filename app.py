from src.logger import logging
import sys
from src.exception import CustomException


if __name__=='__main__':
    logging.info(' code  run  successfully ! ')

    logging.info('you are my  sweetheart ! ')
    logging.info('excuation has staarted')


    try:
        a=1/0
        

    except Exception as E:
        logging.info('here is Custom error happen')
        raise CustomException(E,sys)
