import functools
import json

from fastapi import File, UploadFile
from loguru import logger
from pydantic import BaseModel


class ValueArgs(BaseModel):
    value: object
    args: dict


class ExtractValueArgsFastapi:
    def __init__(self, file: bool = False):
        self.file = file

    def __call__(self, f):
        logger.debug(f"file {self.file}")
        # @functools.wraps(f) # tolto perchè ci obbligava a tipizzare dentro fastAPI
        def extract_file_args(file: UploadFile = File(), args: bytes = File()):
            args = json.loads(args.decode())
            return f(file, args)
        # extract_file_args.__name__ = f.__name__

        def extract_value_args(value_args: ValueArgs):
            logger.debug(f"value::: {value_args}")
            value = value_args.value
            args = value_args.args
            return f(value=value, args=args)
        # extract_value_args.__name__ = f.__name__

        if self.file:
            logger.debug('File')
            return extract_file_args
        else:
            return extract_value_args
