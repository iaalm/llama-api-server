from uuid import uuid4
from time import time


def get_uuid():
    return uuid4()


def get_timestamp():
    return int(time())
