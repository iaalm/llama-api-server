from uuid import uuid4
from time import time
from struct import unpack, unpack_from


def get_uuid():
    return uuid4()


def get_timestamp():
    return int(time())


def unpack_cfloat_array(mem):
    l = int(len(mem) / 4)
    return unpack(f"{l}f", mem)
