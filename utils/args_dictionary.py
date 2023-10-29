import os
import sys

from utils.logger import Logger as Log

def get_present_caller():
    filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
    lineno = sys._getframe().f_back.f_lineno
    prefix = '{}, {}'.format(filename, lineno)
    return prefix


def _get_caller():
    filename = os.path.basename(sys._getframe().f_back.f_back.f_code.co_filename)
    lineno = sys._getframe().f_back.f_back.f_lineno
    prefix = '{}, {}'.format(filename, lineno)
    return prefix


def get_key(args, *key):
    """

    :param args: ---> dictionary
    :param key:
    :return:
    """
    if len(key) == 1:
        if key[0] in args.keys():
            return args[key[0]]
        else:
            Log.error('{} KeyError: {}.'.format(_get_caller(), key))
            exit(1)

    elif len(key) == 2:
        if key[0] in args and key[1] in args[key[0]]:
            return args[key[0]][key[1]]
        else:
            Log.error('{} KeyError: {}.'.format(_get_caller(), key))
            exit(1)

    else:
        Log.error('{} KeyError: {}.'.format(_get_caller(), key))
        exit(1)


def key_exist(args, *key):
    """

        :param args: ---> dictionary
        :param key:
        :return:
        """
    if len(key) == 1 and key[0] in args.keys():
        return True
    elif len(key) == 2 and (key[0] in args.keys() and key[1] in args[key[0]].keys()):
        return True

    return False