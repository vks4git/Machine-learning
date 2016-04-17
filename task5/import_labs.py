import sys

__author__ = 'vks'


def import_labs(list):
    root = "/"
    parts = __file__.split('/')
    for i in range(1, len(parts) - 2):
        root += parts[i] + '/'
    for s in list:
        sys.path.append(root + s)
