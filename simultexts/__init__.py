from multiprocessing import current_process

from .main import SimultaneousExtremes, SimultaneousExtremesPlot

current_process().authkey = 'simultexts'.encode(
    encoding='utf_8', errors='strict')
