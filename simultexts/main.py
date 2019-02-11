'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''

from .alg import SimultaneousExtremesAlgorithm as SEA
from .plot import SimultaneousExtremesPlot


class SimultaneousExtremes(SEA):

    def __init__(self, verbose=True, overwrite=True):

        SEA.__init__(self, verbose=verbose, overwrite=overwrite)
        return

    def verify(self):

        SEA._SimultaneousExtremesAlgorithm__verify(self)
        return

