'''
Created on Dec 27, 2019

@author: Faizan
'''

import matplotlib as mpl

# has to be big enough to accomodate all plotted values
mpl.rcParams['agg.path.chunksize'] = 100000

from .main import PhaseAnnealing

from .plot import PhaseAnnealingPlot
