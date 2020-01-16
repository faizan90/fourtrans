'''
Created on Dec 27, 2019

@author: Faizan
'''

import matplotlib as mpl

mpl.rc('font', size=16)

# has to be big enough to accomodate all plotted values
mpl.rcParams['agg.path.chunksize'] = 100000

from .main import PhaseAnnealing

from .plot import PhaseAnnealingPlot
