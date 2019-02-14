'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''

import os

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

from .simultexts import SimultaneousExtremes, SimultaneousExtremesPlot
