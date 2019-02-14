#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:09:02 2019

@author: masoud
"""

import numpy as np
from scipy.stats import norm, rankdata
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.stats as st
import scipy
from scipy.special import ndtri

# %%
# change directory

import os
# os.chdir('/Users/masoud/my_dir_m')
os.chdir(r'P:\Synchronize\IWS\fourtrans_practice')

# %%
df = pd.read_csv(r'P:\Synchronize\IWS\DWD_meteo_hist_pres\daily_merged_data\precipitation.csv', sep=';')
df2A = df.loc[32872:37971, 'P1089']
df2A = df2A.values
# #%%
Rt = df2A


# %%
def FindFull(Rt):

    ialld = np.copy(Rt)
    ialld[ialld > -1] = 1
    ialld[ialld < -1] = 0
    npos = np.sum(ialld, 0)
    nlen = npos.astype(int)
    sdat = np.copy(Rt)
    nc = sdat.shape[0]
    nc2 = nc // 2
    nc = 2 * nc2
    if nc == sdat.shape[0]:
        sdat = sdat[:-1]
        nc2 = nc2 - 1
    nc = sdat.shape[0]
    pzero = sdat[sdat < 1].shape[0]
    pzero = pzero / (nlen + 1)
    isort1 = st.rankdata(sdat) / (nc + 1)
    isort1[isort1 < pzero] = pzero
    gsig1 = ndtri(isort1)

    spectrum = np.fft.fft(gsig1)
    spectrum = spectrum / gsig1.shape[0]

    magnitude = np.abs(spectrum)
    parc = np.sum(magnitude[1:] ** 2)
    print(pzero, parc)
    phase = np.angle(spectrum)
    phase1 = np.copy(phase)

    phase3 = np.copy(phase1)

    rp = np.random.rand(nc2) * 2 * 3.1417
    rp_rev = rp[::-1]
    phase3[1:nc2 + 1] = phase1[1:nc2 + 1] + rp
    phase3[nc2 + 1:] = phase1[nc2 + 1:] - rp_rev
    magalt = np.copy(magnitude)
    diffmin = 999999999.
    kmin = 0
    for kk in range(5):
        cspe = np.copy(spectrum)
        cspe.real = magalt * np.cos(phase3)
        cspe.imag = magalt * np.sin(phase3)
        cspe = cspe * gsig1.shape[0]
        dd1 = np.fft.ifft(cspe)
        dd = dd1.real
        ddtr = np.copy(dd)
        shelp = np.sort(dd)
#        npz = (pzero*nlen)
        npz = (pzero * nlen).astype(int)
        vzero = shelp[npz]
        print(pzero, npz, vzero)
        isnew = st.rankdata(dd) / (nc + 1)
        isnew[isnew < pzero] = pzero
    #    ddtr = ndtri(isnew)
    #    vzero = ndtri(pzero)/np.std(dd)
        ddtr[ddtr < vzero] = vzero
        print('Std-s', np.std(dd), np.std(ddtr))
#         amu = np.std(dd) / np.std(ddtr)
        specnew = np.fft.fft(ddtr)
        specnew = specnew / nc

        magnew = np.abs(specnew)
        parnew = np.sum(magnew[1:] ** 2)
        print(pzero, parc, parnew)
#         phasenew = np.angle(specnew)

        diff = magnew[1:] - magnitude[1:]
        ds = np.sum(diff ** 2)
        print(ds)
        rp = np.random.rand(nc - 1)
        ab = np.argsort(magnitude[1:])
        diff[:ab[nc - 2000]] = 0
        if ds < diffmin:
            diffmin = ds
            kmin = kk
            magbest = np.copy(magnew)
        magalt[1:] = magalt[1:] - diff
        magalt[magalt < 0] = magnitude[magalt < 0]
    print(kk, ds, diffmin, kmin)
#     plt.scatter(magnitude[1:], magbest[1:])
#     plt.show()

    cspe = np.copy(spectrum)
    cspe.real = magbest * np.cos(phase3)
    cspe.imag = magbest * np.sin(phase3)
    cspe = cspe * gsig1.shape[0]
    dd1 = np.fft.ifft(cspe)
    dd = dd1.real
    ddtr = np.copy(dd)
    shelp = np.sort(dd)
    npz = (pzero * nc).astype(int)
    vzero = shelp[npz]
    isnew = st.rankdata(ddtr).astype(int) - 1
    rsort = np.sort(Rt)
    nser = rsort[isnew]
    print(nser.shape)
    for i in range(10):
        aa = np.corrcoef(Rt, np.roll(Rt, i, 0))[0, 1]
        bb = np.corrcoef(nser, np.roll(nser, i, 0))[0, 1]
        print(i, aa, bb)

#     for ia in range(0, 20000, 200):
#         ie = ia + 200
#         plt.plot(nser[ia:ie])
#         plt.plot(Rt[ia:ie])
#         plt.show()

    return nser, magnitude, magbest, sdat

# %%


def PrcFull(InputRt, iteration, nOutput):
    Rt = InputRt

    sdat = np.copy(Rt)
    nc = sdat.shape[0]
    nc2 = nc // 2
    nc = 2 * nc2
    if nc == sdat.shape[0]:
        sdat = sdat[:-1]
        nc2 = nc2 - 1
    nc = sdat.shape[0]

    Gnser = np.zeros((iteration, nOutput, nc))

    for i in range(iteration):
        nser, magnitude, magbest, sdat = FindFull(Rt)
        Gnser[i, 0, :] = nser
        Gnser[i, 1, :] = magnitude
        Gnser[i, 2, :] = magbest
        Gnser[i, 3, :] = sdat
    return Gnser

# %%


Gnser = PrcFull(Rt, 5, 4)
sims = Gnser[:, 0, :]
