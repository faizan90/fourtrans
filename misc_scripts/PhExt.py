import numpy as np
import os
import matplotlib.pyplot as plt
# import datetime
from scipy.stats import rankdata
# from scipy import spatial as scsp
# import scipy.stats as sct
from scipy.special import ndtri
from scipy import cluster
# import random
from scipy.stats import spearmanr

# from sklearn.cluster import AgglomerativeClustering


def FindPeaks(sig, ndb):
    isig = np.argsort(sig)
    peak = []
    peak = np.append(peak, isig[-1])
    kk = 1
    for _ in range(ndb - 1):
        nk = 0
        while nk < 100:
            kk = kk + 1
            ik = peak - isig[-kk]
            nk = np.min(ik ** 2)
        peak = np.append(peak, isig[-kk])

#    print(peak)
    return peak


def CommonPeaks(sig1, sig2, ndb, nlim):
    nc = 0.0
    for i in range(ndb):
        for j in range(i, ndb, 1):
            dlt = np.abs(sig1[i] - sig2[j])
            if dlt < nlim + 0.1:
                nc = nc + 1.0
    return nc


def read_disch():
    path = "d:\\Spate\\spate_daily_discharge_reorganized_with_figs\\Neckar\\"
    filn = "neckar_discharge_data_AB.csv"
    filn = path + filn
    ab = np.loadtxt(filn, skiprows=1, delimiter=";")
    return ab


def read_disch_NS():
    path = "d:\\Spate\\spate_daily_discharge_reorganized_with_figs\\Niedersachsen\\"
#    filn = "niedersachsen_daily_discharge_AB.csv"
    filn = "niedersachsen_daily_discharge_AB.csv"
    filn = path + filn
    ab = np.loadtxt(filn, skiprows=1, delimiter=";")
    return ab


def read_disch_OE():
    path = "d:\\Spate\\spate_daily_discharge_reorganized_with_figs\\Oesterreich\\"
    filn = "oesterreich_daily_discharge_AB.csv"
    filn = path + filn
    ab = np.loadtxt(filn, skiprows=1, delimiter=";")
    return ab


def read_disch_Neck():
    path = "d:\\Spate\\spate_daily_discharge_reorganized_with_figs\\Neckar\\"
    filn = "neckar_discharge_data_AB.csv"
    filn = path + filn
    ab = np.loadtxt(filn, skiprows=1, delimiter=";")
    return ab


def modNS():
#     path = "d:\\Spate\\spate_daily_discharge_reorganized_with_figs\\Niedersachsen\\"
#     filn = "niedersachsen_daily_discharge.csv"
    path = "d:\\Spate\\spate_daily_discharge_reorganized_with_figs\\Oesterreich\\"
    filn = "oesterreich_daily_discharge.csv"
    filn = path + filn
    fp = open("".join(filn), 'r')
#     giln = "niedersachsen_daily_discharge_AB.csv"
    giln = "oesterreich_daily_discharge_AB.csv"
    giln = path + giln
    g = open("".join(giln), 'w')
    cines = fp.readlines()
    sa = ";\n"
    sb = ";-9\n"
    s1 = "-"
    s2 = ";"
    s3 = ";;"
    s4 = ";-9;"
    for cine in cines:
        mystring = str(cine)
        mystring = mystring.replace(s1, s2, 1)
        mystring = mystring.replace(s1, s2, 1)

        mystring = mystring.replace(s3, s4, 200)
        mystring = mystring.replace(s3, s4, 200)
        mystring = mystring.replace(sa, sb, 1)
        g.writelines(mystring)

    return


def readBritDisch():
    path = "D:\\Discharge\\"
    fnam = "D:\\Discharge\\smnames.dat"
    if os.path.isfile("".join(fnam)):
        f = open("".join(fnam), 'r')
        dc = np.empty([18993, 19])
        k = 0
        for line in f:
            lin = line.strip()
            print(lin)
            fnm = path + lin
            stdata = np.loadtxt(fnm, skiprows=50, delimiter=",")
            print(stdata.shape)
            sta = stdata[stdata[:, 0] > 1960]
            print(sta.shape)
            dc[:, k] = sta[:, 3]
            k = k + 1
        print(k)
    return dc


def Ftester():
    ab = readBritDisch()
#    ab = read_disch_Neck()
#    hnam = 'd:\\Spate\\NieSa10_2.csv'
    hnam = 'UK_Ret.csv'

    gn = open("".join(hnam), 'w')
    hnam = 'UK_Obs.csv'

    hn = open("".join(hnam), 'w')

    for i1 in range(1, 15, 1):
        for i2 in range(i1 + 1, 16, 1):
            print(i1, i2)
    # i1 = 25
    # i2 = 47
            mshift = 2
            mshift = 0
            qa = ab[:, i1 + mshift]
            qb = ab[:, i2 + mshift]
            nc = ab.shape[0]
        # find common time period
            miss = np.minimum(qa, qb)
            qa = qa[miss > 0]
            qb = qb[miss > 0]
            print(nc, qa.shape)
            nc = qa.shape[0]
            nc = nc // 2
            nc = 2 * nc
            if nc != qa.shape[0]:
                qa = qa[:-1]
                qb = qb[:-1]
            if nc > 7300:

            # Transform to normal
                isort = rankdata(qa) / (nc + 1)
                gsiga = ndtri(isort)
                isort = rankdata(qb) / (nc + 1)
                gsigb = ndtri(isort)
            # Find spectrum
                spectruma = np.fft.fft(gsiga)
                spectruma = spectruma * 2 / nc
                magna = np.abs(spectruma)
                phasa = np.angle(spectruma)
                phasa1 = np.copy(phasa)
                spectrumb = np.fft.fft(gsigb)
                spectrumb = spectrumb * 2 / nc
                magnb = np.abs(spectrumb)
                phasb = np.angle(spectrumb)
                phasb1 = np.copy(phasb)
            # Phase differences
                phdiff = np.cos(phasa - phasb)
    #            print(phdiff)
                cc = phdiff * magna * magnb
                cd = np.sum(cc)
    #            print(cc,cd)
                astd = np.sum(magna ** 2)
                bstd = np.sum(magnb ** 2)
#                 cr = cd / np.sqrt(astd * bstd)
    #            print(cr)
                cn = np.corrcoef(gsiga, gsigb)
    #            print(cn)
                phasd = np.copy(phasb)
                phasc = np.copy(phasb)
                ct = magna * magnb
                cw = np.sum(ct)

                cxx = cw / np.sqrt(astd * bstd)
    #            print(i1,i2,cxx)

                su = 0
                isf = 0
                for ij in range(1, nc // 2, 1):
                    su = su + 2 * magna[ij] * magnb[ij]
                    if su < cd:
                        phasc[ij] = phasa[ij]
                        phasc[nc - ij] = phasa[nc - ij]
                    else:
                        if isf == 0:
                            print(su, ij)
                            isf = 1
                        phasc[ij] = phasa[ij] + 3.1417 / 2.0
                        phasc[nc - ij] = phasa[nc - ij] - 3.1417 / 2.0

                phasc1 = np.copy(phasc)

                cspe = np.copy(spectruma)
                cspe.real = magnb * np.cos(phasc)
                cspe.imag = magnb * np.sin(phasc)
                cspe = cspe * nc / 2
                dd = np.fft.ifft(cspe)
                ccr = dd.real
#                 cu = np.corrcoef(gsiga, ccr)
    #            print(cu)
                su = 0
                isf = 0
                for ik in range(1, nc // 2, 1):
                    ij = nc // 2 - ik
                    su = su + 2 * magna[ij] * magnb[ij]
                    if su < cd:
                        phasd[ij] = phasa[ij]
                        phasd[nc - ij] = phasa[nc - ij]
                    else:
                        if isf == 0:
                            print(su, ij)
                            isf = 1
                        phasd[ij] = phasa[ij] + 3.1417 / 2.0
                        phasd[nc - ij] = phasa[nc - ij] - 3.1417 / 2.0
                phasd1 = np.copy(phasd)

                cspd = np.copy(spectruma)
                cspd.real = magnb * np.cos(phasd)
                cspd.imag = magnb * np.sin(phasd)
                cspd = cspd * nc / 2
                dd = np.fft.ifft(cspd)
                ddr = dd.real
#                 cu = np.corrcoef(gsiga, ddr)
#                 cdx = np.corrcoef(ccr, ddr)
    #            print(cu)
    #            print(cdx)
    #            print(phasb[:10])
    #            print(phasd[:10])

    #            plt.plot(gsiga[1000:1500])
    #            plt.plot(gsigb[1000:1500], "-")
    #            plt.plot(ccr[1000:1500], "--")
    #            plt.plot(ddr[1000:1500], "--")
    #            plt.show()
                for nrt in (5, 10, 20):
                    ndb = nc // (365 * nrt)
                    for nlim in (0, 2, 4, 9):

                        pa = FindPeaks(gsiga, ndb)
                        pb = FindPeaks(gsigb, ndb)
                        pc = FindPeaks(ccr, ndb)
                        pd = FindPeaks(ddr, ndb)

                        nab1 = CommonPeaks(pa, pb, ndb, nlim)
                        nac1 = CommonPeaks(pa, pc, ndb, nlim)
                        nad1 = CommonPeaks(pa, pd, ndb, nlim)
                        ab1 = nab1 / ndb
                        ac1 = nac1 / ndb
                        ad1 = nad1 / ndb
                        print(nab1, nac1, nad1)
                nh = "%i," % i1 + "%i," % i2
                nh = nh + "%i," % nrt + "%i," % nlim + "%i," % ndb
                nh = nh + "%0.3f" % ab1
                nh = nh + ",%0.3f," % ac1 + "%0.3f\n" % ad1
                print(nh)
                hn.write(nh)
                nsim = 200
                mab = np.empty([nsim, 12])
                mac = np.empty([nsim, 12])
                mad = np.empty([nsim, 12])

                for ir in range(nsim):
                    rp = np.random.rand(nc // 2) * 2 * 3.1417
                    phasa[:nc // 2] = phasa1[:nc // 2] + rp
                    phasa[nc // 2:] = phasa1[nc // 2:] - rp
                    phasb[:nc // 2] = phasb1[:nc // 2] + rp
                    phasb[nc // 2:] = phasb1[nc // 2:] - rp
                    phasc[:nc // 2] = phasc1[:nc // 2] + rp
                    phasc[nc // 2:] = phasc1[nc // 2:] - rp
                    phasd[:nc // 2] = phasd1[:nc // 2] + rp
                    phasd[nc // 2:] = phasd1[nc // 2:] - rp
                    cspa = np.copy(spectruma)
                    cspa.real = magna * np.cos(phasa)
                    cspa.imag = magna * np.sin(phasa)
                    cspa = cspa * nc / 2
                    aa = np.fft.ifft(cspa)
                    aar = aa.real
                    cspb = np.copy(spectruma)
                    cspb.real = magnb * np.cos(phasb)
                    cspb.imag = magnb * np.sin(phasb)
                    cspb = cspb * nc / 2
                    bb = np.fft.ifft(cspb)
                    bbr = bb.real
                    cspc = np.copy(spectruma)
                    cspc.real = magnb * np.cos(phasc)
                    cspc.imag = magnb * np.sin(phasc)
                    cspc = cspc * nc / 2
                    cc = np.fft.ifft(cspc)
                    ccr = cc.real
                    cspd = np.copy(spectruma)
                    cspd.real = magnb * np.cos(phasd)
                    cspd.imag = magnb * np.sin(phasd)
                    cspd = cspd * nc / 2
                    dd = np.fft.ifft(cspd)
                    ddr = dd.real
                    ik = 0
                    for nrt in (5, 10, 20):
                        ndb = nc // (365 * nrt)
                        for nlim in (0, 2, 4, 9):
                            pa = FindPeaks(aar, ndb)
                            pb = FindPeaks(bbr, ndb)
                            pc = FindPeaks(ccr, ndb)
                            pd = FindPeaks(ddr, ndb)
                            nab = CommonPeaks(pa, pb, ndb, nlim) / ndb
                            nac = CommonPeaks(pa, pc, ndb, nlim) / ndb
                            nad = CommonPeaks(pa, pd, ndb, nlim) / ndb
            #        print(nab, nac, nad)
                            mab[ir, ik] = nab
                            mac[ir, ik] = nac
                            mad[ir, ik] = nad
                            ik = ik + 1
                ik = 0
                for nrt in (5, 10, 20):
                    ndb = nc // (365 * nrt)
                    for nlim in (0, 2, 4, 9):
                        nh = "%i," % nrt + "%i," % nlim
                        nh = nh + "%i" % i1 + ",%i," % i2 + "%0.4f," % cn[0, 1]
                        nh = nh + "%0.4f," % cxx
                        nh = nh + "%0.3f," % np.average(mab[:, ik])
                        nh = nh + "%0.3f," % np.average(mac[:, ik])
                        nh = nh + "%0.3f\n" % np.average(mad[:, ik])
                        print(nh)
                        gn.write(nh)
                        ik = ik + 1


def Ftest_Half():
    ab = readBritDisch()
#    ab = read_disch_Neck()
#    hnam = 'd:\\Spate\\NieSa10_2.csv'
    hnam = 'UK_Ret.csv'

    gn = open("".join(hnam), 'w')
    hnam = 'UK_Obs.csv'

    hn = open("".join(hnam), 'w')

    for i1 in range(1, 15, 1):
        for i2 in range(i1 + 1, 16, 1):
            print(i1, i2)
    # i1 = 25
    # i2 = 47
            mshift = 2
            mshift = 0
            qa = ab[:, i1 + mshift]
            qb = ab[:, i2 + mshift]
            nc = ab.shape[0]
        # find common time period
            miss = np.minimum(qa, qb)
            qa = qa[miss > 0]
            qb = qb[miss > 0]
            print(nc, qa.shape)
            nc = qa.shape[0]
            nc = nc // 2
            nc = 2 * nc
            if nc != qa.shape[0]:
                qa = qa[:-1]
                qb = qb[:-1]
            qar = np.copy(qa)
            qbr = np.copy(qb)
            ncr = nc + 0
            if nc > 14600:
                nc = ncr // 2
                qa = qar[:nc]
                qb = qbr[:nc]

            # Transform to normal
                isort = rankdata(qa) / (nc + 1)
                gsiga = ndtri(isort)
                isort = rankdata(qb) / (nc + 1)
                gsigb = ndtri(isort)
            # Find spectrum
                spectruma = np.fft.fft(gsiga)
                spectruma = spectruma * 2 / nc
                magna = np.abs(spectruma)
                phasa = np.angle(spectruma)
                phasa1 = np.copy(phasa)
                spectrumb = np.fft.fft(gsigb)
                spectrumb = spectrumb * 2 / nc
                magnb = np.abs(spectrumb)
                phasb = np.angle(spectrumb)
                phasb1 = np.copy(phasb)
            # Phase differences
                phdiff = np.cos(phasa - phasb)
    #            print(phdiff)
                cc = phdiff * magna * magnb
                cd = np.sum(cc)
    #            print(cc,cd)
                astd = np.sum(magna ** 2)
                bstd = np.sum(magnb ** 2)
#                 cr = cd / np.sqrt(astd * bstd)
    #            print(cr)
                cn = np.corrcoef(gsiga, gsigb)
    #            print(cn)
                phasd = np.copy(phasb)
                phasc = np.copy(phasb)
                ct = magna * magnb
                cw = np.sum(ct)

                cxx = cw / np.sqrt(astd * bstd)
    #            print(i1,i2,cxx)

                su = 0
                isf = 0
                for ij in range(1, nc // 2, 1):
                    su = su + 2 * magna[ij] * magnb[ij]
                    if su < cd:
                        phasc[ij] = phasa[ij]
                        phasc[nc - ij] = phasa[nc - ij]
                    else:
                        if isf == 0:
                            print(su, ij)
                            isf = 1
                        phasc[ij] = phasa[ij] + 3.1417 / 2.0
                        phasc[nc - ij] = phasa[nc - ij] - 3.1417 / 2.0

                phasc1 = np.copy(phasc)

                cspe = np.copy(spectruma)
                cspe.real = magnb * np.cos(phasc)
                cspe.imag = magnb * np.sin(phasc)
                cspe = cspe * nc / 2
                dd = np.fft.ifft(cspe)
                ccr = dd.real
#                 cu = np.corrcoef(gsiga, ccr)
    #            print(cu)
                su = 0
                isf = 0
                for ik in range(1, nc // 2, 1):
                    ij = nc // 2 - ik
                    su = su + 2 * magna[ij] * magnb[ij]
                    if su < cd:
                        phasd[ij] = phasa[ij]
                        phasd[nc - ij] = phasa[nc - ij]
                    else:
                        if isf == 0:
                            print(su, ij)
                            isf = 1
                        phasd[ij] = phasa[ij] + 3.1417 / 2.0
                        phasd[nc - ij] = phasa[nc - ij] - 3.1417 / 2.0
                phasd1 = np.copy(phasd)

                cspd = np.copy(spectruma)
                cspd.real = magnb * np.cos(phasd)
                cspd.imag = magnb * np.sin(phasd)
                cspd = cspd * nc / 2
                dd = np.fft.ifft(cspd)
                ddr = dd.real
#                 cu = np.corrcoef(gsiga, ddr)
#                 cdx = np.corrcoef(ccr, ddr)
    #            print(cu)
    #            print(cdx)
    #            print(phasb[:10])
    #            print(phasd[:10])

    #            plt.plot(gsiga[1000:1500])
    #            plt.plot(gsigb[1000:1500], "-")
    #            plt.plot(ccr[1000:1500], "--")
    #            plt.plot(ddr[1000:1500], "--")
    #            plt.show()
                for nrt in (5, 10, 20):
                    ndb = nc // (365 * nrt)
                    for nlim in (0, 2, 4, 9):

                        pa = FindPeaks(gsiga, ndb)
                        pb = FindPeaks(gsigb, ndb)
                        pc = FindPeaks(ccr, ndb)
                        pd = FindPeaks(ddr, ndb)

                        nab1 = CommonPeaks(pa, pb, ndb, nlim)
                        nac1 = CommonPeaks(pa, pc, ndb, nlim)
                        nad1 = CommonPeaks(pa, pd, ndb, nlim)
                        ab1 = nab1 / ndb
                        ac1 = nac1 / ndb
                        ad1 = nad1 / ndb
                        print(nab1, nac1, nad1)
                nh = "%i," % i1 + "%i," % i2
                nh = nh + "%i," % nrt + "%i," % nlim + "%i," % ndb
                nh = nh + "%0.3f" % ab1
                nh = nh + ",%0.3f," % ac1 + "%0.3f\n" % ad1
                print(nh)
                hn.write(nh)
                nsim = 200
                mab = np.empty([nsim, 12])
                mac = np.empty([nsim, 12])
                mad = np.empty([nsim, 12])

                for ir in range(nsim):
                    rp = np.random.rand(nc // 2) * 2 * 3.1417
                    phasa[:nc // 2] = phasa1[:nc // 2] + rp
                    phasa[nc // 2:] = phasa1[nc // 2:] - rp
                    phasb[:nc // 2] = phasb1[:nc // 2] + rp
                    phasb[nc // 2:] = phasb1[nc // 2:] - rp
                    phasc[:nc // 2] = phasc1[:nc // 2] + rp
                    phasc[nc // 2:] = phasc1[nc // 2:] - rp
                    phasd[:nc // 2] = phasd1[:nc // 2] + rp
                    phasd[nc // 2:] = phasd1[nc // 2:] - rp
                    cspa = np.copy(spectruma)
                    cspa.real = magna * np.cos(phasa)
                    cspa.imag = magna * np.sin(phasa)
                    cspa = cspa * nc / 2
                    aa = np.fft.ifft(cspa)
                    aar = aa.real
                    cspb = np.copy(spectruma)
                    cspb.real = magnb * np.cos(phasb)
                    cspb.imag = magnb * np.sin(phasb)
                    cspb = cspb * nc / 2
                    bb = np.fft.ifft(cspb)
                    bbr = bb.real
                    cspc = np.copy(spectruma)
                    cspc.real = magnb * np.cos(phasc)
                    cspc.imag = magnb * np.sin(phasc)
                    cspc = cspc * nc / 2
                    cc = np.fft.ifft(cspc)
                    ccr = cc.real
                    cspd = np.copy(spectruma)
                    cspd.real = magnb * np.cos(phasd)
                    cspd.imag = magnb * np.sin(phasd)
                    cspd = cspd * nc / 2
                    dd = np.fft.ifft(cspd)
                    ddr = dd.real
                    ik = 0
                    for nrt in (5, 10, 20):
                        ndb = nc // (365 * nrt)
                        for nlim in (0, 2, 4, 9):
                            pa = FindPeaks(aar, ndb)
                            pb = FindPeaks(bbr, ndb)
                            pc = FindPeaks(ccr, ndb)
                            pd = FindPeaks(ddr, ndb)
                            nab = CommonPeaks(pa, pb, ndb, nlim) / ndb
                            nac = CommonPeaks(pa, pc, ndb, nlim) / ndb
                            nad = CommonPeaks(pa, pd, ndb, nlim) / ndb
            #        print(nab, nac, nad)
                            mab[ir, ik] = nab
                            mac[ir, ik] = nac
                            mad[ir, ik] = nad
                            ik = ik + 1
                ik = 0
                for nrt in (5, 10, 20):
                    ndb = nc // (365 * nrt)
                    for nlim in (0, 2, 4, 9):
                        nh = "%i," % nrt + "%i," % nlim
                        nh = nh + "%i" % i1 + ",%i," % i2 + "%0.4f," % cn[0, 1]
                        nh = nh + "%0.4f," % cxx
                        nh = nh + "%0.3f," % np.average(mab[:, ik])
                        nh = nh + "%0.3f," % np.average(mac[:, ik])
                        nh = nh + "%0.3f\n" % np.average(mad[:, ik])
                        print(nh)
                        gn.write(nh)
                        ik = ik + 1


def Clusterit():
    dmat = np.empty([99, 99], dtype=float)
    dmat.fill(-99.9)
    print(dmat.shape)

    path = "d:\\Spate\\"
    filn = "OE_RetA.csv"
    filn = path + filn
    zvA = np.loadtxt(filn, delimiter=",")
    filn = "OE_RetB.csv"
    filn = path + filn
    zvB = np.loadtxt(filn, delimiter=",")
    filn = "OE_RetC.csv"
    filn = path + filn
    zvC = np.loadtxt(filn, delimiter=",")
    zv = np.append(zvA, zvB)
    zv = np.append(zv, zvC)
    zv = np.reshape(zv, [zv.shape[0] // 9, 9])
    print(zv.shape)
    print(dmat.shape)
    for i in range(zv.shape[0]):
        i1 = zv[i, 2].astype(int) - 1
        i2 = zv[i, 3].astype(int) - 1
        if zv[i, 0] == 20 and zv[i, 1] == 2:
            dmat[i1, i2] = 1.0 - zv[i, 6]
            dmat[i2, i1] = 1.0 - zv[i, 6]
            dmat[i1, i1] = 0.0
    new = np.copy(dmat)
    thelp = np.sum(new, 0)
    thelp = np.argsort(thelp)
    kk = 1
    while np.min(new) < 0:
        new = np.copy(dmat)
        new = np.delete(new, thelp[:kk], 0)
        new = np.delete(new, thelp[:kk], 1)
        kk = kk + 1

    print(kk)
    print(new)
    new = np.triu(new, k=1)
    new = new.flatten()
    new = new[new > 0]
    print(new.shape)
#    new = np.sort(new)
#    new = new[:28]
    clustering = cluster.hierarchy.ward(new)
    print(clustering)
    cluster.hierarchy.dendrogram(clustering)
    nam = "dendro_20_3_OE.png"
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    cluster.hierarchy.dendrogram(
        clustering,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig(nam, bbox_inches='tight')
    plt.close()


def Likely():
    dmat = np.empty([99, 99], dtype=float)
    dmat.fill(-99.9)
    print(dmat.shape)

    path = "d:\\Spate\\"
    filn = "OE_RetA.csv"
    filn = path + filn
    zvA = np.loadtxt(filn, delimiter=",")
    filn = "OE_RetB.csv"
    filn = path + filn
    zvB = np.loadtxt(filn, delimiter=",")
    filn = "OE_RetC.csv"
    filn = path + filn
    zvC = np.loadtxt(filn, delimiter=",")
    zv = np.append(zvA, zvB)
    zv = np.append(zv, zvC)
    zv = np.reshape(zv, [zv.shape[0] // 9, 9])
    print(zv.shape)
    print(dmat.shape)
    for i in range(zv.shape[0]):
        i1 = zv[i, 2].astype(int) - 1
        i2 = zv[i, 3].astype(int) - 1
        if zv[i, 0] == 20 and zv[i, 1] == 2:
            dmat[i1, i2] = 1.0 - zv[i, 6]
            dmat[i2, i1] = 1.0 - zv[i, 6]
            dmat[i1, i1] = 0.0
    new = np.copy(dmat)
    thelp = np.sum(new, 0)
    thelp = np.argsort(thelp)
    kk = 1
    while np.min(new) < 0:
        new = np.copy(dmat)
        new = np.delete(new, thelp[:kk], 0)
        new = np.delete(new, thelp[:kk], 1)
        kk = kk + 1

    print(kk)
    print(new)
    new = np.triu(new, k=1)
    new = new.flatten()
    new = new[new > 0]
    print(new.shape)
#    new = np.sort(new)
#    new = new[:28]
    clustering = cluster.hierarchy.ward(new)
    print(clustering)
    cluster.hierarchy.dendrogram(clustering)
    nam = "dendro_20_3_OE.png"
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    cluster.hierarchy.dendrogram(
        clustering,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig(nam, bbox_inches='tight')
    plt.close()


def Evalit():

    path = "d:\\Spate\\"
    filn = "OE_RetA.csv"
    filn = path + filn
    zvA = np.loadtxt(filn, delimiter=",")
    filn = "OE_RetB.csv"
    filn = path + filn
    zvB = np.loadtxt(filn, delimiter=",")
    filn = "OE_RetC.csv"
    filn = path + filn
    zvC = np.loadtxt(filn, delimiter=",")
    filn = "OE_RetD.csv"
    filn = path + filn
    zvD = np.loadtxt(filn, delimiter=",")
    zv = np.append(zvA, zvB)
    zv = np.append(zv, zvC)
    zv = np.append(zv, zvD)
    zv = np.reshape(zv, [zv.shape[0] // 9, 9])
    zvcopy = np.copy(zv)
    print(zv.shape)
    zv = zv[zv[:, 0] == 5]
    zv = zv[zv[:, 1] == 0]
    ilev = 54
    imx = -2
    imi = 1
    xlow = ilev / 100.0 - 0.005
    xhigh = ilev / 100.0 + 0.005
    zv = zv[zv[:, 4] > xlow]
    zv = zv[zv[:, 4] < xhigh]
    zvmax = np.max(zv, 0)
    zvmin = np.min(zv, 0)
    print(zvmax, zvmin)
    zp = zv[:, 6]
    ip = np.argsort(zp)
    print(zv[ip[imi], :])
    print(zv[ip[imx], :])

    print(zv.shape)
    ab = read_disch_OE()
    i1 = zv[ip[imi], 2].astype(int)
    i2 = zv[ip[imi], 3].astype(int)
    i11 = i1 + 0
    i12 = i2 + 0
    qa = ab[:, i1 + 2]
    qb = ab[:, i2 + 2]
        # find common time period
    miss = np.minimum(qa, qb)
    qa = qa[miss > 0]
    qb = qb[miss > 0]
    nc = qb.shape[0]
    ncab = nc + 0
    cu = np.corrcoef(qa, qb)
    print(cu, i1, i2, zv[ip[imi], 6])
    nrt = 5
    nlim = 0
    ndb = nc // (365 * nrt)
    pa = FindPeaks(qa, ndb)
    pb = FindPeaks(qb, ndb)
    nab1 = CommonPeaks(pa, pb, ndb, nlim)
    print(nab1, nc)
    i1 = zv[ip[imx], 2].astype(int)
    i2 = zv[ip[imx], 3].astype(int)
    qc = ab[:, i1 + 2]
    qd = ab[:, i2 + 2]
        # find common time period
    miss = np.minimum(qc, qd)
    qc = qc[miss > 0]
    qd = qd[miss > 0]
    nc = qc.shape[0]
    nccd = nc + 0
    cu = np.corrcoef(qc, qd)
    ndb = nc // (365 * nrt)
    pa = FindPeaks(qc, ndb)
    pb = FindPeaks(qd, ndb)
    nab1 = CommonPeaks(pa, pb, ndb, nlim)
    print(nab1, nc)

    print(cu, i1, i2, zv[ip[imx], 6])
    aa = np.arange(0, nc, 1)
    jj = 1900
    db = 200
    nam = "test%ia.png" % ilev
    ax = np.max(qa[jj:jj + db])
    bx = np.max(qb[jj:jj + db])
    plt.plot(aa[jj:jj + db], qa[jj:jj + db] / ax)
    plt.plot(aa[jj:jj + db], qb[jj:jj + db] / bx)
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized discharge')
    plt.savefig(nam, bbox_inches='tight')
    plt.close()
    nam = "test%ib.png" % ilev
    cx = np.max(qc[jj:jj + db])
    dx = np.max(qd[jj:jj + db])
    plt.plot(aa[jj:jj + db], qc[jj:jj + db] / cx)
    plt.plot(aa[jj:jj + db], qd[jj:jj + db] / dx)
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized discharge ')
    plt.savefig(nam, bbox_inches='tight')
    plt.close()

# numpy.savetxt("mydata.csv", a, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')

    isorta = rankdata(qa) / (ncab + 1)
    gsiga = ndtri(isorta)
    isortb = rankdata(qb) / (ncab + 1)
    gsigb = ndtri(isortb)
    spc = np.corrcoef(isorta, isortb)
    print("Spab", spc)
# Find spectrum
    spectruma = np.fft.fft(gsiga)
    spectruma = spectruma * 2 / ncab
    magna = np.abs(spectruma)
    phasa = np.angle(spectruma)
#     phasa1 = np.copy(phasa)
    spectrumb = np.fft.fft(gsigb)
    spectrumb = spectrumb * 2 / ncab
    magnb = np.abs(spectrumb)
    phasb = np.angle(spectrumb)
#     phasb1 = np.copy(phasb)
# Phase differences
    phdiff = np.cos(phasa - phasb)
#            print(phdiff)
    cc = phdiff * magna * magnb
    cd = np.sum(cc)
    csum = 0.0
    cab = []
    for i in range(len(cc) // 2):
        csum = csum + cc[i] / cd
        cab = np.append(cab, csum)

    isortc = rankdata(qc) / (nccd + 1)
    gsiga = ndtri(isortc)
    isortd = rankdata(qd) / (nccd + 1)
    gsigb = ndtri(isortd)
    spc = np.corrcoef(gsiga, gsigb)
    print("Spcd", spc)

    # Find spectrum
    spectruma = np.fft.fft(gsiga)
    spectruma = spectruma * 2 / nccd
    magna = np.abs(spectruma)
    phasa = np.angle(spectruma)
#     phasa1 = np.copy(phasa)
    spectrumb = np.fft.fft(gsigb)
    spectrumb = spectrumb * 2 / nccd
    magnb = np.abs(spectrumb)
    phasb = np.angle(spectrumb)
#     phasb1 = np.copy(phasb)
    # Phase differences
    phdiff = np.cos(phasa - phasb)
    #            print(phdiff)
    cc = phdiff * magna * magnb
    cd = np.sum(cc)
    csum = 0.0
    ccd = []
    for i in range(len(cc) // 2):
        csum = csum + cc[i] / cd
        ccd = np.append(ccd, csum)
    nam = "cumul%iab.png" % ilev
    plt.xlabel('Wave (k)')
    plt.ylabel('Cumulative contribution')
    plt.plot(cab[:3000] * 2)
    plt.plot(ccd[:3000] * 2)
    plt.savefig(nam, bbox_inches='tight')
    plt.close()
    miss = np.minimum(isorta, isortb)
    ia = isorta.shape[0]

    tlim = 0.99
    tfact = (1. - tlim) ** 2

    isorta = isorta[miss > tlim]
    isortb = isortb[miss > tlim]
    tp = isorta.shape[0] / (ia * tfact)
    tq = tp * (1. - tlim)
    print("Cdens_a", tp, tq)
    nam = "cop%iabx.png" % ilev

    plt.scatter(isorta, isortb)
    plt.savefig(nam, bbox_inches='tight')
    plt.close()

    ic = isortc.shape[0]

    miss = np.minimum(isortc, isortd)
    isortc = isortc[miss > tlim]
    isortd = isortd[miss > tlim]
    tp = isortc.shape[0] / (ic * tfact)
    tq = tp * (1. - tlim)
    print("Cdens_c", tp, tq)

    nam = "cop%icdx.png" % ilev

    plt.scatter(isortc, isortd)
    plt.savefig(nam, bbox_inches='tight')
    plt.close()

    hnam = 'table_%icd.tex' % ilev
    gn = open("".join(hnam), 'w')

    zred = zvcopy[zvcopy[:, 2] == i1]
    zred = zred[zred[:, 3] == i2]
    print(zred)

    nh = " 5 &" + " %0.3f &" % zred[0, 6] + " %0.3f &" % zred[1, 6] + " %0.3f &" % zred[2, 6] + " %0.3f \\" % zred[3, 6] + "\\ \n"
    gn.write(nh)
    nh = " 10 &" + " %0.3f &" % zred[4, 6] + " %0.3f &" % zred[5, 6] + " %0.3f &" % zred[6, 6] + " %0.3f \\" % zred[7, 6] + "\\ \n"
    gn.write(nh)
    nh = " 20 &" + " %0.3f &" % zred[8, 6] + " %0.3f &" % zred[9, 6] + " %0.3f &" % zred[10, 6] + " %0.3f \\" % zred[11, 6] + "\\ \n"
    gn.write(nh)
    nh = "(%i," % i1 + "%i) \n" % i2
    gn.write(nh)

    hnam = 'table_%iab.tex' % ilev
    gn = open("".join(hnam), 'w')

    zred = zvcopy[zvcopy[:, 2] == i11]
    zred = zred[zred[:, 3] == i12]
    print(zred)

    nh = " 5 &" + " %0.3f &" % zred[0, 6] + " %0.3f &" % zred[1, 6] + " %0.3f &" % zred[2, 6] + " %0.3f \\" % zred[3, 6] + "\\ \n"
    gn.write(nh)
    nh = " 10 &" + " %0.3f &" % zred[4, 6] + " %0.3f &" % zred[5, 6] + " %0.3f &" % zred[6, 6] + " %0.3f \\" % zred[7, 6] + "\\ \n"
    gn.write(nh)
    nh = " 20 &" + " %0.3f &" % zred[8, 6] + " %0.3f &" % zred[9, 6] + " %0.3f &" % zred[10, 6] + " %0.3f \\" % zred[11, 6] + "\\ \n"
    gn.write(nh)
    nh = "(%i," % i11 + "%i) \n" % i12
    gn.write(nh)


def SelfTest():
#    ab = readBritDisch()
    ab = read_disch_NS()
    idg = 0
    icg = 0
    iall = 0
    nsumc = 0
    nsumd = 0
    nsume = 0

    hnam = 'NS_Self09.csv'
    hn = open("".join(hnam), 'w')

    for i1 in range(1, 250, 1):
        mshift = 2
#        mshift = 0
        qa = ab[:, i1 + mshift]
        nc = ab.shape[0]
    # find common time period
        qa = qa[qa > 0]
        nc = qa.shape[0]
        nc = nc // 2
        nc = 2 * nc
        if nc != qa.shape[0]:
            qa = qa[:-1]
        if nc > 7300:
            sqa = np.sort(qa)
        # Transform to normal
            isort = rankdata(qa) / (nc + 1)
            gsiga = ndtri(isort)
        # Find spectrum
            spectruma = np.fft.fft(gsiga)
            spectruma = spectruma * 2 / nc
            magna = np.abs(spectruma)
            phasa = np.angle(spectruma)
            phasc = np.copy(phasa)
#             phasd = np.copy(phasa)
            pars = np.sum(magna ** 2)
            rh1 = 0.75
            rup = 1.0
            rdo = 0.0
            rtarget = 0.9
            isf = np.argsort(magna[:nc // 2])[-1].astype(int)
            erc = 1.0
            isbad = 0
            inodo = 0
            while (erc > 0.05) and (isbad < 15):
                phasc[isf] = phasa[isf]
                phasc[nc - isf] = phasa[nc - isf]
                su = 2 * magna[isf] * magna[isf]
                print(su, isf, magna[isf] ** 2, magna[1] ** 2)
                for ij in range(1, nc // 2, 1):
                    if ij != isf:
                        su = su + 2 * magna[ij] * magna[ij]
                        if su < rh1 * pars:
                            sbb = su + 0.0
                            ibb = ij + 0
                            phasc[ij] = phasa[ij]
                            phasc[nc - ij] = phasa[nc - ij]
                        else:
                            phasc[ij] = phasa[ij] + 3.1417  # /2.0
                            phasc[nc - ij] = phasa[nc - ij] - 3.1417  # /2.0
                print(su, pars, sbb, ibb, isf)
#                 phasc1 = np.copy(phasc)
                cspe = np.copy(spectruma)
                cspe.real = magna * np.cos(phasc)
                cspe.imag = magna * np.sin(phasc)
                cspe = cspe * nc / 2
                dd = np.fft.ifft(cspe)
                ccr = dd.real
                sccr = sqa[rankdata(ccr).astype(int) - 1]
                scc = np.corrcoef(sccr, qa)[0, 1]
                print(np.corrcoef(sccr, qa))
                erc = np.abs(rtarget - scc)
                if erc > 0.05:
                    if scc > rtarget:
                        rup = rh1
                        rh1 = (rh1 + rdo) / 2.0
                    else:
                        rdo = rh1
                        rh1 = (rh1 + rup) / 2.0
                    isbad = isbad + 1
                    print(scc, isbad, rh1, rup, rdo)
            if erc > 0.05:
                inodo = 1
            print(scc, inodo)

            rup = 1.0
            rdo = 0.0

            rh1 = 0.9
            erc = 1.0
            isbad = 0
            while (erc > 0.05) and (isbad < 15):
                phasc[isf] = phasa[isf]
                phasc[nc - isf] = phasa[nc - isf]
                sv = 2 * magna[isf] * magna[isf]
                for ij in range(nc // 2, 1, -1):
                    if ij != isf:
                        sv = sv + 2 * magna[ij] * magna[ij]
                        if sv < rh1 * pars:
                            sbb = sv + 0.0
                            ibb = ij + 0
                            phasc[ij] = phasa[ij]
                            phasc[nc - ij] = phasa[nc - ij]
                        else:
                            phasc[ij] = phasa[ij] + 3.1417  # / 2.0
                            phasc[nc - ij] = phasa[nc - ij] - 3.1417  # / 2.0

#                 phasc1 = np.copy(phasc)
                cspe = np.copy(spectruma)
                cspe.real = magna * np.cos(phasc)
                cspe.imag = magna * np.sin(phasc)
                cspe = cspe * nc / 2
                dd = np.fft.ifft(cspe)
                ddr = dd.real
                sddr = sqa[rankdata(ddr).astype(int) - 1]
                sdd = np.corrcoef(sddr, qa)[0, 1]
                print(np.corrcoef(sddr, qa), rh1, sbb, ibb)
                erc = np.abs(rtarget - sdd)
                if erc > 0.05:
                    if sdd > rtarget:
                        rup = rh1
                        rh1 = (rh1 + rdo) / 2.0
                    else:
                        rdo = rh1
                        rh1 = (rh1 + rup) / 2.0
                    isbad = isbad + 1
                    print(scc, isbad, rh1, rup, rdo)
            if erc > 0.05:
                inodo = 1
            print("sdd", sdd, inodo)

            angle = np.arccos(rtarget)
            print(angle, np.cos(angle))

            for ij in range(1, nc // 2, 1):
                phasc[ij] = phasa[ij] + angle
                phasc[nc - ij] = -phasc[ij]

#             phasc1 = np.copy(phasc)
            cspe = np.copy(spectruma)
            cspe.real = magna * np.cos(phasc)
            cspe.imag = magna * np.sin(phasc)
            cspe = cspe * nc / 2
            dd = np.fft.ifft(cspe)
            eer = dd.real
            seer = sqa[rankdata(eer).astype(int) - 1]
            print(np.corrcoef(seer, qa))
            # ddr = np.copy(eer)
            if inodo < 1 :
                for nrt in (5, 10, 20):
                    ndb = nc // (365 * nrt)
                    for nlim in (0, 9):

                        pa = FindPeaks(gsiga, ndb)
                        pc = FindPeaks(ccr, ndb)
                        pd = FindPeaks(ddr, ndb)
                        pe = FindPeaks(eer, ndb)

                        nac1 = CommonPeaks(pa, pc, ndb, nlim)
                        nad1 = CommonPeaks(pa, pd, ndb, nlim)
                        nae1 = CommonPeaks(pa, pe, ndb, nlim)
                        ac1 = nac1 / ndb
                        ad1 = nad1 / ndb
                        ae1 = nae1 / ndb
                        if nlim == 0 and nrt == 5:
                            nsumc = nsumc + nac1
                            nsumd = nsumd + nad1
                            nsume = nsume + nae1
                            iall = iall + 1
                            if ac1 < ad1:
                                idg = idg + 1

                            if ad1 < ac1:
                                icg = icg + 1

                            nh = "%i," % i1
                            nh = nh + "%i," % nrt + "%i," % nlim + "%i," % ndb
                            nh = nh + " %0.3f," % ac1 + " %0.3f," % ae1 + "%0.3f\n" % ad1
                            print(nh)
                            hn.write(nh)
                        print("ICG, IDG", icg, idg, iall)
                        print(nsumc, nsumd, nsume)


def Compare2():
    z1data = []
    iv = 1
    path = "d:\\Spate\\"
    fnam = "D:\\Spate\\OEdnames.dat"
    if os.path.isfile("".join(fnam)):
        f = open("".join(fnam), 'r')
        for line in f:
            lin = line.strip()
            print(lin)
            fnm = path + lin + "%i" % iv + ".csv"
            stdata = np.loadtxt(fnm, delimiter=",")
            z1data = np.append(z1data, stdata)
    z2data = []
    iv = 2
    path = "d:\\Spate\\"
    fnam = "D:\\Spate\\OEdnames.dat"
    if os.path.isfile("".join(fnam)):
        f = open("".join(fnam), 'r')
        for line in f:
            lin = line.strip()
            print(lin)
            fnm = path + lin + "%i" % iv + ".csv"
            stdata = np.loadtxt(fnm, delimiter=",")
            z2data = np.append(z2data, stdata)
    z1data = np.reshape(z1data, [z1data.shape[0] // 9, 9])
    z2data = np.reshape(z2data, [z2data.shape[0] // 9, 9])

    print(z1data.shape)
    z1d = z1data[z1data[:, 0] == 5]
    print(z1d.shape)
    print(z2data.shape)
    z2d = z2data[z2data[:, 0] == 5]
    z1d = z1d[z1d[:, 1] == 9]
    z2d = z2d[z2d[:, 1] == 9]
    print(z2d.shape)
    print(z1d.shape)
    ba = z1d[:, 6]
    bb = z2d[:, 6]
    print(np.corrcoef(ba, bb))
    print(spearmanr(ba, bb))

    nam = "OE_scd59.png"
    plt.xlabel('Probability time period 1')
    plt.ylabel('Probability time period 2')

    plt.scatter(ba, bb)
    plt.savefig(nam, bbox_inches='tight')
    plt.close()
#    plt.show()


def d_Margs():
    ab = read_disch_OE()
    ets1 = []
    ets2 = []

    for i1 in range(1, 98, 1):
        for i2 in range(i1 + 1, 98, 1):
            # i1 = 25
            # i2 = 47
            qa = ab[:, i1 + 2]
            qb = ab[:, i2 + 2]
            nc = ab.shape[0]
            # find common time period
            miss = np.minimum(qa, qb)
            qa = qa[miss > 0]
            qb = qb[miss > 0]
            print(nc, qa.shape)
            nc = qa.shape[0]
            qar = np.copy(qa)
            qbr = np.copy(qb)
#             ncr = nc + 0
            if nc > 14600:
                nc = nc // 2
                qa = qar[nc:]
                qb = qbr[nc:]
                qas = np.sort(qa)[-5] / np.mean(qar)
#                 qbs = np.sort(qb)[-5] / np.mean(qbr)
                ets2 = np.append(ets2, qas)
#                ets2 = np.append(ets2,qbs)
                qa = qar[:nc]
                qb = qbr[:nc]
                qas = np.sort(qa)[-5] / np.mean(qar)
#                 qbs = np.sort(qb)[-5] / np.mean(qbr)
                ets1 = np.append(ets1, qas)
#                ets1 = np.append(ets1,qbs)
    plt.xlabel('Q5/Qmean time period 1')
    plt.ylabel('Q5/Qmean time period 2')
    nam = "OE_5d5.png"

    plt.scatter(ets1, ets2)
    plt.savefig(nam, bbox_inches='tight')
    plt.close()
    # plt.show()


# d_Margs()
# Compare2()
Evalit()
# Ftester()
# modNS()
# Clusterit()
# SelfTest()
