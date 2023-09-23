import math

import numpy as np
import matplotlib.pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# Author: Bernhard Eisvogel 10.09.23
import time
print("Sie haben das Programm zur Visualisierrung des ARZ-Modells \nmit Hilfe des Godunov-Schemas gestartet.")
# %% Vorbereitung
XLimit = 1
TLimit = 1

dx = 1/600
CFL = 0.5
dt = dx * CFL
print("CFL-Bedingung: ", CFL)

xminusLimit = 0
x = np.arange(-xminusLimit, XLimit, dx)
t = np.arange(0, TLimit, dt)
rho_max = 1
v_max = 1
def v_e(rho):
    if rho > rho_max:
        return 0
    if rho < 0:
        return v_max
    return v_max-rho

def v_eStrich(rho):
    if rho < 0:
        return 0
    if rho > 0.5:
        return 0
    return -1
def ve_minus1(v):
    if v > v_max:
        return 0
    if v < 0:
        return rho_max
    return v_max-v

def QStrichMinus1(q):
    if q < -1:
        return 0
    if q > 1:
        return 0
    return (1 -q)/2

def solveRiemann(v_links, rho_links, v_rechts, rho_rechts):
    if (v_rechts - v_links + v_e(rho_links)) > v_max:
        #Case 1
        if ((v_links + rho_links * v_eStrich(rho_links)) < 0):
            #Case 1.1
            rho_w = QStrichMinus1(-v_links + v_e(rho_links))
            v_w   =  v_e(rho_w) + v_links -v_e(rho_links)
            return (v_w,
                    rho_w,
                    rho_w * v_w,
                    rho_w * v_w * (v_links - v_e(rho_links)))
        else:
            #Case 1.2
            return (v_links,
                    rho_links,
                    v_links * rho_links,
                    rho_links * v_links * (v_links - v_e(rho_links)))

    elif((0 <= (v_rechts - v_links + v_e(rho_links))) and (v_max >= (v_rechts - v_links + v_e(rho_links)))):
        #Case 2
        rho_null = ve_minus1(v_rechts - v_links + v_e(rho_links))
        v_null   = v_rechts
        if (v_rechts <= v_links):
            #Case 2.1
             if (rho_null * v_rechts - v_links * rho_links <= 0):
                 # Case 2.1.1
                return (v_rechts,
                        rho_null,
                        rho_null * v_rechts,
                        rho_null * v_rechts * (v_links - v_e(rho_links)))
             else:
                 # Case 2.1.2
                return (v_links,
                        rho_links,
                        rho_links * v_links,
                        rho_links * v_links * (v_links - v_e(rho_links)))
        else:
            # Case 2.2
            if (v_links + rho_links * v_eStrich(rho_links) >= 0):
                # Case 2.2.1
                return (v_links,
                        rho_links,
                        rho_links * v_links,
                        rho_links * v_links * (v_links - v_e(rho_links)))

            elif(v_null + rho_null*ve_minus1(rho_null)) <= 0:
                # Case 2.2.2
                return  (v_rechts,
                         rho_null,
                         rho_null * v_rechts,
                         rho_null * v_rechts * (v_links - v_e(rho_links)))
            else:
                # Case 2.2.3
                pw = QStrichMinus1(v_links + v_e(rho_links))
                vw = v_e(pw) + v_links - v_e(rho_links)
                q = pw * vw
                return (vw,
                        pw,
                        q,
                        q * (v_links - v_e(rho_links)))
    else:
        # Case 3
        if (v_rechts >= (rho_links/rho_max) * v_links):
            # Case 3.1
            q = rho_links * v_links
            return (v_links,
                    rho_links,
                    q,
                    q * (v_links - v_e(rho_links)))
        else:
            # Case 3.2
            q = rho_max * v_rechts
            return (v_rechts,
                    rho_max,
                    q,
                    q * (v_links - v_e(rho_links)))

def simulation(startwerte):
    u = np.full((t.shape[0], x.shape[0], 2), 0.0)

    # Overload the arguments
    if callable(startwerte):
        x0 = np.array([startwerte(i) for i in x])
    else:
        x0 = startwerte

    # Border Conditions
    u[0, :] = x0
    u[:, 0, :] = x0[0]
    u[:, -1, :] = x0[-1]

    #For numerical Stability
    threshold = 0.00
    # %% Rechnen
    for i in range(1, u.shape[0]):
        if i%100 == 0:
            print(str(int(i/u.shape[0] * 100)) + "\t%")
        for j in range(1, u.shape[1]-1):
            y_links =       u[i - 1, j - 1, 1]
            rho_links =     u[i - 1, j - 1, 0]
            if abs(rho_links) < threshold:
                v_links = v_max
            else:
                v_links = y_links / rho_links + v_e(rho_links)

            y_mitte =       u[i - 1,      j, 1]
            rho_mitte =     u[i - 1,      j, 0]
            if abs(rho_mitte) < threshold:
                v_mitte = v_max
            else:
                v_mitte = y_mitte / rho_mitte + v_e(rho_mitte)

            y_rechts =      u[i - 1, j + 1, 1]
            rho_rechts =    u[i - 1, j + 1, 0]

            if abs(rho_rechts) < threshold:
                v_rechts = v_max
            else:
                v_rechts   = y_rechts / rho_rechts + v_e(rho_rechts)

            rhol, vl, ql, pl = solveRiemann(v_links, rho_links, v_mitte, rho_mitte)
            rhor, vr, qr, pr = solveRiemann(v_mitte, rho_mitte, v_rechts, rho_rechts)
            f1 = np.array([ql, pl])
            f2 = np.array([qr, pr])
            c = dt/dx
            u[i, j] = u[i-1, j] - c * (f2 - f1)
    print("Done")
    return u

def visualisieren(u, title = "test", beideTeile = True):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(title)
    ax1.set_title("rho")

    im1 = plt.imshow(u[:, :, 0], origin = "lower", extent = [xminusLimit, XLimit, 0,TLimit])
    plt.clim(vmin = 0, vmax = 1)
    plt.colorbar(im1)

    ax1.set_ylabel('t')
    ax1.set_xlabel('x')

    plt.savefig(dir_path + "/" + title + "GodunovSim.png")
    fig.show()

def createRiemannProblem(rho_links = 0.4,
           v_links   = 0.6,
           rho_rechts =  0.8,
           v_rechts   = 0.5):
    def startwerteFunktion(z):
        if z < 0.5:
            return np.array([rho_links,  rho_links * (v_links - v_e(rho_links))])
        else:
            return np.array([rho_rechts, rho_rechts * (v_rechts - v_e(rho_rechts))])
    return startwerteFunktion

def main(startwerte, title = "Visualisierung"):
    u = simulation(startwerte = startwerte)
    visualisieren(u, title = title)
    return u

def testRiemannCases():
    _ = main(startwerte = createRiemannProblem(rho_links = 0.6, v_links   = 0.6, rho_rechts =  0.8, v_rechts   = 0.7), title = "Case 1.1")
    _ = main(startwerte = createRiemannProblem(rho_links = 0.8, v_links   = 0.7, rho_rechts =  0.6, v_rechts   = 0.6), title = "Case 1.2")
    _ = main(startwerte = createRiemannProblem(rho_links = 0.7, v_links   = 0.7, rho_rechts =  0.2, v_rechts   = 0.2), title = "Case 3")
    _ = main(startwerte = createRiemannProblem(rho_links = 0.3, v_links   = 0.3, rho_rechts =  0.8, v_rechts   = 0.7), title = "Case 2")

if __name__ == '__main__':
    testRiemannCases()
