# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:24:06 2023

@author: be
"""
import numpy as np
import matplotlib.pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

print("Sie haben das Programm zur Visualiserierung des ARZ-Modells für Verkehrsfluss gestartet.")
# %% Vorbereitung
XLimit = 1
TLimit = 1
xminusLimit = 0

CFL = 0.5
dx = 1/400
dt = dx * CFL

print("CFL-Bedingung: ", CFL)
x = np.arange(-xminusLimit, XLimit, dx)
t = np.arange(0, TLimit, dt)

def h(rho):
    if rho < 0:
        return 1
        print("rho is too small: ", rho)
        quit(0)
    if rho > 1:
        return 0
        print("rho > rhomax = 1,: ", rho)
        quit(0)

    return 1-rho

def f1(p, pressure):
    return pressure + p * h(p)
    #return p - 2 * pressure
def f2(p, pressure):
    if p < 0 or p > 1:
        print(p)
        quit()
    return (pressure**2)/p + pressure * h(p)
    #return 2*p - pressure
def simulation(startwerte):
    u = np.full((t.shape[0], x.shape[0], 3), 0.0)
    x0 = np.array([startwerte(i) for i in x])
    print(u.shape)
    u[0, :] = x0
    u[:, 0, :] = x0[0]
    u[:,-1, :] = x0[-1]

    # %% Rechnen
    for i in range(1, u.shape[0]):

        for j in range(1, u.shape[1]-1):
            # p, pressur
            rho_plus1 = u[i - 1, j + 1, 0]
            rho_minus1 = u[i - 1, j - 1, 0]
            rho = u[i - 1, j , 0]
            pressure = u[i - 1, j , 1]
            pressure_plus1    = u[i - 1, j + 1, 1]
            pressure_minus1   = u[i - 1, j - 1, 1]
            neuesrho     =  (rho_plus1 + rho_minus1)/2 - dt/(2 * dx) * (f1(rho_plus1, pressure_plus1) - f1(rho_minus1, pressure_minus1))
            neuesPressure = (pressure_plus1 + pressure_minus1)/2 - dt/(2 * dx) * (f2(rho_plus1, pressure_plus1) - f2(rho_minus1, pressure_minus1))

            u[i, j, 0] = neuesrho
            u[i, j, 1] = neuesPressure
            u[i, j, 2] = neuesPressure / neuesrho + h(neuesrho)
    return u

def visualisieren(u, title = "standard Title", beideTeile = True):
    if beideTeile:
        fig, (ax1, ax2) = plt.subplots(2,1)
    else:
        fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(4, 7)
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(title)
    ax1.set_title("rho")
    im1 = ax1.imshow(u[:,:,0], origin = "lower", extent=[xminusLimit, XLimit, 0,TLimit])
    plt.colorbar(im1, ax = ax1)
    ax1.set_ylabel('t')
    ax1.set_xlabel('x')

    if beideTeile:
        ax2.set_title("v")
        ax2.set_ylabel('t')
        ax2.set_xlabel('x')
        im2 = ax2.imshow(u[:,:,2], origin = "lower", extent=[xminusLimit,XLimit, 0,TLimit])
        plt.colorbar(im2, ax = ax2)
    plt.savefig(dir_path + "/" + title + "ARZSIM.png")
    fig.show()

def doCalc(rholinks = 0.4,
           vlinks   = 0.2,
           rhorechts =  0.8,
           vrechts   = 0.5,
           title = "Visualisierung "):

    def startwerteFunktion(z):
        if z < 0.5:
            return np.array([rholinks,  (vlinks  - h(rholinks)) * rholinks, vlinks])
        else:
            return np.array([rhorechts, (vrechts - h(rhorechts)) * rhorechts, vrechts])

    u = simulation(startwerte=startwerteFunktion)
    visualisieren(u, title = title)

doCalc(rholinks = 0.2, vlinks   = 0.3, rhorechts =  0.8, vrechts   = 0.4, title = "Visualisierung des Riemannproblems")

