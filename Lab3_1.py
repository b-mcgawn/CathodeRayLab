######################################################################
# Imports
import numpy as np, matplotlib.pyplot as plt, pandas as pd, matplotlib.gridspec as grids

# Data init
data = pd.read_csv("Thompson e_m - Brady_CSV.csv")

# fit a line to data
# plot E field as y, B field as x
# fit a line to that

d = 0.008 #m
k = 0.00417 # Teslas / Amp
V = data['Electric Field Voltage (V)']
I = data['Magnetic Field (A)']
E = np.copy(V)/d
B = np.copy(I) * k

V_std = data['error']
E_std = np.copy(V_std)/d
I_std = data['error.1']
B_std = np.copy(I_std)*k

#inds = np.where(E == np.nan)
#fixed CSV, but energy breaks were at 6,7 and 14,15

##################################################
# Try ODR (Orthogonal Distance Regression)
from scipy.odr import *

#initialize data
fitdata = RealData(B, E, sx = B_std, sy = E_std)

#define your function, initalize model
def f(m , x):
    return m*x

model = Model(f)

#initialize ODR
# intial slope came from polyfit
myODR = ODR(fitdata, model, beta0 = [28578530.26157819])

#run the fit
myoutput = myODR.run()

# Extract everything
velo = myoutput.beta
sd_velo = myoutput.sd_beta

######################################################################
#plot:
def plot(B=B, E=E, B_std=B_std, E_std=E_std,
         velo=velo, sd_velo=sd_velo, fignum=10, qmr = False):
    #format:
    fig = plt.figure(fignum)
    plt.ion()
    plt.clf()

    gs = grids.GridSpec(3,1)

    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[2,0])
    plt.subplots_adjust(hspace=0)

    ax1.ticklabel_format(axis='both', style='sci', scilimits= (1,-1))

    ##################################################
    # Top plot: 
    plotx = np.linspace(np.min(B),np.max(B),100)

    velo_UL = velo + sd_velo
    ploty = velo_UL * plotx
    ax1.plot(plotx, ploty, "-y", label = "$\pm 1 \sigma $")

    velo_LL = velo - sd_velo
    ploty = velo_LL * plotx
    ax1.plot(plotx, ploty, "-y")

    ploty = velo * plotx
    if(qmr):
        ax1.plot(plotx, ploty, "-b", label = r"Best-fit $\frac{q}{m}$")
    else: ax1.plot(plotx, ploty, "-r", label = "Best-fit velocity")

    ax1.errorbar(B, E, xerr = B_std, yerr = E_std, color = "k", fmt = ".")
    ax1.legend()

    if(qmr):ax1.set_title("Charge to Mass Ratio Fit")
    else: ax1.set_title("Velocity Fit")

    if(qmr): ax1.set_ylabel('Square of Electron Velocity ($m^2 / s^2$)')
    else: ax1.set_ylabel('Electric Field Strength (Coulombs)')
    ax1.set_xticklabels(labels='')

    ##################################################
    #plot residuals
    pred = velo*np.copy(B)
    resid = E - pred

    ax2.ticklabel_format(axis='both', style='sci', scilimits= (1,-1))

    ax2.axhline(0,color = "red")
    # !!!!! What are errors for residuals? 
    ax2.errorbar(B,resid,  xerr = B_std, yerr = E_std,color = "k",fmt = '.')

    if(qmr):ax2.set_xlabel('Electric Field Strength (Coulombs)')
    else:ax2.set_xlabel('Magnetic Field Strength (Teslas)')
    ax2.set_ylabel('Residuals')

plot()

######################################################################
# q / m

# q / m = (2y v^2) / (E x^2)
# B^2 * k * q_m = E

# q / m = (2y v^2) / (E x^2)

x = data['x (m)']
y = data['y']

k = x**2/(2*y)

#velo
v = E/B
v_std = v*(np.copy((E_std/E)+(B_std/B)))

v_2 = v**2
v_2_std = 2*v_std

#Voltage
En1 = 148.03
En2 = 258.3
En3 = 204.72
En_err = data['Energy Error: '][0]

En = np.ones(len(E))
En[:6] *= En1
En[6:12] *= En2
En[12:18] *= En3

En_sd = np.ones(len(E)) * En_err

#Convert to E field
Ef = En/d

Ef_sd = En_sd/d

fitdata_q = RealData(Ef, v_2, sx = Ef_sd, sy = v_2_std)

#define your function, initalize model
def f_q(m , x):
    return m*x*k[0]

model_q = Model(f_q)

#initialize ODR
# intial slope came from polyfit
q_mODR = ODR(fitdata_q, model_q, beta0 = [1.52e11])

#run the fit
myoutput = q_mODR.run()

# Extract everything
q_m = myoutput.beta
sd_q_m = myoutput.sd_beta

plot(B=Ef, E=v_2, B_std=Ef_sd, E_std=v_2_std, velo=k[0]*q_m, sd_velo=k[0]*sd_q_m, fignum=120, qmr = True)
