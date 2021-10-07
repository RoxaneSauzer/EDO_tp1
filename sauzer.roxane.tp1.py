import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#Exercice1

#modèle de Gompertz

y0=0.

def f(y,t):
    return y*np.log(10/y)



def euler(f ,t0, tf, n, y0):
#f est la fonction telle que y'=f(y,t), t0 et tf sont les bornes de l'intervalle, n est le nombre de points affichés)
    h = (tf-t0)/n
    y = y0
    t = t0
    Y = [y0]

    for k in range(n):
        y =  y + h*f(y,t)
        t = t + h
        Y.append(y)

    return Y


plt.plot(euler(f,0,10,10,0.1))
plt.show()
