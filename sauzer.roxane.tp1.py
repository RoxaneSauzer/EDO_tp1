import numpy as np
import matplotlib.pyplot as plt



#Exercice1

#modèle de Gompertz



def f(y,t):
    return y*np.log(10/y)

def euler(F,tf,n, y0):
    h =tf/n
    y = y0
    t = 0
    Y = [y0]
    T = [0]
    for k in range(n):
        y = y + h*F(y,t)
        t=t+h
        Y.append(y)
        T.append(t)
    return T,Y

def y(t):
    return 10*(0.1/10)**(np.exp(-t))

print(euler(f,10,20,0.1))

t=np.linspace(0,10,10)
T,Y=euler(f,10,50,0.1)
plt.plot(T,Y)
plt.plot(t,y(t))
plt.show()



