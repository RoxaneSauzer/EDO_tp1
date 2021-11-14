import numpy as np
import matplotlib.pyplot as plt



#Exercice1
#1

def f(y,t):
    return y*np.log(10/y)

def euler(f,y0,t,n):
    y=[y0]
    T=np.arange(0,t,n)
    for k in range(len(T)-1):
        h=T[k+1]-T[k]
        p1=f(y[k],T[k])
        y.append(y[k]+p1*h)
    return y

def y(t,y0):
    return 10*(y0/10)**(np.exp(-t))

def plot_multiple_steps(g,x0,T,h1,h2,h3):
    y1=euler(g,x0,T,h1)
    y2=euler(g,x0,T,h2)
    y3=euler(g,x0,T,h3)

    plt.plot(np.arange(0,T,h1),y1)
    plt.plot(np.arange(0,T,h2),y2)
    plt.plot(np.arange(0,T,h3),y3)
    plt.show()

def plot_multiple_steps_with_solution(g,x,x0,T,h1,h2,h3):

    y1=euler(g,x0,T)
    y2=euler(g,x0,T)
    y3=euler(g,x0,T)

    t1=np.arange(0,T,h1)
    t2=np.arange(0,T,h2)
    t3=np.arange(0,T,h3)

    plt.plot(t1,y1)
    plt.plot(t1,x(t1,x0))
    plt.plot(t2,y2)
    plt.plot(t2,x(t2,x0))
    plt.plot(t3,y3)
    plt.plot(t3,x(t3,x0))
    plt.show()

#plot_multiple_steps(f,0.1,10,0.1,0.5,1)
#plot_multiple_steps_with_solution(f,y,0.1,10,0.1,0.5,1)

#2

def g(x,t):
    return x**2

def x(t,x0):
    return 1/((1/x0) -t)

#plot_multiple_steps(g,x,0.2,2,0.1,0.01,0.001)

#Exercice 2
#1
def h(y,t):
    return -y**2

def z(t,z0):
    return 1/((1/z0) +t)

#plot_multiple_steps(h,z,1,1.5,0.1,0.01,0.001)
#plot_multiple_steps(h,z,2,10,0.1,0.01,0.001)

def plot_multiple_y0(h,z0,z1,T,n):

    y1=euler(h,z0,T,n)
    y2=euler(h,z1,T,n)

    t=np.arange(0,T,n)
    plt.plot(t,y1)
    plt.plot(t,y2)
    plt.show()

def plot_multiple_y0_with_solution(h,z,z0,z1,T,n):
    t=np.arange(0,T,n)

    y1=euler(h,z0,t,n)
    y2=euler(h,z1,t,n)

    plt.plot(t,y1)
    plt.plot(t,y2)
    plt.plot(t,z(t,z0))
    plt.plot(t,z(t,z1))
    plt.show()

#plot_multiple_y0_with_solution(h,z,1,2,10,0.01)
#plot_multiple_y0_with_solution(h,z,1,2,1.5,0.01)


#plot_multiple_y0(h,1,2,1.5,0.01)

#2

def j(y,t):
    return - np.sqrt(np.abs(y))

def u(t,u0):
    return (np.sqrt(u0)-t/2)**2

def plot_mult_y0_and_mult_steps(j,u0,u1,T1,T2,n):

    y1=euler(j,u0,T1,n)
    y2=euler(j,u1,T2,n)

    t1=np.arange(0,T1,n)
    t2=np.arange(0,T2,n)

    plt.plot(t1,y1)
    plt.plot(t2,y2)
    plt.show()

#plot_mult_y0_and_mult_steps(j,1,2,2,2*np.sqrt(2),0.01)

def check_positive(f,y0,t,n):
    T=np.arange(0,t,n)
    for c in euler(f,y0,T):
        return c <=0

#print(check_positive(j,2,2*np.sqrt(2),0.01))

#Exercice3

def k(y,t):
    return y*(1+np.exp(-y))+np.exp(2)

def heun(f, x0, t, n):
    T=np.arange(0,t,n)
    x = [x0]
    for k in range(len(T)-1):
        h = T[k+1] - T[k]
        c1 = f(x[k], T[k])
        c2 = f(x[k] + h * c1, T[k+1])
        x.append(x[k] + h * (c1 + c2) / 2)
    return x

def rk4(f, x0, t,n):
    T=np.arange(0,t,n)
    x = [x0]
    for k in range(len(T)-1):
        h = T[k+1] - T[k]
        c1 = f(x[k], T[k])
        c2 = f(x[k] + h * c1 / 2, T[k] + h / 2)
        c3 = f(x[k] + h * c2 / 2, T[k] + h / 2)
        c4 = f(x[k] + h * c3, T[k+1])
        x.append(x[k] + h * (c1+2*c2+2*c3+c4) / 6)
    return x