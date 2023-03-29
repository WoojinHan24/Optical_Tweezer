import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import scipy.integrate as integrate
import scipy.stats as stats

def get_coefficient(
    n: np.float64, incident_angle: np.float64
) -> list :
    refract_angle = np.arcsin(np.sin(incident_angle)/n) #Snell's law
    r_s = ((np.cos(incident_angle)-n*np.cos(refract_angle))/(np.cos(incident_angle)+n*np.cos(refract_angle)))
    r_p = ((np.cos(refract_angle)-n*np.cos(incident_angle)))/(np.cos(refract_angle)+n*np.cos(incident_angle))
    R_s = r_s*r_s
    R_p = r_p*r_p
    R = (R_s+R_p)/2

    return [R , 1-R]

def get_force_single_beam(
    n: np.float64, incident_angle: np.float64, P: np.float64, R:np.float64, T: np.float64
)->np.ndarray:
    refract_angle = np.arcsin(np.sin(incident_angle)/n)
    F_z = n*P/constants.c * (1+R*np.cos(2*incident_angle)-(T**2 *(np.cos(2*incident_angle-2*refract_angle)+R*np.cos(2*incident_angle)))/(1+R**2 +2*R*np.cos(2*refract_angle)))
    F_y = n*P/constants.c * (R*np.sin(2*incident_angle)-(T**2 *(np.sin(2*incident_angle-2*refract_angle)+R*np.sin(2*incident_angle)))/(1+R**2 +2*R*np.cos(2*refract_angle)))

    return np.array([F_y,F_z])

def get_gaussian_profile(
    r:np.float64,P:np.float64,w_0:np.float64
)->np.float64:
    #r is an distance from center of gaussian beam to incident beam
    dP = P /np.pi/w_0**2 * np.exp(-r**2/w_0**2)
    return dP

def get_dF_y(
    r, beta, x, n, P, a, w_0, f
)-> np.float64:
    dP=get_gaussian_profile(r,P,w_0)
    gamma = np.arctan(f/r)
    incident_angle = np.arcsin(x/a*np.sin(gamma))
    [R,T]=get_coefficient(n,incident_angle)
    dF = get_force_single_beam(n,incident_angle,dP,R,T)

    return dF[0]*np.cos(beta)

def get_trapping_force(
    x,n,P,a,w_0,f
)->np.float64:
    return 4*integrate.dblquad(lambda beta,r:get_dF_y(r,beta,x,n,P,a,w_0,f), 0, 5*w_0, 0, np.pi/2)[0]


def plot_trapping_force(
    n,P,w_0,a,f
) -> plt.plot:
    x=np.linspace(-2*a,2*a,50)

    fig=plt.figure(figsize=(8,4))
    ax=fig.add_subplot(1,1,1)
    force=[get_trapping_force(X,n,P,a,w_0,f) for X in x]
    ax.plot(x,force,'k', linewidth=1.2)
    ax.set_xlabel('focus position [m]')
    ax.set_ylabel('trapping force [N]')
    

    return fig

