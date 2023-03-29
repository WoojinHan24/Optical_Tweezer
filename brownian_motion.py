import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.constants as constants

def plot_brownian_raw_fig(
    t:np.float64, x : np.float64, y : np.float64
) -> plt.figure():
    raw_fig=plt.figure(figsize=(8,4))
    ax = raw_fig.subplots(1,2)

    ax[0].plot(x,y,'k-', linewidth = 1,)
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('y [m]')

    x0=x[0]
    y0=y[0]
    r =np.sqrt((x-x0)*(x-x0) +(y-y0)*(y-y0))
    ax[1].plot(t,r,'b')
    ax[1].set_xlabel('t [s]')
    ax[1].set_ylabel('r [m]')
    
    raw_fig.tight_layout()
    return raw_fig

def plot_brownian_distribution_fig(
    delta_x:np.float64, delta_y:np.float64
)-> plt.figure():
    distribution_fig=plt.figure(figsize=(8,4))
    ax = distribution_fig.subplots(1,2)


    ax[0].hist(delta_x,bins=30)
    ax[0].set_xlabel('$\Delta$ x [m]')
    ax[0].set_ylabel('number')

    ax[1].hist(delta_y,bins=30)
    ax[1].set_xlabel('$\Delta$ y [m]')
    ax[1].set_ylabel('number')

    distribution_fig.tight_layout()
    return distribution_fig

def plot_brownian_qqplot(
    delta_x:np.float64, delta_y:np.float64
)-> plt.figure():
    qqplot_fig=plt.figure(figsize=(8,4))
    ax = qqplot_fig.subplots(1,2)
    stats.probplot(delta_x, dist=stats.norm, plot=ax[0])
    stats.probplot(delta_y, dist=stats.norm, plot=ax[1])

    ax[0].set_ylabel('$\Delta$ x [m]')
    ax[0].set_xlabel('theoretical quantities')

    ax[1].set_ylabel('$\Delta$ y [m]')
    ax[1].set_xlabel('theoretical quantities')

    qqplot_fig.tight_layout()
    return qqplot_fig



def plot_modified_brownian_fig(
       delta_x_list:list, delta_y_list:list, tau:np.float64
)-> tuple[plt.figure(),np.float64, np.float64]:
    r_square=np.array([(delta_x_list[i]*delta_x_list[i]+delta_y_list[i]*delta_y_list[i]).mean() for i in range(0,len(tau))])
    r_square_error=np.array([(delta_x_list[i]*delta_x_list[i]+delta_y_list[i]*delta_y_list[i]).std() for i in range(0,len(tau))])
    slope, intercept, _,_,std_error = stats.linregress( tau[:2000],r_square[:2000])
    modified_brownian_fig=plt.figure(figsize=(8,4))
    ax = modified_brownian_fig.subplots(1,2)


    ax[0].plot(tau,r_square,'k-',linewidth=1)
    ax[0].plot(tau,r_square+r_square_error,'k-',linewidth=1,alpha=0.2)
    ax[0].plot(tau,r_square-r_square_error,'k-',linewidth=1,alpha=0.2)
    ax[0].plot(tau,slope*tau+intercept, 'r-', linewidth=1.5)
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('$r^2$ [$m^2$]')

    ax[1].plot(tau,r_square_error,'k-',linewidth=1)
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('$\sigma_{r^2}$ [m^2]')
    modified_brownian_fig.tight_layout()
    return (modified_brownian_fig, slope, std_error)


def get_brownian_data(
    file_name : str, delimiter='\t'
) -> tuple[np.ndarray[np.float64],np.ndarray[np.float64], np.ndarray[np.float64]] | None:
    input=np.loadtxt(file_name, dtype=np.float64,delimiter=delimiter)
    t,x,y = input.T
    x=x/1000
    y=y/1000
    return [t,x,y]

def get_brownian_ensemble(
    t:np.float64, x : np.float64, y : np.float64
) -> tuple[list[np.float64],list[np.float64],np.float64]:
    delta_x_list=[]
    delta_y_list=[]
    tau=np.float64([]) #actually vector
    for index in range(1, len(t)):
        delta_x_list.append(x[index:]-x[:-index])
        delta_y_list.append(y[index:]-y[:-index])
        tau=np.append(tau,np.array([t[index]-t[0]]))
    
    return (delta_x_list,delta_y_list,tau)

def is_drift(
    t:np.float64, x : np.float64, y : np.float64, n=3 #this function determines whether is drifted for m+-n\sigma
) -> bool:
    x_drift_determinant=np.heaviside(x[1:]-x[:-1],0.5).sum()
    y_drift_determinant=np.heaviside(y[1:]-y[:-1],0.5).sum()
    mean_val=len(t)/2
    delta_val=np.sqrt(len(t)/4)*n
    if x_drift_determinant>mean_val+delta_val or x_drift_determinant<mean_val-delta_val:
        return True
    if y_drift_determinant>mean_val+delta_val or y_drift_determinant<mean_val-delta_val:
        return True
    
    return False

def get_viscosity(
    R,alpha,T=298
) -> np.float64:
    return 2*constants.k*T/(3*np.pi*R*alpha)