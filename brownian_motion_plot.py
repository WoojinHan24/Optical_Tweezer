import numpy as np
import matplotlib.pyplot as plt


def plot_brownian_raw_fig(
    t:np.float64, x : np.float64, y : np.float64
) -> plt.figure():
    raw_fig=plt.figure(figsize=(8,4))
    ax = raw_fig.subplots(1,2)

    ax[0].plot(x,y,'k-', linewidth = 1,)
    ax[0].set_xlabel('x [mm]')
    ax[0].set_ylabel('y [mm]')

    x0=x[0]
    y0=y[0]
    r =np.sqrt((x-x0)*(x-x0) +(y-y0)*(y-y0))
    ax[1].plot(t,r,'b')
    ax[1].set_xlabel('t [s]')
    ax[1].set_ylabel('r [mm]')
    
    raw_fig.tight_layout()
    return raw_fig

def plot_brownian_distribution_fig(
    delta_x:np.float64, delta_y:np.float64
)-> plt.figure():
    distribution_fig=plt.figure(figsize=(8,4))
    ax = distribution_fig.subplots(1,2)

    ax[0].hist(delta_x,bins=30)
    ax[0].set_xlabel('delta x')
    ax[0].set_ylabel('number')

    ax[1].hist(delta_y,bins=30)
    ax[1].set_xlabel('delta y')
    ax[1].set_ylabel('number')

    distribution_fig.tight_layout()
    return distribution_fig

def plot_modified_brownian_fig(
       delta_x_list:list, delta_y_list:list, tau:np.float64
)-> plt.figure():
    r_square=np.array([(delta_x_list[i]*delta_x_list[i]+delta_y_list[i]*delta_y_list[i]).mean() for i in range(0,len(tau))])
    r_square_error=np.array([(delta_x_list[i]*delta_x_list[i]+delta_y_list[i]*delta_y_list[i]).std() for i in range(0,len(tau))])
    modified_brownian_fig=plt.figure(figsize=(8,4))
    ax = modified_brownian_fig.subplots(1,2)

    ax[0].plot(tau,r_square,'k-',linewidth=1)
    ax[0].plot(tau,r_square+r_square_error,'k-',linewidth=1,alpha=0.2)
    ax[0].plot(tau,r_square-r_square_error,'k-',linewidth=1,alpha=0.2)
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('$r^2$ [$mm^2$]')

    ax[1].plot(tau,r_square_error,'k-',linewidth=1)


    return modified_brownian_fig


def get_brownian_data(
    file_name : str, delimiter='\t'
) -> tuple[np.ndarray[np.float64],np.ndarray[np.float64], np.ndarray[np.float64]] | None:
    input=np.loadtxt(file_name, dtype=np.float64,delimiter=delimiter)
    t,x,y = input.T
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

for index in range(0,6):
    file_name = f"./raw_data/2um_brownian_motion_{index}.txt"
    (t,x,y) = get_brownian_data(file_name)
    (delta_x_list, delta_y_list, tau) = get_brownian_ensemble(t,x,y)

    # raw_fig = plot_brownian_raw_fig(t,x,y)
    # raw_fig.savefig(f"./results/2um_brownian_motion_raw_fig_{index}.png")
    # print(is_drift(t,x,y))

    # distribution_fig=plot_brownian_distribution_fig(delta_x_list[1],delta_y_list[1])
    # distribution_fig.savefig(f"./results/2um_brownian_distribution_fig_{index}.png")

    modified_brownian_fig=plot_modified_brownian_fig(delta_x_list, delta_y_list, tau)
    modified_brownian_fig.savefig(f"./results/2um_modified_brownian_fig_{index}.png")

