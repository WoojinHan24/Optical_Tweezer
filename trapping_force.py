import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.constants as constants

def get_velocity_data(
    file_name : str, delimiter='\t'
) -> tuple[np.ndarray[np.float64],np.ndarray[np.float64]] | None:
    input=np.loadtxt(file_name, dtype=np.float64,delimiter=delimiter)
    current,critical_velocity = input.T
    current=current/1000
    critical_velocity=critical_velocity/1000
    return [current,critical_velocity]

def plot_force_raw_fig(
    P:np.float64, F: np.float64
) -> plt.figure():
    raw_fig=plt.figure(figsize=(4,4))
    ax = raw_fig.add_subplot(1,1,1)
    ax.plot(P,F,'k.')
    slope, intercept, rvalue,_,std_error = stats.linregress(P,F)
    ax.plot(P, P*slope+intercept, 'r-', linewidth=1.0)
    ax.set_xlabel('laser power [W]')
    ax.set_ylabel('trapping force [N]')

    print(slope, std_error, intercept,rvalue**2)
    raw_fig.tight_layout()
    return raw_fig

