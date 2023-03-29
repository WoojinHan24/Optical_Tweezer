import numpy as np
import matplotlib.pyplot as plt
import brownian_motion as brn
import optical_tweezer as trz
import trapping_force as trp
import scipy.stats as stats
import warnings


warnings.filterwarnings(action='ignore')

for size in [1, 2, 3]:
    slopes=[]
    std_errs=[]
    for index in range(0,10):
        if size==1 and index in [1,3]:
            continue
        file_name = f"./raw_data/{size}um_brownian_motion_{index}.txt"
        try:
            (t,x,y) = brn.get_brownian_data(file_name)
        except (FileNotFoundError):
            continue

        (delta_x_list, delta_y_list, tau) = brn.get_brownian_ensemble(t,x,y)

        raw_fig = brn.plot_brownian_raw_fig(t,x,y)
        raw_fig.savefig(f"./results/{size}um_brownian_motion_raw_fig_{index}.png")
        print(brn.is_drift(t,x,y))

        distribution_fig=brn.plot_brownian_distribution_fig(delta_x_list[1],delta_y_list[1])
        distribution_fig.savefig(f"./results/{size}um_brownian_distribution_fig_{index}.png")

        qqplot_fig=brn.plot_brownian_qqplot(delta_x_list[1],delta_y_list[1])
        qqplot_fig.savefig(f"./results/{size}um_brownian_qqplot_fig_{index}.png")

        modified_brownian_fig, slope, std_err=brn.plot_modified_brownian_fig(delta_x_list, delta_y_list, tau)
        modified_brownian_fig.savefig(f"./results/{size}um_modified_brownian_fig_{index}.png")
        slopes.append(slope)
        std_errs.append(std_err)

    slope_ave=sum(slopes)/len(slopes)
    eta=brn.get_viscosity(size*1e-6,slope_ave)
    print(eta,eta/slope_ave*max(std_errs))


for size in [1, 2, 3]:
    if size==1 and index in [1,3]:
        continue
    file_name = f"./raw_data/{size}um_trapping_force.txt"
    try:
        (current, critical_velocity) = trp.get_velocity_data(file_name)
    except (FileNotFoundError):
        print(f"size{size} critical velocity not found")
        continue
    b = 1.58*1e-8 #drag force coefficient
    I_0 = 0.035 #Threshold current
    xi = 1 # Power-current ratio
    force = b*critical_velocity
    power = xi*(current-I_0)
    
    force_raw_plot=trp.plot_force_raw_fig(power,force)
    force_raw_plot.savefig(f"./results/{size}um_force_raw_plot.png")


    
    



size = 3
n0=1.58/1.33 #relative refraction index
P0=40*0.001/100**2 #power of lazer
w_0 = 5.6*1e-3/100#laser profile
a = size*1e-6 #beads radius
f = 1e-3 #microscope profile

optical_tweezer_intro_fig=trz.plot_trapping_force(n0,P0,w_0,a,f)
optical_tweezer_intro_fig.savefig(f"./results/Optical_tweezer_intro_fig.png")

n=np.linspace(1.1,1.3,13)
P=np.linspace(0,P0,10)

coef=[]
dk_dP=[]
for n_i in n:
    coef=[]
    for P_i in P:
        coef.append(trz.get_trapping_force(-0.5*a,n_i,P_i,a,w_0,f)/(0.5*a))
    
    coef_arr=np.array(coef)
    slope,_,_,_,_ = stats.linregress(P,coef_arr)
    dk_dP.append(slope)


slope,intercept,_,_,_ = stats.linregress(n,dk_dP)

final_fig = plt.figure(figsize = (4,4))
ax = final_fig.add_subplot(1,1,1)

ax.plot(n,dk_dP,'k.')
ax.plot(n,n*slope+intercept,'r-')
ax.set_xlabel('refraction index')
ax.set_ylabel(' $\gamma$ ')

print(slope,intercept)
final_fig.tight_layout()
final_fig.savefig("Final_figure.png")
