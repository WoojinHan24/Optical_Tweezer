import numpy as np
import matplotlib.pyplot as plt
import brownian_motion as brn
import optical_tweezer as trz





for index in range(0,10):
    for size in [1, 2, 3]:    
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

        modified_brownian_fig=brn.plot_modified_brownian_fig(delta_x_list, delta_y_list, tau)
        modified_brownian_fig.savefig(f"./results/{size}um_modified_brownian_fig_{index}.png")





n=1.58/1.33 #relative refraction index
P=40*0.001 #power of lazer
w_0 = 0.000001 #laser profile
a = 0.000001 #beads radius
f = 0.000001 #microscope profile

optical_tweezer_intro_fig=trz.plot_trapping_force(n,P,w_0,a,f)
optical_tweezer_intro_fig.savefig(f"./results/Optical_tweezer_intro_fig.png")
