import numpy as np
import matplotlib.pyplot as plt
import os

def plot_percentage_burned(validation_data_path='datasets/processed/validation_cumulative.npy', generated_data_path='datasets/generated'):
    cumul = np.load(validation_data_path)
    total_amount_cells = np.size(cumul[0])

    diffs = []
    burned_val = []
    burned_sim = []
    val_day_count = 0
    filenames = sorted(os.listdir(generated_data_path))
    for i in range(0, len(filenames), 72):
        sim_day = np.load(generated_data_path+'/'+filenames[i])
        
        sim_day[sim_day != 8] = 0
        sim_day[sim_day == 8] = 1
        
        val_day = cumul[val_day_count]
        diffs.append(abs(np.sum(sim_day - val_day)))
        burned_sim.append((np.sum(sim_day)/235571)*100)
        burned_val.append((np.sum(val_day)/235571)*100)
        
        val_day_count += 1

        # break the loop if the validation data is all used
        if val_day_count == len(cumul):
            break

    plt.plot(burned_sim, label='Simulated data', color='firebrick')
    plt.plot(burned_val, label='Validation data', color='green')
    plt.title('Total number of tiles burned')
    plt.xlabel('Days')
    plt.ylabel('Percentage of tiles burned')
    plt.legend()
    plt.show()
