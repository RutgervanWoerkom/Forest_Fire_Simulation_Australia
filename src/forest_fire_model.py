from src.helpfiles import *
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import clear_output


class ForestFires:
    def __init__(self, p_spread=0.6, p_orange=0.2, p_red=0.1, p_black=0.1):
        
        # map dimensions and pixel size
        self.height = None
        self.width = None
        self.tile_size = None

        # pixel level map data and features
        self.map_matrix = None
        self.vegetation_map = None
        self.temperature_map = None
        self.precipitation_map = None
        self.height_map = None
        self.validation_map = None

        # optional slice of the map
        self.map_slice = None

        # mask that cuts out water from plots
        self.mask = None

        # holds all starting coordinates of fires per day
        self.origin_points = None

        # initial probability of fire to spread
        self.p_spread = p_spread

        # probabilities of burning tiles to switch to next fire state
        self.p_orange = p_orange
        self.p_red = p_red
        self.p_black = p_black

        # probabilities of yellow and orande fires to go extinct
        self.p_yellow_out = None
        self.p_orange_out = None
        
        # neighbour coordinates and their corresponding miltipliers
        # (diagonal tiles are further away thus fire is less likely to 
        # spread to these tiles)
        self.neigh_dirs = np.array([[(-1,-1), 0.83], [(-1,0),1], [(-1,1), 0.83],
                                    [(0,-1),1], [(0,1),1],
                                    [(1,-1), 0.83], [(1,0),1], [(1,1), 0.83]])

        

        # multipliers for each environnemental feature
        self.height_multiplier = None
        self.temp_multiplier = None
        self.rain_multiplier = None
        self.wind = None

        # wind normalizer
        self.max_wind = None

        # init dictionary containing vegetation population
        self.veg_population = None

        self.prev_day = 0

    def set_vegetation_population(self, map_matrix):
        """ 
        Input:
        - map matrix: a numpy array with zero's, one's and two's representing
        water, land and vegetation respectively.
        Output:
        - veg_population: a dictionary containing class Vegetation objects as values,
        with their x,y coordinates as keys.
        """

        # create local variables of often used variables for faster python lookup speed.
        map_mat = map_matrix
        rand = random.random

        # fill vegetation dict
        veg_population = {}
        for (y,x) in np.argwhere(map_mat == 2):
            # add vegetation that is on fire based on historical fire origin coordinates in the validation map
            if [y,x] in self.origin_points[0]:
                veg_population[(x,y)] = Vegetation(0, self.temperature_map[y,x], 
                                                        self.precipitation_map[y,x], 
                                                        self.height_map[y,x])
            # else add normal vegetation
            else:
                veg_population[(x,y)] = Vegetation(4, self.temperature_map[y,x], 
                                                        self.precipitation_map[y,x], 
                                                        self.height_map[y,x])
        return veg_population

        

    def update(self, day):

        # optional slice of the map.
        [y_min, y_max, x_min, x_max] = self.map_slice

        # refresh temperature, precipitation and validation map every new day.
        if day != self.prev_day:
            self.temperature_map = np.flip(temperature(day, self.width, self.height),0)[y_min:y_max, x_min:x_max]
            self.precipitation_map = np.flip(precipitation(day, self.width, self.height),0)[y_min:y_max, x_min:x_max]
            self.validation_map = np.flip(validation(day, self.width, self.height),0)[y_min:y_max, x_min:x_max]
  
        # create local variables of often used variables for faster python lookup speed.
        map_matrix = self.map_matrix
        temperature_map = self.temperature_map
        precipitation_map = self.precipitation_map
        height_map = self.height_map

        p_spread = self.p_spread
        p_orange = self.p_orange
        p_red = self.p_red
        p_black = self.p_black

        randnum = random.random

        neigh_dirs = self.neigh_dirs

        # normalize wind speed.
        wind_matrx = wind_matrix(self.wind[day][0], self.wind[day][1] / self.max_wind) 

        # copy the current vegetation population dictionary and create a temporary one that 
        # holds the updated states of the vegetation.
        current_population = self.veg_population.copy()
        temp_population = {}

        for (x,y) in current_population:

            # new day
            if day != self.prev_day:
                # update vegetation states to 'on fire' based on historical fire origin coordinates in the validation map.
                if [y,x] in self.origin_points[day]:
                    temp_population[(x,y)] = Vegetation(0, self.temperature_map[y,x], 
                                                        self.precipitation_map[y,x], 
                                                        self.height_map[y,x])
            
            # create local variables of map current tile features for faster python lookup speed.
            tempr = temperature_map[y,x]
            prec = precipitation_map[y,x]
            height = height_map[y,x]
            
            # if current tile is vegetation
            if current_population[(x,y)].status == 4:
                
                # iterate over neighbours
                for ((dy, dx), distance_multiplier) in neigh_dirs:

                    # neighbour coordinates on map
                    y1 = y + dy
                    x1 = x + dx

                    # if neighbour is on fire
                    if (x1,y1) in current_population and current_population[(x1,y1)].status == 0:
                        neighbour = current_population[(x1,y1)]
                        
                        # calculate influence of environnemental factors.
                        t_influence = self.temp_multiplier[round(int(tempr))]
                        p_influence = self.rain_multiplier[round(int(prec))]
                        h_influence = self.height_multiplier[round(int(height))] * \
                            rel_height_multiplier(neighbour.height, height, self.tile_size)
                        w_influence = wind_matrx[dy+1,dx+1]
                        
                        env_influences = t_influence * p_influence * h_influence * w_influence

                        # if the environnemental factors and spread probability are favourable enough
                        # fire can spread the the current tile with a certain probability.
                        if randnum() < (env_influences * distance_multiplier) * p_spread:
                            temp_population[(x,y)] = Vegetation(0, tempr, prec, height)  
                            continue
                        
            # if vegetation fire state is yellow.
            elif current_population[(x,y)].status == 0:

                # probability of fire to go extinct
                if randnum() < self.p_yellow_out[round(int(prec))]:

                    # vegetation state becomes moderately affected
                    temp_population[(x,y)] = Vegetation(5, tempr, prec, height)

                # probability of fire to advance to orange state
                elif randnum() < p_orange:
                    temp_population[(x,y)] = Vegetation(1, tempr, prec, height)
                    
            # if vegetation fire state is orange.
            elif current_population[(x,y)].status == 1:

                # probability of fire to go extinct
                if randnum() < self.p_orange_out[round(int(prec))]:

                    # vegetation state becomes strongly affected
                    temp_population[(x,y)] = Vegetation(6, tempr, prec, height)

                # probability of fire to advance to red state
                elif randnum() < p_red:
                    temp_population[(x,y)] = Vegetation(2, tempr, prec, height)

            # if vegetation fire state is red.
            elif current_population[(x,y)].status == 2:

                # probability of fire to advance to black state
                if randnum() < p_black:
                    # vegetation is destroyed
                    temp_population[(x,y)] = Vegetation(3, tempr, prec, height)
            
        # apply changes to the vegetation dictionary
        for (x,y) in temp_population:
            current_population[(x,y)] = temp_population[(x,y)]
        self.veg_population = current_population.copy()

        # set previous day
        self.prev_day = day
        
    def visualize(self, day, hour, minute, 
                    plot=True, render=True, save_as_array=True):

        # create local variables of current population for faster python lookup speed.
        current_population = self.veg_population.copy()

        # create easy accessible format for day, hour, minute.
        datetime = [str('0' + str(day) if len(str(day)) < 2 else day),
                    str('0' + str(hour) if len(str(hour)) < 2 else str(hour)),
                    str('0' + str(minute) if len(str(minute)) < 2 else str(minute))]
        
        current_map = self.map_matrix.copy()
        # update burned_ratio with gradations if they are present in matrix.
        for (x,y) in current_population:

            # create local variable state for easier readability and faster python lookup speed.
            state = current_population[(x,y)].status

            # assign different values in an array for seperability and correct color display in plot

            # yellow fire
            if state == 0:
                current_map[y,x] = 3
                
            # orange fire
            elif state == 1:
                current_map[y,x] = 4
                
            # red fire
            elif state == 2:
                current_map[y,x] = 5
                
            # black ash
            elif state == 3:
                current_map[y,x] = 8
            
            # moderately affected vegetaion
            elif state == 5:
                current_map[y,x] = 6
                
            # strongly affected vegetation
            elif state == 6:
                current_map[y,x] = 7      
                
        if render:
            # create an image to be redered
            image = np.empty((np.shape(current_map)[0],np.shape(current_map)[1],3),dtype=object)
            image = image.tolist()     

            # assign rgb values   
            for y in range(len(current_map)):
                for x in range(len(current_map[0])):
                    if current_map[y,x] == 0:
                        image[y][x] = [255,0,0]
                    elif current_map[y,x] == 1:
                        image[y][x] = [192,192,192]
                    elif current_map[y,x] == 2:
                        image[y][x] = [0,153,0]
                    elif current_map[y,x] == 3:
                        image[y][x] = [0,255,255]
                    elif current_map[y,x] == 4:
                        image[y][x] = [0,128,255]
                    elif current_map[y,x] == 5:
                        image[y][x] = [0,0,255]
                    elif current_map[y,x] == 6:
                        image[y][x] = [0,204,0]
                    elif current_map[y,x] == 7:
                        image[y][x] = [0,255,0]
                    elif current_map[y,x] == 8:
                        image[y][x] = [0,0,0]

            image = np.array(image) 

            # image properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (320,500)
            fontScale = .4
            fontColor = (255,255,255)
            lineType = 1
            text = f"Day: {datetime[0]} Time: {datetime[1]}:{datetime[2]}"
            
            # apply image properties and save image
            image = cv2.putText(image, 
                                text, 
                                org, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)
            
            cv2.imwrite(f"datasets/rendered/{datetime[0]}-{datetime[1]}-{datetime[2]}.png", image)

        
        if render or plot:
            # custom colormap 1
            colorMap = ['midnightblue', 'lightgrey', 'darkgreen', 'yellow', 'orange', 'red',  'olivedrab', 'darkolivegreen', 'black']
            values, colors = list(zip(*[(i, c) for i,c in enumerate(colorMap) if i in current_map]))

            norm = plt.Normalize(min(values), max(values))
            tuples = list(zip(map(norm, values), colors))
            cmap = LinearSegmentedColormap.from_list("", tuples)

            # custom colormap 2
            colors2 = ('lightgrey', 'paleturquoise', 'deepskyblue',  'mediumblue')
            values2 = (0, 20, 80, 200)

            norm2 = mpl.colors.Normalize(min(values2), max(values2))
            tuples2 = list(zip(map(norm2, values2), colors2))
            cmap2 = LinearSegmentedColormap.from_list("", tuples2)

            # custom colormap 3
            colors3 = ('green', 'black')
            bounds3 = np.array([0, 2])
            values3 = (0, 2)
            bound_norm3 = mpl.colors.BoundaryNorm(boundaries=bounds3, ncolors=2)

            norm3 = mpl.colors.Normalize(min(values3), max(values3))
            tuples3 = list(zip(map(norm3, values3), colors3))
            cmap3 = LinearSegmentedColormap.from_list("", tuples3)

            # create figure
            fig = plt.figure(constrained_layout=False)

            # image properties
            plt.title(f"Day = {datetime[0]}\nTime = {datetime[1]}:{datetime[2]}")
            fig_grid = fig.add_gridspec(20, 28, wspace=5.0, hspace=0.0)
            plt.axis('off')     

            # remove water from all maps by applying a mask
            temp_map = np.ma.masked_where(self.mask == 0, self.temperature_map)
            prec_map = np.ma.masked_where(self.mask == 0, self.precipitation_map)
            height_map = np.ma.masked_where(self.mask == 0, self.height_map)
            val_map = np.ma.masked_where(self.mask == 0, self.validation_map)

            # add subplots 

            # insert map of simulation into figure subplot
            ax1 = fig.add_subplot(fig_grid[:, :-14])
            simulation_plot = ax1.imshow(current_map, cmap=cmap, norm=norm, interpolation='none')
            ax1.set_title(f'Simulation')
            ax1.set_axis_off()

            # insert temperature map into figure subplot
            ax2 = fig.add_subplot(fig_grid[2:10, -13:-7])
            temperature_plot = ax2.imshow(temp_map, cmap='inferno', interpolation='none')

            ax2.set_title(f'Temperature')
            ax2.set_axis_off()
            cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=60), cmap='inferno'), ax=ax2, shrink=0.7)
            cbar2.set_label("°C", fontsize=5)
            cbar2.ax.tick_params(labelsize=5)

            # insert rainfall map into figure subplot
            ax3 = fig.add_subplot(fig_grid[11:-1, -13:-7])
            rain_plot = ax3.imshow(prec_map, cmap=cmap2, interpolation='none')
            ax3.set_title(f'Rainfall')
            ax3.set_axis_off()
            cbar3 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=ax3, shrink=0.7)
            cbar3.set_label("mm", fontsize=5)
            cbar3.ax.tick_params(labelsize=5)

            # insert height map into figure subplot
            ax4 = fig.add_subplot(fig_grid[11:-1, -6:])
            height_plot = ax4.imshow(height_map, cmap='YlOrBr', interpolation='none')
            ax4.set_title(f'Height')
            ax4.set_axis_off()
            cbar4 = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=2228), cmap='YlOrBr'), ax=ax4, shrink=0.7)
            cbar4.set_label("m", fontsize=5)
            cbar4.ax.tick_params(labelsize=5)

            # insert validation map into figure subplot
            ax5 = fig.add_subplot(fig_grid[2:10, -6:])
            validation_plot = ax5.imshow(val_map, cmap=cmap3, interpolation='none') 
            ax5.set_title(f'Validation')
            ax5.set_axis_off()
            cbar5 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm3, cmap=cmap3), ax=ax5, spacing='uniform', shrink=0.7)
            cbar5.set_label("unburned / burned", fontsize=5)
            cbar5.set_ticks([])

            # insert wind direction and speed into figure subplot
            ax_wind = fig.add_subplot(fig_grid[2:4, :2])
            ax_wind.arrow(0, 0, deg_to_dir(self.wind[day][0])[0], deg_to_dir(self.wind[day][0])[1], head_width=0.1, head_length=0.2, fc='k', ec='k', )
            ax_wind.set_title(f'{self.wind[day][1]} km/h', fontsize=5, y=-.5, loc='left',fontweight="bold")
            ax_wind.set_axis_off()
            plt.xlim(1.2, -1.2)
            plt.ylim(-1.2, 1.2)

        # optionally save rendered image as .png file
        if render:
            plt.savefig(f'datasets/rendered/{datetime[0]}-{datetime[1]}-{datetime[2]}.png', bbox_inches='tight', dpi=300)

        # optionally show rendered image 
        if plot:
            plt.show()
            clear_output(wait=True)
            
        # optionally save rendered image as .npy file
        if save_as_array:
            np.save(f'datasets/generated/{datetime[0]}-{datetime[1]}-{datetime[2]}', current_map)
                      

    
        
class Vegetation:
    def __init__(self, status, height, temperature, precipitation):
    
        # vegetation properties
        self.status = status
        self.temperature = temperature
        self.precipitation = precipitation
        self.height = height


# function to run the model
def forest_fire_simulation(days, map_slice=None, plot=True, render=True, 
                            save_as_array=False, 
                            p_spread=0.6, p_orange=.2, p_red=.1, p_black=.1, 
                            height_weight=1, temp_weight=2, rain_weight=2):
    
    # the width and height are hardcoded as it greatly influences the speed at which fire travels
    # and we found these values to deliver the best quality/performance.
    width = 1279
    height = 1023

    # default slice
    if not map_slice:
        map_slice = [y_min, y_max, x_min, x_max] = (0, height, 0, width)
    # custom slice
    else:
        map_slice = [y_min, y_max, x_min, x_max] = map_slice 


    forest_sim = ForestFires(p_orange=p_orange, p_red=p_red, p_black=p_black)
    
    # map dimensions and pixel size
    forest_sim.width = width
    forest_sim.height = height
    forest_sim.tile_size = 3600

    # data used to load all maps
    data = np.flip(np.genfromtxt("datasets/raw/veg/vegetation.grid",
                         skip_header=6, skip_footer=18), 0)

    # pixel level map data and features
    forest_sim.map_matrix = construct_density_map(data, width, height)[y_min:y_max, x_min:x_max]

    forest_sim.map_slice = map_slice        

    # holds all starting coordinates of fires per day
    forest_sim.origin_points = np.load('datasets/processed/starting_points.npy', allow_pickle=True).tolist()


    # initial (day 0) map features
    forest_sim.temperature_map = np.flip(temperature(0, width, height),0)[y_min:y_max, x_min:x_max]
    forest_sim.precipitation_map = np.flip(precipitation(0, width, height),0)[y_min:y_max, x_min:x_max]
    forest_sim.height_map = np.flip(construct_height(width, height),0)[y_min:y_max, x_min:x_max] * (2228 / 255)
    forest_sim.validation_map = np.flip(validation(0, width, height),0)[y_min:y_max, x_min:x_max]
    forest_sim.vegetation_map = np.flip(np.genfromtxt("datasets/raw/veg/vegetation.grid", 
                                       skip_header=6, skip_footer=18), 0)

    # initial probability of fire to spread
    forest_sim.p_spread = p_spread

    # probabilities of yellow and orande fires to go extinct
    forest_sim.p_yellow_out = np.concatenate((np.zeros((20)), np.linspace(0,80,81) / 80))
    forest_sim.p_orange_out = np.concatenate((np.zeros((50)), np.linspace(0,50,51) / 50))

    # multipliers for each environnemental feature
    forest_sim.height_multiplier = np.flip(np.linspace(0,2134,2135)) / 2134
    forest_sim.temp_multiplier = np.linspace(0,60,61) / 60
    forest_sim.rain_multiplier = np.flip(np.linspace(0,100,101)) / 100
    forest_sim.wind = np.load("datasets/processed/daily_wind.npy")
        
    # wind normalizer value
    forest_sim.max_wind = np.max(forest_sim.wind)

    # mask that cuts out water from plots
    forest_sim.mask = construct_mask(forest_sim.vegetation_map, width, height)[y_min:y_max, x_min:x_max]

    # init dictionary containing vegetation population
    forest_sim.veg_population = forest_sim.set_vegetation_population(forest_sim.map_matrix)    

    # run simulation for a given amount of days
    for day in range(days):
        for hour in range(24):
            for minute in [0,20,40]:
                forest_sim.update(day)
                forest_sim.visualize(day, hour, minute, plot=plot, render=render, save_as_array=save_as_array)

