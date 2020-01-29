import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from IPython.display import clear_output
from matplotlib.pyplot import figure

def construct_mask(data, dim_x, dim_y):
    """
    Constructs a binary mask image from vegatation data for specified output dimensions
    """    
    # Scale to a binary image
    data = np.where(data > 1, data, 1)
    data = np.where(data < 2, data, 0)
    data = data.astype(int)
    
    data = np.flip(data, 0)
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    return data

def construct_vegetation(data, dim_x, dim_y):
    """
    Constructs a vegetation image for specified dimensions and optionally save them
    Normalizes vegatation data to 0.0:1.0
    Sets ocean to -1
    Original ratio: 672, 841 -->  +/- 1:1.25
    """
    # Normalize vegetation
    min_value = np.min(data)
    max_value = np.unique(data)[-2]
    
    data = data - min_value
    data = data / (max_value - min_value)
    
    # Set ocean values to -1
    data = np.where(data < 4, data, -1)
    
    data = np.flip(data, 0)
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    return data

def construct_height(dim_x, dim_y):
    """
    Constructs a heightmap for specified dimensions 
    Height scaled between 0 and 255
    The actual height is scaled between 0 and 2228
    """
    data = cv2.imread("datasets/raw/height/australia_heightmap.jpg", cv2.COLOR_BGR2GRAY)[:,:,0]
    data = np.flip(data, 0)
    
    # rescale bottom top
    data = data[60:-106,:]

    # add ocean to sides
    shape = np.shape(data)
    data = np.concatenate((np.zeros((shape[0], 41)), data, np.zeros((shape[0], 12))), 1)
    
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    return data

def construct_precipitation(data, dim_x, dim_y):
    """
    Constructs a precipitation image in mm for specified dimensions with an optional mask image.
    """
    data = np.flip(data, 0)
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    return data

def construct_temperature(data, dim_x, dim_y, round=False):
    """
    """
    data = np.flip(data, 0)
    
    # remove right ocean
    data = data[:, :-45]
    
    # remove bottom ocean
    data = data[14:-10]    
    
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    
    # Convert floats to int
    data = data.astype(int) if round is True else data  
    return data

def select_timeframe(dataframe, start, end):
    """
    Select a start and an end date (int) of the 92 days of available data
    """
    assert end - start > 0    
    date_range = dataframe.acq_date.unique()[start : end]
    
    return dataframe.loc[dataframe['acq_date'].isin(date_range)].reset_index()

def normalize_coordinates(input_data, x_scale, y_scale):
    """
    Scales fire-data provided by NASA in MODIS extention to dataframe in specified dimensions
    """
    # Extract extrema
    lat = input_data.latitude
    lon = input_data.longitude
    
    raw_data = pd.read_csv("datasets/raw/fire/fire_nrt_V1_95405.csv")

    max_lat = max(raw_data.latitude)
    min_lat = min(raw_data.latitude)
    max_lon = max(raw_data.longitude)
    min_lon = min(raw_data.longitude)

    # Normalize coordinates
    df = pd.DataFrame()
    df['lat'] = (lat - min_lat) / (max_lat - min_lat)
    df['lon'] = (lon - min_lon) / (max_lon - min_lon)

    df['lon'] = df.lon * (x_scale - 1)
    df['lat'] = df.lat * (y_scale - 1)

    # Round
    df['lon'] = round(df.lon)
    df['lat'] = round(df.lat)
    return df

def construct_heatmap(df, dim_x, dim_y):
    """
    Plot the normalized coordinates on a heatmap for specified dimension
    """    
    im = np.zeros((dim_y, dim_x))
    heat_range = len(df)
    
    # Add every line in dataframe to the plot no duplicates
    for i in range(heat_range):
        if im[int(df.lat[i]), int(df.lon[i])] == 0:
            im[int(df.lat[i]), int(df.lon[i])] = 2    
    im = cv2.resize(im, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    return im

def scale_firemap(firemap, dim_x, dim_y, mask=False):  
    """
    Scale the constructed heatmap to the same croped layout as the mask 
    """
    # left 
    firemap = np.concatenate((np.zeros((dim_y, 45)), firemap), 1)
    # right
    firemap = np.concatenate((firemap, np.zeros((dim_y, 18))), 1)
    # top
    firemap = firemap[:-17]
    # bottom 
    shape = np.shape(firemap) 
    firemap = np.concatenate((np.zeros((10, shape[1])), firemap), 0)    
    
    firemap = cv2.resize(firemap, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)

    if mask is not False:
        firemap = np.ma.masked_where(mask == 0, firemap)
    return firemap

def construct_density_map(data, dim_x, dim_y):
    """
    Construct a trinary image with a vegetation density scaled to the actual vegetation in Australia
    0 = water
    1 = land (not flameble)
    2 = vegatation (flameble)
    """
    data = construct_vegetation(data, dim_x, dim_y)
    density_map = np.ones((dim_y, dim_x))
    
    # Add a vegetation pixel with a certain probability
    for row in range(dim_y):
        for col in range(dim_x):
            res = data[row, col]
            # Water
            if res == -1:
                density_map[row, col] = 0     
            else:
                # Land 
                if res < random.random():
                    density_map[row, col] = 1
                # Vegetation
                else:
                    density_map[row, col] = 2      
    return density_map

def get_data(folder, name):
    """Open data of the Australian BOM data of a specified folder"""
    return np.genfromtxt(folder + name, skip_header=6, skip_footer=18)

def temperature(day, dim_x, dim_y):
    folder = "datasets/raw/temp/"
    days = sorted(os.listdir(folder)) 
    data = get_data(folder, days[day])    
    return construct_temperature(data, dim_x, dim_y)

def precipitation(day, dim_x, dim_y):
    """Construct a precipition map for a specified day"""
    folder = "datasets/raw/rain/"
    days = sorted(os.listdir(folder))
    
    data = get_data(folder, days[day])
    return construct_precipitation(data, dim_x, dim_y)

def validation(day, dim_x, dim_y):
    """Construct a validation map for a specified day"""
    fire = pd.read_csv("datasets/raw/fire/fire_nrt_V1_95405.csv")
    mask_data = data = np.genfromtxt("datasets/raw/veg/vegetation.grid", skip_header=6,
                                     skip_footer=18)   
    timeframe_firemap=select_timeframe(fire, day, day+1)
    normalized_firemap = normalize_coordinates(timeframe_firemap, dim_x, dim_y)
    heatmap = construct_heatmap(normalized_firemap, dim_x, dim_y)
    scaled_heatmap = scale_firemap(heatmap, dim_x, dim_y, construct_mask(mask_data, dim_x, dim_y))
    return scaled_heatmap

def wind_matrix(degrees, multiplier=1):
    """This function takes in the degree and multiplier for a wind matrix and outputs 
    said 3x3 matrix normalized to its max value so that all values <= 1"""

    # allow for degrees > 360 to work
    degrees = degrees % 360
    
    # to ensure the computation of the shortest distance, the array is 
    # temporarily flipped if the degrees > 180
    mirror = False
    if degrees > 180:
        mirror = True
        degrees -= 180
    
    # compute differences between the given degrees and the degrees
    # corresponding to north, south, west, northwest, etc..
    n_diff = min(abs(360 - degrees), abs(360 - (degrees + 360))) / 180
    s_diff = min(abs(180 - degrees), abs(180 - (degrees + 360))) / 180
    w_diff = min(abs(270 - degrees), abs(270 - (degrees + 360))) / 180
    e_diff = min(abs(90  - degrees), abs(90  - (degrees + 360))) / 180
    
    nw_diff = min(abs(315 - degrees), abs(315 - (degrees + 360))) / 180
    sw_diff = min(abs(225 - degrees), abs(225 - (degrees + 360))) / 180
    se_diff = min(abs(135 - degrees), abs(135 - (degrees + 360))) / 180
    ne_diff = min(abs(45  - degrees), abs(45  - (degrees + 360))) / 180
    
    wind_matrix = np.zeros((3,3))
    wind_matrix[1, 1] = 1
    
    wind_matrix[0, 1] = (1 + (n_diff - .5)) ** multiplier
    wind_matrix[2, 1] = (1 + (s_diff - .5)) ** multiplier
    wind_matrix[1, 0] = (1 + (w_diff - .5)) ** multiplier
    wind_matrix[1, 2] = (1 + (e_diff - .5)) ** multiplier
    
    wind_matrix[0, 0] = (1 + (nw_diff - .5)) ** multiplier
    wind_matrix[2, 0] = (1 + (sw_diff - .5)) ** multiplier
    wind_matrix[2, 2] = (1 + (se_diff - .5)) ** multiplier
    wind_matrix[0, 2] = (1 + (ne_diff - .5)) ** multiplier
    
    # the temporary flip is undone
    if mirror:
        wind_matrix = np.flip(np.flip(wind_matrix, 0), 1)
        
    # return a normalized matrix by dividing with max value
    return wind_matrix / np.max(wind_matrix)

def rel_height_multiplier(height1, height2, tile_size):
    degree = np.arctan((height2-height1) / tile_size) / (2 *np.pi) * 360
    multiplier = np.array(((degree ** 3 * 4) / 1600000) + 0.8)
    multiplier[multiplier<0.4] = 0.4
    multiplier[multiplier>1] = 1
    return multiplier

def deg_to_dir(deg):
    """Converts degrees to a direction vector"""
    deg = deg + 90
    rad = (deg / 180) * np.pi
    return (np.cos(rad), np.sin(rad))

def renderToVideo(path, output_name, framerate):
    """Use ffmpeg to render a video to an mp4 format"""
    (
        ffmpeg
        .input(path, pattern_type='glob', framerate=framerate)
        .output(output_name)
        .run()
    )
