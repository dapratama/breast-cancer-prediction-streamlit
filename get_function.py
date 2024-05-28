import json
import time
import numpy as np
import streamlit as st
from keras.models import load_model, Sequential
from keras.layers import Dense


def get_model(new_model):
    global model
    if new_model=='Yes':
        with st.spinner('Loading model...'):
            time.sleep(2.5)
            model = load_model('new_model.h5')
    else:
        with st.spinner('Loading model...'):
            time.sleep(2.5)
            model = load_model('normal.h5')
    return model

def get_architecture():
    with open('ann_architecture.json', "r") as json_file:
        architecture = json.load(json_file)
    ann_layers = architecture["layers"]
    ann_neurons = architecture["neurons"]
    ann_iters = architecture["iterations"]
    apply_restarting = architecture['apply_restarting']
    ann_cycles = architecture["cycles"]
    ann_optimizer = architecture["optimizer"]
    new_model = architecture['new_model']
    return ann_layers, ann_neurons, ann_iters, apply_restarting, ann_cycles, ann_optimizer, new_model

def get_value():
    with open('parameters_value.json', "r") as json_file:
        parameters = json.load(json_file)
    r_max = parameters["r_max"]
    N_b = parameters["N_b"]
    N_u = parameters["N_u"]
    return r_max, N_b, N_u

def get_data():
    with open('get_data.json', "r") as json_file:
        parameters = json.load(json_file)

    x_star = np.array(parameters["x_star"])
    exact_u = np.array(parameters["exact_u"])
    x_u_train = np.array(parameters["x_u_train"])
    y_u_train = np.array(parameters["y_u_train"])
    x_b0_train = np.array(parameters["x_b0_train"])
    y_b0_train = np.array(parameters["y_b0_train"])
    x_b_train = np.array(parameters['x_b_train'])
    y_b_train = np.array(parameters['y_b_train'])
    mask1 = parameters['mask1']
    num_circle = parameters['num_circle']
    location = parameters['location']
    radius = parameters['radius']

    return x_star, exact_u, x_u_train, y_u_train, x_b0_train, y_b0_train, x_b_train, y_b_train, mask1, num_circle, location, radius

def build_model(num_layers, neurons_per_layer):
    #initializers = keras.initializers.RandomUniform(minval=0., maxval=100.)
    with st.spinner('Creating model...'):
        time.sleep(2.5)
        model = Sequential()

        # Add the input layer
        model.add(Dense(neurons_per_layer, activation='relu', input_shape=(3,)))

        # Add hidden layers
        for _ in range(num_layers): 
            model.add(Dense(neurons_per_layer, activation='relu'))

        # Add the output layer
        model.add(Dense(1))  # Adjust the activation function as needed


    return model