import streamlit as st

from ann_sol import data_plot, ann_plot
from get_function import get_architecture
import json
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Breast Cancer Prediction',
                   page_icon=':chart_with_upwards_trend:',
                   layout='wide',
                   initial_sidebar_state='expanded')

################################ Main dashboard ################################

st.title('Web-based restarting-ANN for early diagnose breast cancer')
st.markdown('<style>div.block-container{padding-top:20px;}</style>', unsafe_allow_html=True)
st.markdown("## Pennes's Bio-heat Equation in steady-state condition: ")
latext = r'''
$$ 
\frac{1}{r} \frac{\partial}{\partial r} \left(kr \frac{\partial T}{\partial r} \right) + \
\frac{1}{r^2} \frac{\partial}{\partial \Theta} \left(k \frac{\partial T}{\partial \Theta} \right) + \
\rho\omega c(T-T_a)+q=0
$$ 
'''

st.write(latext)

st.markdown('### PDE Parameters Metrics')
init_main_col1, init_main_col2, init_main_col3, init_main_col4  = st.columns(4)
init_main_col1.markdown('#### Parameters')
init_main_col1.write(r'$k \left(\frac{w}{mK}\right)$')
init_main_col1.write(r'$T_a \left(\degree C \right)$')
init_main_col1.write(r'$\rho \left(\frac{kg}{m^3}\right)$')
init_main_col1.write(r'$c \left(\frac{J}{kg}\right)$')
init_main_col1.write(r'$\omega \left(\frac{1}{s}\right)$')
init_main_col1.write(r'$\dot{q} \left(\frac{W}{m^3}\right)$')


init_main_col2.markdown('#### Details')
init_main_col2.write('Coefficient of thermal conductivity')
init_main_col2.write('Arterial temperature')
init_main_col2.write('Density of the breast model')
init_main_col2.write('Specific heat of the material')
init_main_col2.write('Blood perfusion rate')
init_main_col2.write('Generated energy per unit volume')

init_main_col3.markdown('#### Normal')
init_main_col3.write(0.42)
init_main_col3.write(37)
init_main_col3.write(920)
init_main_col3.write(3000)
init_main_col3.write(0.00018)
init_main_col3.write(450)


init_main_col4.markdown('#### Cancerous')
init_main_col4.write(0.56)
init_main_col4.write(37)
init_main_col4.write(920)
init_main_col4.write(3000)
init_main_col4.write('`-`')
init_main_col4.write('`-`')

st.markdown('### Domain values')
dom_col1, dom_col2, dom_col3 = st.columns(3)
r_max = dom_col1.number_input('Maximum radius $r$', value=2, placeholder="Type a number...")
N_b = dom_col2.number_input('Boundary sample size (%)', value=50, placeholder="Type a number...")
N_u = dom_col3.number_input('Colocation sample size (%)', value=50, placeholder="Type a number...")

parameters_value = {
    "r_max": r_max,
    "N_b": N_b,
    "N_u": N_u
}

file_path = "parameters_value.json"

with open(file_path, "w") as json_file:
    json.dump(parameters_value, json_file, indent=4)

st.markdown('### Data sample distribution')

ann_layers, ann_neurons, ann_iters, apply_restarting, ann_cycles, ann_optimizer, new_model = get_architecture()

data_plot()

st.markdown('### Build a model')
################ tab 1 ############################

ann_col1, ann_col2, ann_col3, ann_col4 = st.columns(4)
ann_layers = ann_col1.number_input('Enter the number of hidden layer', value=1, placeholder="Type a number...")
ann_neurons = ann_col2.number_input('Enter the number of neurons each layer', value=10, placeholder="Type a number...")
ann_iters = ann_col3.number_input('Enter the maximum number of iteration', value=10, placeholder="Type a number...")
ann_cycles = 1
apply_restarting = ann_col4.selectbox(
    'Apply Restarting?',
    ('Yes',
    'No'))

if apply_restarting=='Yes':
    ann_cycles = st.number_input('Enter the number of cycle', value=5, placeholder="Type a number...")
else:
    st.empty()

ann_optimizer = st.selectbox(
    'Select the optimizer',
    ('Adadelta',
    'Adagrad',
    'Adam',
    'Adamax',
    'Ftrl',
    'Nadam',
    'RMSprop',
    'SGD'))

_, _, _, _, build_btn_mid, _, _, _, _ = st.columns(9)
build_btn = build_btn_mid.button('Run Model')

if build_btn:
    
    new_model = 'Yes'
    ann_architecture = {
    "layers": ann_layers,
    "neurons": ann_neurons,
    "iterations": ann_iters,
    "apply_restarting": apply_restarting,
    "cycles": ann_cycles,
    "optimizer": ann_optimizer,
    'new_model': 'Yes'
    }
    file_path = "ann_architecture.json"
    with open(file_path, "w") as json_file:
        json.dump(ann_architecture, json_file, indent=4)
    ann_plot()