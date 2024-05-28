import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow import keras


from data_generator import get_normal_data

from tensorflow import keras
from get_function import get_value, get_architecture, get_model, build_model, get_value, get_data

import warnings
warnings.filterwarnings('ignore')


class StreamlitProgressBarCallback(keras.callbacks.Callback):
    def __init__(self, max_epochs, current_cycle, max_cycles):
        super(StreamlitProgressBarCallback, self).__init__()
        _, _, _, _, ann_cycles, _, _ = get_architecture()
        self.ann_cycles = ann_cycles
        self.max_epochs = max_epochs
        self.max_cycles = max_cycles
        self.current_cycle = current_cycle
        self.start_time = None
        self.loss_values = []
        self.prediction_plots = []
        self.training_completed = False
        self.progress_bar = None
        self.info_text = None
        self.info_cycle = None

    def on_train_begin(self, logs=None):
        self.progress_bar = st.progress(0)
        self.info_text = st.text('Training process begin. Please wait..')
        self.epochs_completed = 0
        self.loss_figure = st.empty()
        self.result_figure = st.empty()

    def on_epoch_end(self, epoch, logs=None):


        self.epochs_completed += 1
        progress_percent = self.epochs_completed / self.max_epochs
        self.progress_bar.progress(progress_percent)

        if logs is not None:
            loss = logs.get('loss')
            self.loss_values.append(loss)
            info = f'Iteration: {self.epochs_completed}/{self.max_epochs} - Completion: {int(progress_percent * 100)}% - Loss: {loss}'
            self.info_text.text(info)

            if self.training_completed:
                # Clear the figures and return to avoid plotting
                self.loss_figure.empty()
                self.result_figure.empty()
                self.info_text.empty()
                return

            self.plot_loss_graph()

    def on_train_end(self, logs=None):
        self.training_completed = True
        # Clear the loss figure
        self.loss_figure.empty()
        self.progress_bar.empty()
        self.info_text.empty()

    def plot_loss_graph(self):
        
        r_max, _, _ = get_value()

        if self.loss_values:
            fig = plt.figure(figsize=(10, 7))

            x_star = get_normal_data()

            ax2 = fig.add_subplot(122)
            
            # loss plot figure
            epochs = list(range(1, len(self.loss_values) + 1))
            ax2.plot(epochs, self.loss_values)
            ax2.grid(True)
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0*2.5, box.width*0.8, box.height*0.5])
            ax2.set_title('Loss function results')
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel('MSE')

            # prediction plot figure
            if self.model is not None:
                # Generate predictions
                try:
                    predictions = self.model.predict(x_star)
                    self.prediction_plots.append(predictions)
                except BrokenPipeError as bpe:
                    st.error("A BrokenPipeError occurred. Likely due to client disconnection.")
                    st.error(f"Details: {bpe}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    raise

                if len(self.prediction_plots) > 1:
                    nr, nt = (100, 100)   # means that in the domain there are 100.000 data points
                    zeniths = np.linspace(0, r_max, nr)
                    azimuths = np.radians(np.linspace(0, 180, nt))

                    r, phi = np.meshgrid(zeniths, azimuths)
                    last_predictions = self.prediction_plots[-1]
                    values = last_predictions.reshape((nr, nt))
                    ax1 = fig.add_subplot(121, projection='polar')
                    ax1.set_thetamin(0)
                    ax1.set_thetamax(180)
                    ax1.grid(False)
                    ax1.set_title('Approximatioin results')
                    cax = ax1.contourf(phi, r, values, 30, cmap='jet', origin='lower')
                    colorbar_axes = fig.add_axes([0.125, 0.3, 0.35, 0.025])
                    fig.colorbar(cax, cax=colorbar_axes, orientation='horizontal')

                    self.loss_figure.pyplot(fig)

def plot_final_results(loss_values, val_loss_cycle, num_cycles, final_predictions):
        
        _, _, _, _, ann_cycles, _, _ = get_architecture()
        _, _, _, _, _, _, _, mask1, num_circle, circle_center, circle_radius = get_data()
        # loss plot figure

        fig = plt.figure(figsize=(10, 7))

        r_max, _, _ = get_value()

        nr, nt = (100, 100)   # means that in the domain there are 100.000 data points
        zeniths = np.linspace(0, r_max, nr)
        azimuths = np.radians(np.linspace(0, 180, nt))

        r, phi = np.meshgrid(zeniths, azimuths)
        values = final_predictions.reshape((nr, nt))

        #-- Plot... ------------------------------------------------
        fig = plt.figure(figsize=(10, 7))

        ax1 = fig.add_subplot(121, projection='polar')
        for mask in mask1:
            ax1.contour(phi, r, mask, levels=[0.5], colors='r')
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        ax1.grid(False)
        ax1.set_title('Approximatioin results')
        cax = ax1.contourf(phi, r, values, 30, cmap='jet', origin='lower')
        colorbar_axes = fig.add_axes([0.125, 0.3, 0.35, 0.025])
        fig.colorbar(cax, cax=colorbar_axes, orientation='horizontal')

        x_values = []
        sum = 0
        for i in range(ann_cycles):
            min_x = sum
            sum += len(loss_values[i])
            max_x = sum
            x = np.linspace(min_x, max_x, max_x - min_x)
            y = np.random.uniform(1, 10, len(x))
            x_values.append(x)
        
        ax2 = fig.add_subplot(122)
        for i in range(num_cycles):
            ax2.plot(x_values[i], loss_values[i], label='Cycle {}'.format(i+1))
            ax2.plot(x_values[i], val_loss_cycle[i], label='Cycle {}'.format(i+1))
        ax2.grid(True)
        ymax = 10*min(loss_values[i])
        ax2.set_ylim([0, ymax])
        box = ax2.get_position()
        ax2.set_title('Loss function results')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('MSE')
        ax2.set_position([box.x0, box.y0*2.5, box.width*0.8, box.height*0.5])
        st.pyplot(fig)
        st.markdown('### Diagnosis:')
        if num_circle==0:
            st.markdown('# Normal')
        elif num_circle==1:
            ann_col1, ann_col2 = st.columns(2)
            ann_col1.markdown('# Single Cancer')

            ann_col2.markdown('# Located at:')
            text = rf'## $r$: {round(circle_center[0][0], 2)}'
            ann_col2.markdown(text)
            text = rf'## $\theta$: {round(np.degrees(circle_center[0][1]), 2)}&deg;'
            ann_col2.markdown(text)
            ann_col2.markdown('# With radius:')
            ann_col2.markdown(f"## {round(circle_radius[0], 2)}")

        elif num_circle==2:
            ann_col1, ann_col2, ann_col3 = st.columns(3)

            ann_col1.markdown('# Multiple Cancer')
            ann_col2.markdown('# Located at:')
            text = rf'## $r$: {round(circle_center[0][0], 2)}'
            ann_col2.markdown(text)
            text = rf'## $\theta$: {round(np.degrees(circle_center[0][1]), 2)}&deg;'

            ann_col2.markdown(text)
            ann_col2.markdown('# With radius:')
            ann_col2.markdown(f"## {round(circle_radius[0], 2)}")

            ann_col3.markdown('# Located at:')
            text = rf'## $r$: {round(circle_center[1][0], 2)}'
            ann_col3.markdown(text)
            text = rf'## $\theta$: {round(np.degrees(circle_center[1][1]), 2)}&deg;'
            ann_col3.markdown(text)
            ann_col3.markdown('# With radius:')
            ann_col3.markdown(f"## {round(circle_radius[1], 2)}")
        elif num_circle==3:
            ann_col1, ann_col2, ann_col3, ann_col4 = st.columns(4)
            ann_col1.markdown('# Multiple Cancer')

            ann_col2.markdown('# Located at:')
            text = rf'## $r$: {round(circle_center[0][0], 2)}'
            ann_col2.markdown(text)
            text = rf'## $\theta$: {round(np.degrees(circle_center[0][1]), 2)}&deg;'
            ann_col2.markdown(text)
            ann_col2.markdown('# With radius:')
            ann_col2.markdown(f"## {round(circle_radius[0], 2)}")

            ann_col3.markdown('# Located at:')
            text = rf'## $r$: {round(circle_center[1][0], 2)}'
            ann_col3.markdown(text)
            text = rf'## $\theta$: {round(np.degrees(circle_center[1][1]), 2)}&deg;'
            ann_col3.markdown(text)
            ann_col3.markdown('# With radius:')
            ann_col3.markdown(f"## {round(circle_radius[1], 2)}")

            ann_col4.markdown('# Located at:')
            text = rf'## $r$: {round(circle_center[2][0], 2)}'
            ann_col4.markdown(text)
            text = rf'## $\theta$: {round(np.degrees(circle_center[2][1]), 2)}&deg;'
            ann_col4.markdown(text)
            ann_col4.markdown('# With radius:')
            ann_col4.markdown(f"## {round(circle_radius[2], 2)}")
            


def data_plot():
    
    get_normal_data()
    _, x_u_train, _, x_b0_train, _, x_b_train, _, _, _, _, _ = get_data()

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(r=x_u_train[:, :1].flatten(), 
                                  theta=np.degrees(x_u_train[:, 1:2].flatten()), mode='markers', name='PDE points'))
    fig.add_trace(go.Scatterpolar(r=x_b_train[:, :1].flatten(), 
                                  theta=np.degrees(x_b_train[:, 1:2].flatten()), mode='markers', 
                                  marker=dict(symbol='x'), name='Curve boundary points'))
    fig.add_trace(go.Scatterpolar(r=x_b0_train[:, :1].flatten(), 
                                  theta=np.degrees(x_b0_train[:, 1:2].flatten()), mode='markers', 
                                  marker=dict(symbol='x'), name='Curve boundary points'))
    fig.update_layout(polar=dict(sector=[0, 180]))

    return st.plotly_chart(fig, use_container_width=True)


def ann_plot():
    ann_layers, ann_neurons, ann_iters, apply_restarting, ann_cycles, ann_optimizer, new_model = get_architecture()

    _, x_u_train, y_u_train, x_b0_train, y_b0_train, x_b_train, y_b_train, _, _, _, _ = get_data()
    x_star = get_normal_data()
    
    x_train = np.vstack([x_b_train, x_b0_train, x_u_train])
    y_train = np.vstack([y_b_train, y_b0_train, y_u_train])

    # Create a neural network model
    if new_model=='Yes':
        model = build_model(int(ann_layers), int(ann_neurons))
    else:
        model = get_model(new_model)

    # Compile the model
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    model.compile(optimizer=ann_optimizer, loss='mean_squared_error')
   
    start_time = time.time()

    history_mse_loss = []
    history_mse_cycle = []
    history_val_mse_loss = []
    history_val_mse_cycle = []

    if apply_restarting=='Yes':
        progress_cycle = st.progress(0)
        # Train the model
        for i in range(ann_cycles): 
            
            progress_percent = i / ann_cycles
            progress_cycle.progress(progress_percent)
            if (history_mse_loss==[]):
                info = f'Cycles: {i} - Completion: {int(progress_percent * 100)}% - Current best loss: None'
            else:
                info = f'Cycles: {i} - Completion: {int(progress_percent * 100)}% - Current best loss: {min(history_mse_cycle)}'
            info_text = st.text(info)

            history = model.fit(x_train, y_train, epochs=ann_iters,
                                callbacks=[StreamlitProgressBarCallback(ann_iters, i, ann_cycles), earlystopping], 
                                verbose=0, validation_split=0.15)
            history_mse_cycle.extend(history.history['loss'])
            history_mse_loss.append(history.history['loss'])
            history_val_mse_loss.extend(history.history['val_loss'])
            history_val_mse_cycle.append(history.history['val_loss'])
            
            info_text.empty()
        progress_percent = ann_cycles / ann_cycles
        progress_cycle.progress(progress_percent)
        info = f'Cycles: {ann_cycles} - Completion: {int(progress_percent * 100)}% - Best loss: {min(history_mse_loss[i])}'
        info_text = st.text(info)
        
    else:
        for i in range(ann_cycles):
            history = model.fit(x_train, y_train, epochs=ann_iters,
                                callbacks=[StreamlitProgressBarCallback(ann_iters, i, ann_cycles), earlystopping], 
                                verbose=0, validation_split=0.15)
            history_mse_loss.append(history.history['loss'])
            history_val_mse_loss.extend(history.history['val_loss'])
            history_val_mse_loss.extend(history.history['val_loss'])
            history_val_mse_cycle.append(history.history['val_loss'])
        st.progress(100)
        info = f'Iteration: {ann_iters} - Completion: {100}% - Loss: {min(history_mse_loss[i])}'
        info_text = st.text(info)
    
    predictions = model.predict(x_star)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    st.write(f"Training complete! Total elapsed time: {minutes} minutes {seconds} seconds")
    if new_model=='Yes':
        model.save('new_model.h5')
    else:
        model.save('normal.h5')
    return plot_final_results(history_mse_loss, history_val_mse_cycle, ann_cycles, predictions)

def predictions(time):

    _, _, _, _, _, _, new_model = get_architecture()

    time = np.array(time).reshape(-1, 1)

    if new_model=='Yes':
        model = get_model(new_model)
        pred = model.predict(time)
    else:
        model = get_model(new_model)
        pred = model.predict(time)

    return pred
