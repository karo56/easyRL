import datetime as dt
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from omegaconf import DictConfig

from easyRL import get_project_root

st.set_option("deprecation.showPyplotGlobalUse", False)
st.set_page_config(layout="wide")


def _remove_top_bar():
    st.markdown(
        """
            <style>
                   .block-container {
                        padding-top: 0rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
            </style>
            """,
        unsafe_allow_html=True,
    )


def _list_directories(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def _return_start_time(descrption_path):
    with open(descrption_path) as f:
        for line in f:
            line_striped = line.strip()
            if "Training starts on:" in line_striped:
                only_date = line_striped.replace("Training starts on: ", "")
                return dt.datetime.strptime(only_date, "%Y-%m-%d %H:%M:%S")


def _format_timedelta(delta):  # TODO: add descriptions
    days = delta.days
    hours, remainder_seconds = divmod(delta.seconds, 3600)
    minutes = remainder_seconds // 60

    return f"{days} days {hours} hours {minutes} minutes"


def _read_yaml_config(config_path):
    with open(config_path, "r") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return config_dict


def dashboard_print_estimation_times(df_monitor, experiment_path):
    steps_number = df_monitor["l"].sum()

    config_dict = _read_yaml_config(
        os.path.join(experiment_path, "params", "config.yaml")
    )
    training_steps = config_dict["total_timesteps"]

    description_path = os.path.join(experiment_path, "description.txt")

    time_starts = _return_start_time(description_path)
    time_now = dt.datetime.now()

    delta = time_now - time_starts
    full_training_time = delta * (training_steps / steps_number)

    col1, col2 = st.columns(2)

    # st.write(":heavy_minus_sign:" * 44)

    with col1:
        st.write("### Information about experiment:")

        st.write(f"Training starts at **{time_starts.strftime('%H:%M:%S %Y-%m-%d')}**")
        st.write(f"Current training time: **{_format_timedelta(delta)}**")

        st.write(f"Number of games played: {len(df_monitor)}")
        st.write(
            f"Number of steps: **{steps_number:_}** / **{training_steps:_}**, "
            f"this is **{np.round(steps_number / training_steps * 100, 3)}**% "
            f"of training"
        )

    with col2:
        st.write("### Estimations:")
        st.write(
            f"Estimation of hour when training ends:"
            f" **{(time_starts + full_training_time).strftime('%H:%M:%S %Y-%m-%d')}**"
        )
        st.write(
            f"Estimation time when training ends: "
            f"**{_format_timedelta(full_training_time - delta)}**"
        )
        st.write(
            f"Estimation of TOTAL training time: "
            f"**{_format_timedelta(full_training_time)}**"
        )


def plot_multiple_lists(time, value, value_2, value_3, length_or_reward):
    plt.figure(figsize=(12, 5))

    # Plotting value vs. time
    plt.plot(time, value, label=length_or_reward)

    # Plotting value_2 vs. time
    plt.plot(time, value_2, label=f"{length_or_reward} avg 10")

    # Plotting value_3 vs. time
    plt.plot(time, value_3, label=f"{length_or_reward} avg 100")

    plt.xlabel("Number of games")
    plt.ylabel(length_or_reward)
    plt.title(f"{length_or_reward} for environment: {env_name} and algorithm {algo_id}")
    plt.legend()

    st.pyplot()


def dashboard_create_plots(df_monitor, length_or_reward):

    x = "l" if length_or_reward == "Lengths" else "r"
    value = list(df_monitor[x])  # y axis
    time = list(range(len(value)))  # x axis

    N = 10
    value_2 = [np.nan] * N + value
    value_2 = [np.nanmean(value_2[i : i + N]) for i in range(1, len(value_2) - N + 1)]

    N = 100
    value_3 = [np.nan] * N + value
    value_3 = [np.nanmean(value_3[i : i + N]) for i in range(1, len(value_3) - N + 1)]

    plot_multiple_lists(time, value, value_2, value_3, length_or_reward)


@hydra.main(
    config_path=os.path.join(get_project_root(), "config"),
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    global env_name, algo_id

    # remove top bar
    _remove_top_bar()

    output_folder = cfg.path_to_outputs
    st.title("EasyRL experiments monitoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        env_name = st.selectbox(
            "Select Environment Name", _list_directories(output_folder)
        )

    with col2:
        algo_id = st.selectbox(
            "Select Algorithm ID",
            _list_directories(os.path.join(output_folder, env_name)),
        )

    with col3:
        experiment_name = st.selectbox(
            "Select Experiment Name",
            _list_directories(os.path.join(output_folder, env_name, algo_id)),
        )

    with col1:
        col1, col2 = st.columns([0.7, 1.3])
        with col1:
            button = st.button("Refresh experiment")
        with col2:
            st.warning("Please remember to refresh the experiment", icon="⚠️")
    if button:
        # Read logger
        experiment_path = os.path.join(
            output_folder, env_name, algo_id, experiment_name
        )
        path_monitor = os.path.join(experiment_path, "logger", "monitor.csv")
        df_monitor = pd.read_csv(path_monitor, skiprows=[0])

        dashboard_print_estimation_times(df_monitor, experiment_path)

        st.write("### Plots:")
        col1, col2 = st.columns(2)

        with col1:
            dashboard_create_plots(df_monitor, "Lengths")

        with col2:
            dashboard_create_plots(df_monitor, "Rewards")


if __name__ == "__main__":
    main()
