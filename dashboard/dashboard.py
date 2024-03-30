import datetime as dt
import os

import hydra
import numpy as np
import pandas as pd
import streamlit as st
from omegaconf import DictConfig

from easyRL import get_project_root


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


def dashboard_print_estimation_times(df_monitor, cfg, experiment_path):
    steps_number = df_monitor["l"].sum()
    training_steps = cfg["total_timesteps"]

    description_path = os.path.join(experiment_path, "description.txt")

    time_starts = _return_start_time(description_path)
    time_now = dt.datetime.now()

    delta = time_now - time_starts
    full_training_time = delta * (training_steps / steps_number)

    st.write(f"Training starts at **{time_starts.strftime('%H:%M:%S %Y-%m-%d')}**")
    st.write(f"Current training time: **{_format_timedelta(delta)}**")

    st.write(f"Number of games played: {len(df_monitor)}")
    st.write(
        f"Number of steps: **{steps_number:_}** / **{training_steps:_}**, this is **{np.round(steps_number / training_steps * 100, 3)}**% of traning"
    )

    st.write(":heavy_minus_sign:" * 34)
    st.write("### Estimations:")
    st.write(
        f"Estimation of hour when training ends: **{(time_starts + full_training_time).strftime('%H:%M:%S %Y-%m-%d')}**"
    )
    st.write(
        f"Estimation time when training ends: **{_format_timedelta(full_training_time - delta)}**"
    )
    st.write(
        f"Estimation of TOTAL training time: **{_format_timedelta(full_training_time)}**"
    )


def dashboard_create_plots(df_monitor, length_or_reward):

    value = list(df_monitor["l"])  # y axis
    time = list(range(len(value)))  # x axis

    last_n = len(value)

    N = 10
    value_2 = [np.nan] * N + value
    value_2 = [np.nanmean(value_2[i : i + N]) for i in range(1, len(value_2) - N + 1)]

    N = 100
    value_3 = [np.nan] * N + value
    value_3 = [np.nanmean(value_3[i : i + N]) for i in range(1, len(value_3) - N + 1)]

    # create plots
    df_to_plot = pd.DataFrame({"x": time, "y": value})
    st.line_chart(df_to_plot.set_index("x"))

    df_to_plot = pd.DataFrame({"x": time, "y": value_2})
    st.line_chart(df_to_plot.set_index("x"))


@hydra.main(
    config_path=os.path.join(get_project_root(), "config"),
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    # TODO: fix plots, refresh everything

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

    if st.button("Refresh experiment"):
        # Read logger
        experiment_path = os.path.join(
            output_folder, env_name, algo_id, experiment_name
        )
        path_monitor = os.path.join(experiment_path, "logger", "monitor.csv")
        df_monitor = pd.read_csv(path_monitor, skiprows=[0])

        dashboard_print_estimation_times(df_monitor, cfg, experiment_path)
        dashboard_create_plots(df_monitor, "l")


if __name__ == "__main__":
    main()
