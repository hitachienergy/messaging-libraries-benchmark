from argparse import ArgumentParser
import os
import re

import pandas as pd
from bml_optimizer.plots.plot_lib import PlotResult, plot_best_points, plot_vs_payload, plot_vs_subscribers, plot_optimal_library, get_plot_results_from_df, load_df_from_file
from bml_optimizer.utils.logger import get_logger
from bml_optimizer.simulator.simulator import Protocol, parse_protocol
from bml_optimizer.scripts.workload_runner import load_results_from_csv
from bml_optimizer.plots.plot_lib import plot_workloads
from bml_optimizer.scripts.optimizer import load_gp_results
from bml_optimizer.plots.plot_gp import produce_gp_plots

logger = get_logger(__name__)


def produce_workload_plots(results_folder: str) -> None:
    """
    Load all the csv files in the results folder and produce visualizations
    """
    all_files = [f for f in os.listdir(results_folder) if "workloads" in f and f.endswith(".csv")]
    for f in all_files:
        logger.info(f"Processing file {f}")
        p, wl = load_results_from_csv(os.path.join(results_folder, f))
        plot_workloads(wl, p)


def produce_benchmark_plots(results_file: str, plots_to_produce: list[str] = ["optimal", "pub-int-0", "pub-int-1000"], use_log_scale: bool=False) -> None:
    """
    Load the results from a file and produce visualizations
    """
    df: pd.DataFrame = load_df_from_file(results_file)
    unique_payloads: int = df['Payload Length'].unique()
    unique_subscribers: int = df['Num Subscribers'].unique()
    unique_intervals: int = df['Pub Interval'].unique()
    logger.info(f"Unique Payloads: {len(unique_payloads)}, Unique Subscribers: {len(unique_subscribers)}, Unique Intervals: {len(unique_intervals)}")

    constant_pub_delay = df['Pub Delay'].unique()[0]
    constant_message_count = df['Messages Sent'].unique()[0]
    logger.info(f"Fixed Pub Delay: {constant_pub_delay}, Fixed Message Count: {constant_message_count}")
    
    """
    Setup: 
            If there are >= 10 payload values, use a scatter plot, otherwise use a bar
    """
    results: list[PlotResult] = get_plot_results_from_df(df)

    plot_kind = 'scatter' if len(unique_payloads) >= 10 else 'bar'
    common_fom = ['Mean Jitter', 'Median CPU', 'Median MEM']
    throughput_fom = ['Throughput', 'Payload Throughput']
    latency_fom = ['Min Latency', 'Avg Latency', 'Max Latency']

    considered_payload = 100000 if 100000 in unique_payloads else 128000 # either normal or exponential
    considered_subscribers = unique_subscribers

    
    """
    1:  Fix the message count to 5000, the delay to 1000
            and the publisher interval to 0 
            to compare the libraries in terms of throughput
        Produce a plot vs. payload (fixed subscribers)
        Produce a plot vs. subscribers (fixed payload)
            for each protocol
    """
    if "pub-int-0" in plots_to_produce:
        fom = common_fom + throughput_fom
        for protocol in Protocol:
            if len(unique_subscribers) > 1:
                plot_vs_subscribers(all_plot_results=results, fixed_protocol=protocol, fixed_message_count=constant_message_count, fixed_pub_interval=0, 
                                    fixed_pub_delay=constant_pub_delay, fixed_payload_length=considered_payload, target_fom=fom, use_log_scale=use_log_scale)
            if len(unique_payloads) > 1:
                for s in considered_subscribers:
                    plot_vs_payload(all_plot_results=results, fixed_protocol=protocol, fixed_message_count=constant_message_count, fixed_pub_interval=0,
                                    fixed_pub_delay=constant_pub_delay, fixed_subscribers=s, target_fom=fom, plot_kind=plot_kind, use_log_scale=use_log_scale)
                
    """
    2:  Fix the message count to 5000, the delay to 1000
            and the publisher delay to 500ms 
            to compare the libraries in terms of latency
        Produce a plot vs. payload 
        Produce a plot vs. subscribers 
            for each protocol
    """
    if "pub-int-1000" in plots_to_produce:
        fom = common_fom + latency_fom
        for protocol in Protocol:
            if len(unique_subscribers) > 1:
                plot_vs_subscribers(all_plot_results=results, fixed_protocol=protocol, fixed_message_count=constant_message_count, fixed_pub_interval=1000, 
                                    fixed_pub_delay=constant_pub_delay, fixed_payload_length=considered_payload, target_fom=fom, use_log_scale=use_log_scale)
             
            if len(unique_payloads) > 1:
                for s in considered_subscribers:
                    plot_vs_payload(all_plot_results=results, fixed_protocol=protocol, fixed_message_count=constant_message_count, fixed_pub_interval=1000, 
                                    fixed_pub_delay=constant_pub_delay, fixed_subscribers=s, target_fom=fom, plot_kind=plot_kind, use_log_scale=use_log_scale)
                    
    """
    3:  Fix the message count to 5000, the delay to 1000
            consider all the payload (x-axis) and intervals (y-axis) in a single plot

        We consider, for each point (payload, interval), all the libraries evaluated (i.e. ZeroMQ, NanoMsg, NNG)
        Each point is colored based on the optimal library for the figure of merit for that specific combination

        We produce a plot of this kind for each protocol
    """
    if "optimal" in plots_to_produce:
        fom_best: dict[str, bool] = {'Min Latency': True, 'Avg Latency': True, 'Max Latency': True, 
                                    'Mean Jitter': True, 
                                    'Median CPU': True, 'Median MEM': True}
        fom_best_throughput: dict[str, bool] = {'Throughput': False, 'Payload Throughput': False}

        res_sub_plot: list[PlotResult] = [r for r in results if r._config._pub_interval == 1000]
        res_sub_plot_throughput: list[PlotResult] = [r for r in results if r._config._pub_interval == 0]

        fixed_subscribers = 1
        res_interval_plot: list[PlotResult] = [r for r in results if r._config._subscribers == fixed_subscribers]
        
        for protocol in Protocol:
            for fom, best_is_min in fom_best.items():
                plot_optimal_library(all_plot_results=res_sub_plot, fixed_protocol=protocol, 
                                     fixed_message_count=constant_message_count, fixed_pub_delay=constant_pub_delay, figure_of_merit=fom, 
                                     x_axis_metric='Payload Length', y_axis_metric='Num Subscribers', best_is_min=best_is_min, 
                                     use_log_scale_x=use_log_scale, use_log_scale_y=use_log_scale)
                
                plot_optimal_library(all_plot_results=res_interval_plot, fixed_protocol=protocol,
                                     fixed_message_count=constant_message_count, fixed_pub_delay=constant_pub_delay, figure_of_merit=fom, 
                                     x_axis_metric='Payload Length', y_axis_metric='Pub Interval', best_is_min=best_is_min, 
                                     use_log_scale_x=use_log_scale, use_log_scale_y=False)
        
            for fom, best_is_min in fom_best_throughput.items():
                plot_optimal_library(all_plot_results=res_sub_plot_throughput, fixed_protocol=protocol, 
                                     fixed_message_count=constant_message_count, fixed_pub_delay=constant_pub_delay, figure_of_merit=fom, 
                                     x_axis_metric='Payload Length', y_axis_metric='Num Subscribers', best_is_min=best_is_min,
                                        use_log_scale_x=use_log_scale, use_log_scale_y=use_log_scale)

    """
    4:  Without filtering, plot the best points minimizing the latency
    """
    if "best-points" in plots_to_produce:
        for protocol in Protocol:
            plot_best_points(plot_results=results, fixed_protocol=protocol, figure_of_merit='Avg Latency', points=4)
            plot_best_points(plot_results=results, fixed_protocol=protocol, figure_of_merit='Payload Throughput', points=4, best_is_min=False)


def produce_optimize_plots(folder_optimize: str) -> None:
    """
    Load dumped GP results from the folder_optimize, parse the filenames to retrieve parameters, 
    then call produce_gp_plots for each result.

    Parse naming scheme: 
        gp_results_{bytes}_{proto}_{gp_shots}_{gp_init}_{runs}_{pub_delay}_{subscribers}_{constraints_short_str}_{uuid}.pkl
    """
    from bml_optimizer.scripts.optimizer import Constraints
    pkl_files = [f for f in os.listdir(folder_optimize) if f.endswith(".pkl")]
    for f in pkl_files:
        filepath = os.path.join(folder_optimize, f)
        try:
            result = load_gp_results(filepath)
            splitted = f.split("_")
            bytes_to_send = int(splitted[2])
            max_time = int(splitted[3])
            protocol = parse_protocol(splitted[4])
            gp_shots = int(splitted[5])
            gp_initial_points = int(splitted[6])
            runs = int(splitted[7])
            pub_delay = int(splitted[8])
            num_subscribers = int(splitted[9])
            cpu_max = float(splitted[10])
            cpu_weight = float(splitted[11])
            mem_max = float(splitted[12])
            mem_weight = float(splitted[13])
            sim_id = splitted[14].split(".")[0]
            
            logger.info(f"Processing file {f}, corresponding to the simulation with {bytes_to_send} bytes, {protocol} protocol, {gp_shots} GP shots, {gp_initial_points} GP initial points, {runs} runs, {pub_delay} pub delay, {num_subscribers} subscribers, {cpu_max} CPU max, {cpu_weight} CPU weight, {mem_max} MEM max, {mem_weight} MEM weight")
            produce_gp_plots(
                result=result,
                bytes_to_send=bytes_to_send,
                max_time=max_time,
                gp_shots=gp_shots,
                gp_initial_points=gp_initial_points,
                runs=runs,
                protocol=protocol,
                pub_delay=pub_delay,
                num_subscribers=num_subscribers,
                constraints=Constraints(cpu_max=cpu_max, cpu_weight=cpu_weight, mem_max=mem_max, mem_weight=mem_weight),
                plot_id=sim_id
            )
        except Exception as e:
            logger.warning(f"Failed to load and plot {f}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot the results from a file")
    parser.add_argument('--file_benchmark', type=str, default="results/results.csv", help='File to load the results from')
    parser.add_argument('--folder_workload', type=str, default="results", help='File to load the results from')
    parser.add_argument('--folder_optimize', type=str, default="results/optimize_dump", help='Folder containing dumped optimization results')
    parser.add_argument('--use_log_scale', action='store_true', help='Enable log scale for x/y in scatter & optimality plots')
    args = parser.parse_args()

    results_file = args.file_benchmark
    try:
        produce_benchmark_plots(results_file=results_file, use_log_scale=args.use_log_scale)
    except FileNotFoundError as e:
        logger.warning(f"File {results_file} not found, skipping")
    except IndexError as e:
        logger.warning(f"File {results_file} is empty, skipping")

    results_folder = args.folder_workload
    try:
        produce_workload_plots(results_folder=results_folder)
    except FileNotFoundError as e:
        logger.warning(f"Folder {results_folder} not found, skipping")

    try:
        logger.info(f"Producing optimization plots from folder: {args.folder_optimize}")
        produce_optimize_plots(args.folder_optimize)
    except FileNotFoundError as e:
        logger.warning(f"Folder {args.folder_optimize} not found, skipping")