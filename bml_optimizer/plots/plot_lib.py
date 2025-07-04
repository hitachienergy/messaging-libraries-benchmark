from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter
from bml_optimizer.simulator.simulator import BML, parse_bml, Protocol, parse_protocol, SimResults, Configuration, Workload
from bml_optimizer.utils.logger import get_logger
from bml_optimizer.plots.plot_utils import set_palette, get_plot_config, get_metric_plot_info, upper_camel_case_bml, upper_camel_case_protocol
from scipy.interpolate import griddata
from typing import List

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


logger = get_logger(__name__)
set_palette(type='normal')
plot_type = 'paper'
formatter, figsize_bar, figsize_scatter, figsize_optimal, figsize_gp, file_format, output_dir = get_plot_config(plot_type)

class PlotResult:
    def __init__(self, library: BML, protocol: Protocol, config: Configuration, workload: Workload, results: SimResults):
        self._library = library
        self._protocol = protocol
        self._config = config
        self._workload = workload
        self._results = results


def get_plot_results_from_df(df: pd.DataFrame) -> List[PlotResult]:
    """
    Convert a DataFrame to a list of PlotResult
    """
    results = []
    for _, row in df.iterrows():
        library, protocol = parse_bml(row['Library']), parse_protocol(row['Protocol'])
        config = Configuration(pub_interval=row['Pub Interval'], pub_delay=row['Pub Delay'], subscribers=row['Num Subscribers'])
        workload = Workload(message_count=row['Messages Sent'], payload_length=row['Payload Length'])
        sim_results = SimResults(message_received=row['Messages Received'], time=row['Total Time'], 
                                 throughput=row['Throughput'], payload_throughput=row['Payload Throughput'], 
                                 min_latency=row['Min Latency'], avg_latency=row['Avg Latency'], max_latency=row['Max Latency'], 
                                 p90_latency=row['P90 Latency'], p99_latency=row['P99 Latency'], mean_jitter=row['Mean Jitter'],
                                 median_cpu=row['Median CPU'], median_mem=row['Median MEM'])
        results.append(PlotResult(library=library, protocol=protocol, config=config, workload=workload, results=sim_results))
    return results

"""
Plot classes
"""

def load_df_from_file(file_path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a file (if it exists)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        return pd.read_csv(file_path)

class PlotResults:
    def __init__(self, results: List[PlotResult]):
        self._results = results
    
    def get_df(self) -> pd.DataFrame:
        """
        Convert the list of PlotResult to a pandas DataFrame
        Considering a row is a PlotResult
        """
        records = [
            {
                'Library': result._library.name,
                'Protocol': result._protocol.name,
                'Messages Sent': result._workload._message_count,
                'Payload Length': result._workload._payload_length,
                'Pub Delay': result._config._pub_delay,
                'Pub Interval': result._config._pub_interval,
                'Num Subscribers': result._config._subscribers,
                'Messages Received': result._results._message_received,
                'Total Time': result._results._time,
                'Throughput': result._results._throughput,
                'Payload Throughput': result._results._payload_throughput,
                'Min Latency': result._results._min_latency,
                'Avg Latency': result._results._avg_latency,
                'Max Latency': result._results._max_latency,
                'P90 Latency': result._results._p90_latency,
                'P99 Latency': result._results._p99_latency,
                'Mean Jitter': result._results._mean_jitter,
                'Median CPU': result._results._median_cpu,
                'Median MEM': result._results._median_mem
            }
            for result in self._results
        ]
        return pd.DataFrame.from_records(records)

"""
Plot functions
"""

def plot_vs_payload(all_plot_results: List[PlotResult], 
                        fixed_protocol: Protocol, fixed_message_count: int,
                        fixed_pub_interval: int, fixed_pub_delay: int, fixed_subscribers: int,
                        target_fom: list[str] = ['Min Latency', 'Avg Latency', 'Max Latency', 'Mean Jitter'],
                        plot_kind: str = 'bar', use_log_scale: bool = False):
    """
    Plot the results vs. payload length, filtering over a single protocol, message count, pub interval, pub delay and number of subscribers
    """
    filtered_results: list[PlotResult] = [r for r in all_plot_results 
                                                if r._workload._message_count == fixed_message_count 
                                                and r._config._pub_interval == fixed_pub_interval
                                                and r._config._pub_delay == fixed_pub_delay 
                                                and r._config._subscribers == fixed_subscribers
                                                and r._protocol == fixed_protocol]
    if len(filtered_results) == 0:
        logger.warning(f"No results found for {fixed_protocol} with message count {fixed_message_count}, {fixed_subscribers} subs and pub interval {fixed_pub_interval} and pub delay {fixed_pub_delay}")
        return
    else:
        logger.info(f"Plotting vs. payload length for {fixed_protocol} with message count {fixed_message_count}, {fixed_subscribers} subs and pub interval {fixed_pub_interval} and pub delay {fixed_pub_delay}")

    experiment_name = f'{fixed_pub_interval}us/payloads-{fixed_subscribers}subs'
    plot_func = plot_bar if plot_kind == 'bar' else plot_scatter

    for fom in target_fom:
        plot_func(plot_results=PlotResults(filtered_results), group_by_parameter='Payload Length', figure_of_merit=fom,
                 experiment_name=experiment_name,
                 title=f'{fom} vs. Payload Length\nProtocol: {fixed_protocol.name.lower()}, Subscribers: {fixed_subscribers}\nMessage Count: {fixed_message_count}, Publishing Interval: {fixed_pub_interval} μs',
                 use_log_scale=use_log_scale)


def plot_vs_subscribers(all_plot_results: List[PlotResult],
                            fixed_protocol: Protocol, fixed_message_count: int,
                            fixed_pub_interval: int, fixed_pub_delay: int, fixed_payload_length: int,
                            target_fom: list[str] = ['Min Latency', 'Avg Latency', 'Max Latency', 'Mean Jitter'],
                            plot_kind: str = 'bar', use_log_scale: bool = False):
    """
    Plot the results vs. number of subscribers, filtering over a single protocol, message count, pub interval, pub delay and payload length
    """
    filtered_results: list[PlotResult] = [r for r in all_plot_results
                                                if r._workload._message_count == fixed_message_count
                                                and r._config._pub_interval == fixed_pub_interval
                                                and r._config._pub_delay == fixed_pub_delay
                                                and r._workload._payload_length == fixed_payload_length
                                                and r._protocol == fixed_protocol]
    if len(filtered_results) == 0:
        logger.warning(f"No results found for {fixed_protocol} with message count {fixed_message_count}, payload {fixed_payload_length} and pub interval {fixed_pub_interval} and pub delay {fixed_pub_delay}")
        return
    else:
        logger.info(f"Plotting vs. number of subscribers for {fixed_protocol} with message count {fixed_message_count}, payload {fixed_payload_length} and pub interval {fixed_pub_interval} and pub delay {fixed_pub_delay}")

    experiment_name = f'{fixed_pub_interval}us/subs-{fixed_payload_length}B'
    plot_func = plot_bar if plot_kind == 'bar' else plot_scatter
    
    for fom in target_fom:
        plot_func(plot_results=PlotResults(filtered_results), group_by_parameter='Num Subscribers', figure_of_merit=fom,
                 experiment_name=experiment_name,
                 title=f'{fom} vs. Payload Length\nProtocol: {fixed_protocol.name.lower()}, Payload: {fixed_payload_length}B\nMessage Count: {fixed_message_count}, Publishing Interval: {fixed_pub_interval} μs',
                 use_log_scale=use_log_scale)


def plot_bar(plot_results: PlotResults,
                 group_by_parameter: str,
                 figure_of_merit: str,
                 experiment_name: str = 'experiment',
                 title: str = None,
                 use_log_scale: bool = False):
    """
    Plot a bar plot of the figure of merit against the group_by_parameter
    Pass to this function only the results to be plotted (e.g. only the results for a specific paradigm, or a specific time interval)
    Group by a parameter (+ the libraries), and plot the figure of merit against the group parameter
    """
    try:
        df = plot_results.get_df()
        
        libraries = df['Library'].unique()
        protocols = df['Protocol'].unique()

        assert len(protocols) == 1
        protocol = protocols[0]
        
        grouped = df.groupby([group_by_parameter, 'Library'])

        mean_fom = grouped[figure_of_merit].mean().reset_index()
        grouped = mean_fom.pivot(index=group_by_parameter, columns='Library', values=figure_of_merit)
        grouped = grouped.reindex(columns=libraries)

        ax = grouped.plot(kind='bar', figsize=figsize_bar)
        ax.set_xlabel(get_metric_plot_info(group_by_parameter))
        ax.set_ylabel(get_metric_plot_info(figure_of_merit))

        ax.yaxis.set_major_formatter(formatter)
        if title and plot_type == 'normal': ax.set_title(title)
        
        plt.xticks(rotation=0)
        handles, labels = ax.get_legend_handles_labels()
        labels = [upper_camel_case_bml[label.lower()] for label in labels]
        ax.legend(handles, labels, frameon=True)
        plt.tight_layout()
        
        folder = f'{output_dir}/benchmark/{protocol.lower()}-{experiment_name}/{figure_of_merit.lower().replace(" ", "")}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig = ax.get_figure()

        plt.savefig(f'{folder}/bar_plot_by_{group_by_parameter.lower().replace(" ", "")}.{file_format}')
        plt.close(fig)
    
    except Exception as e:
        print(f'Error in plot_bar: {e.__class__.__name__} ({figure_of_merit}) - {e}')
        raise e

def plot_scatter(plot_results: PlotResults,
                 group_by_parameter: str,
                 figure_of_merit: str,
                 experiment_name: str = 'experiment',
                 title: str = None,
                 use_log_scale: bool = False):
    """
    Produce a scatter plot of the figure_of_merit against a parameter (e.g. payload length)
    Each library is shown in a different color, forming a distinct curve.
    """
    try:
        df = plot_results.get_df()
        libraries = df['Library'].unique()
        fig, ax = plt.subplots(figsize=figsize_scatter)

        protocols = df['Protocol'].unique()
        assert len(protocols) == 1
        protocol = protocols[0]

        for lib in libraries:
            sub_df = df[df['Library'] == lib].sort_values(by=group_by_parameter)
            ax.plot(sub_df[group_by_parameter], sub_df[figure_of_merit], label=upper_camel_case_bml[lib.lower()], marker='o', markersize=4)
        ax.set_xlabel(get_metric_plot_info(group_by_parameter))
        ax.set_ylabel(get_metric_plot_info(figure_of_merit))

        if use_log_scale:
            ax.set_xscale('log', base=2)

        if title and plot_type == 'normal': 
            ax.set_title(title)
        ax.yaxis.set_major_formatter(formatter)
        plt.legend(frameon=True)
        plt.tight_layout()

        folder = f'{output_dir}/benchmark/{protocol.lower()}-{experiment_name}/{figure_of_merit.lower().replace(" ", "")}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(f'{folder}/scatter_plot_{figure_of_merit.lower().replace(" ", "")}.{file_format}')
        plt.close(fig)
    except Exception as e:
        print(f'Error in plot_scatter: {e.__class__.__name__} ({figure_of_merit}) - {e}')
        raise e


def plot_optimal_library(all_plot_results: List[PlotResult],
                            fixed_protocol: Protocol,
                            fixed_message_count: int,
                            fixed_pub_delay: int,
                            figure_of_merit: str,
                            x_axis_metric: str,
                            y_axis_metric: str,
                            best_is_min: bool = True,
                            use_log_scale_x: bool = False,
                            use_log_scale_y: bool = False):
    """
    For each (payload, interval), pick the library with the best (lowest/highest) figure_of_merit
    and produce a 2D plot with x=payload, y=interval, colored by the chosen library.
    """
    filtered_results: list[PlotResult] = [r for r in all_plot_results
                                                if r._protocol == fixed_protocol
                                                and r._workload._message_count == fixed_message_count
                                                and r._config._pub_delay == fixed_pub_delay]

    if len(filtered_results) == 0:
        logger.warning(f"No results found for {fixed_protocol} with message count {fixed_message_count} and pub delay {fixed_pub_delay}")
        return
    else:
        logger.info(f"Plotting ({figure_of_merit} / {fixed_protocol}) optimal library using ({x_axis_metric}, {y_axis_metric}) as (x, y) axis metrics")

    experiment_name = f'optimality'
    plot_contour(plot_results=PlotResults(filtered_results),
                figure_of_merit=figure_of_merit,
                x_axis_metric=x_axis_metric,
                y_axis_metric=y_axis_metric,
                experiment_name=experiment_name,
                best_is_min=best_is_min,
                use_log_scale_x=use_log_scale_x,
                use_log_scale_y=use_log_scale_y)


def get_best_library(figure_of_merit, x_axis_metric, y_axis_metric, best_is_min, grouped) -> pd.DataFrame:
    """
    Auxiliary function to produce the dataframe with the best libraries for each (x, y) pair
    """
    best_rows = []
    for _, sub_df in grouped.groupby([x_axis_metric, y_axis_metric]):
        if best_is_min:
            row = sub_df.nsmallest(1, figure_of_merit).iloc[0]
        else:
            row = sub_df.nlargest(1, figure_of_merit).iloc[0]
        best_rows.append(row)
    best_df = pd.DataFrame(best_rows)
    return best_df


def get_rectangles_edges(v: np.ndarray, use_log_scale: bool = False):
    """
    Compute the edges for the rectangles in the contour plot
    """
    def compute_edges(u):
        return np.concatenate(([u[0] - (u[1] - u[0]) / 2],
                                (u[:-1] + u[1:]) / 2,
                                [u[-1] + (u[-1] - u[-2]) / 2]))
    if use_log_scale:
        eps = 1e-10
        v = np.where(v > 0, v, eps)
        log_x = np.log2(v)
        log_x_unique = np.sort(np.unique(log_x))
        x_unique = 2**log_x_unique
        log_x_edges = compute_edges(log_x_unique)
        x_edges = 2 ** log_x_edges
    
    else:
        x_unique = np.sort(np.unique(v))
        x_edges = compute_edges(x_unique)
    
    return x_edges, x_unique

def plot_contour(plot_results: PlotResults, figure_of_merit: str, 
                x_axis_metric: str, y_axis_metric: str, 
                experiment_name: str = 'experiment',
                best_is_min: bool = True,
                use_log_scale_x: bool = False,
                use_log_scale_y: bool = False):
    """
    Produce a contour plot of the figure_of_merit against two parameters (e.g. payload length and pub interval)
        For each group of (x, y), pick the best library
        i.e., the one maximizing or minimizing the figure_of_merit
    
        Produce the plot
        The z dimension represents the library indices, 
            assigned using the BML enum found in simulator.py.
        Each unique library is assigned a different color, 
            and the z values determine which color is used at each point in the plot.
    """
    df: pd.DataFrame = plot_results.get_df()
    
    # map from library name to an integer index (based on the enum order)
    bml_values: list[str] = [b.name for b in BML]
    bml_mapping: dict[str, int] = {name: i for i, name in enumerate(bml_values)}

    group_cols: list[str] = [x_axis_metric, y_axis_metric, 'Library']
    grouped = df.groupby(group_cols)[figure_of_merit].mean().reset_index()
    
    fixed_protocol: Protocol = parse_protocol(df['Protocol'].unique()[0])
    fixed_message_count: int = df['Messages Sent'].unique()[0]
    fixed_pub_delay: int = df['Pub Delay'].unique()[0]

    best_df: pd.DataFrame = get_best_library(figure_of_merit, x_axis_metric, y_axis_metric, best_is_min, grouped)

    _, ax = plt.subplots(figsize=figsize_optimal)
    x: np.ndarray = best_df[x_axis_metric].values
    y: np.ndarray = best_df[y_axis_metric].values
    logger.debug(f"x unique: {np.unique(x)}, y unique: {np.unique(y)}")
    # map string library names to their numeric index via bml_mapping
    z = best_df['Library'].apply(lambda lib: bml_mapping[lib]).values
    
    x_edges, x_unique = get_rectangles_edges(x, use_log_scale=use_log_scale_x)
    y_edges, y_unique = get_rectangles_edges(y, use_log_scale=use_log_scale_y)
    
    xi, yi = np.meshgrid(x_unique, y_unique)
    zi = griddata((x, y), z, (xi, yi), method='nearest')
    
    # color map for the libraries actually present in best_df
    unique_libs = [lib for lib in bml_values if lib in best_df['Library'].unique()]
    
    palette = set_palette(type='normal')
    lib_color_map = {}
    for i, lib in enumerate(bml_values):
        lib_color_map[lib] = palette[i]
    cmap = ListedColormap([lib_color_map[lib] for lib in unique_libs])

    ax.pcolormesh(x_edges, y_edges, zi, cmap=cmap, shading='auto')
    
    ax.set_xlabel(get_metric_plot_info(x_axis_metric))
    ax.set_ylabel(get_metric_plot_info(y_axis_metric))

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])

    if use_log_scale_x:
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks([2**i*1000 for i in range(0, int(math.log2(x.max()//1000))+1)])
    
    if use_log_scale_y:
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks([2**i for i in range(0, int(math.log2(y.max()))+1)])
    
    ax.legend(
        handles=[mpatches.Patch(color=lib_color_map[l], label=upper_camel_case_bml[l.lower()]) for l in unique_libs],
        title="Library"
    )

    if plot_type == 'normal': 
        title =  (f'{upper_camel_case_protocol[fixed_protocol.name]} Communication\n'
                  f'Message Count: {fixed_message_count}, Pub. Delay: {fixed_pub_delay}ms')
        if df['Num Subscribers'].nunique() == 1:
            title += f', Subscribers: {df["Num Subscribers"].unique()[0]}\n'
        elif df['Pub Interval'].nunique() == 1:
            title += f', Pub. Interval: {df["Pub Interval"].unique()[0]}μs\n'
        else:
            logger.error(f"Multiple values for both Pub Interval and Subscribers in the dataset (should be either)")
        title += f'Optimal Library by {figure_of_merit}'
        ax.set_title(title)
    else:
        plt.tight_layout()
        ax.xaxis.set_major_formatter(formatter)

    folder = f'{output_dir}/benchmark/{fixed_protocol.name.lower()}-{experiment_name}/{figure_of_merit.lower().replace(" ", "")}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/optimal_plot_vs{y_axis_metric.replace(" ", "").lower()}_{figure_of_merit.lower().replace(" ", "")}.{file_format}')
    plt.close()



def plot_workloads(workload_results, protocol: Protocol):
    """
    Produces a scatter plot for a given protocol, with 
        one line for each library (plus the optimized version),
        the current cumulate time passed, on the x-axis,
        the current figure of merit after that time passed, on the y-axis.

    Then, produce two (extra) plots with the steps on the x-axis
        one with the time on the y-axis
        one with the figure of merit on the y-axis
    """

    plot_metrics = [
        ('Total Time', 'Avg Latency', 'aggregated'),
        ('Steps', 'Avg Latency', 'latencies'),
        ('Steps', 'Total Time', 'times')
    ]
    palette = set_palette(type='normal')
    marker_map = {
        BML.ZEROMQ: palette[0],
        BML.NANOMSG: palette[1],
        BML.NNG: palette[2]
    }

    for x_axis_metric, y_axis_metric, axis_type in plot_metrics:
        plt.figure(figsize=figsize_scatter)
        
        for (lib_type, bml) in [
            ("Fixed", BML.ZEROMQ),
            ("Fixed", BML.NANOMSG),
            ("Fixed", BML.NNG),
            ("Predicted", None),
        ]:
            check_condition = lambda wr: wr.library[0] == lib_type and (bml is None or bml == wr.library[1])
            steps: list[int] = [(wr.step+1) for wr in workload_results if check_condition(wr)]
            times: list[float] = [wr.current_time for wr in workload_results if check_condition(wr)]
            foms: list[float] = [wr.current_fom for wr in workload_results if check_condition(wr)]

            x_axis_values, y_axis_values =  steps if axis_type != 'aggregated' else times, \
                                            foms  if axis_type != 'times' else times
            
            if lib_type == "Fixed":
                plt.plot(x_axis_values, y_axis_values, marker='o', label=upper_camel_case_bml[bml.name])
            else:
                markers = [marker_map[wr.library[1]] for wr in workload_results if wr.library[0] == "Predicted"]
                plt.plot(x_axis_values, y_axis_values, marker='o', label=f"Optimized")
                for marker, x, y in zip(markers, x_axis_values, y_axis_values):
                    plt.plot(x, y, marker='x', color=marker, markersize=4)                                                                              

        plt.plot([], [], marker='x', color='black', markersize=4, label='Chosen Library')

        plt.xlabel(get_metric_plot_info(x_axis_metric))
        plt.ylabel(get_metric_plot_info(y_axis_metric))
        plt.gca().yaxis.set_major_formatter(formatter)

        if plot_type == 'normal':
            plt.title(f'Performance of Libraries over {x_axis_metric} for {upper_camel_case_protocol[protocol.name]} Communication')
        
        plt.tight_layout()
        plt.legend()

        folder = f'{output_dir}/workloads'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f'{output_dir}/workloads/{protocol.name.lower()}_{axis_type}_workloads.{file_format}')
        plt.close()


def plot_best_points(plot_results: list[PlotResult], fixed_protocol: Protocol, figure_of_merit: str, 
                 points: int = 10, best_is_min: bool = True):
    """
    Plot the top configurations minimizing the target column
    """
    filtered_results: list[PlotResult] = [r for r in plot_results if r._protocol == fixed_protocol]

    if len(filtered_results) == 0:
        logger.warning(f"No results found for {fixed_protocol}")
        return
    else:
        logger.info(f"Plotting ({figure_of_merit} / {fixed_protocol}) top configurations minimizing {figure_of_merit}")

    plot_results_obj = PlotResults(filtered_results)
    results = plot_results_obj._results
    
    fom_str: str = f"_{figure_of_merit.lower().replace(' ', '_')}"

    sorted_results = sorted(results, key=lambda r: getattr(r._results, fom_str), reverse=not best_is_min)

    best_results = sorted_results[:points]

    palette = set_palette(type='normal')
    marker_map = {
        BML.ZEROMQ: palette[0],
        BML.NANOMSG: palette[1],
        BML.NNG: palette[2]
    }

    plt.figure(figsize=figsize_bar)

    for r in best_results:
        plt.bar([best_results.index(r)], [getattr(r._results, fom_str)], color=marker_map[r._library])
        
    plt.xticks(range(len(best_results)), [f"{r._workload._payload_length},{r._config._pub_interval},{r._config._subscribers}"
                                            for r in best_results])
    plt.ylabel(figure_of_merit)
    
    if plot_type == 'normal':
        plt.title(f"Top {points} Configurations Optimizing {figure_of_merit} for {upper_camel_case_protocol[fixed_protocol.name]} Communication")

    plt.xlabel(f"{get_metric_plot_info('Payload Length')}, {get_metric_plot_info('Pub Interval')}, {get_metric_plot_info('Num Subscribers')}")
    plt.ylabel(get_metric_plot_info(figure_of_merit))
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend(
        handles=[mpatches.Patch(color=marker_map[l], label=upper_camel_case_bml[l.name.lower()]) for l in set([r._library for r in best_results])],
        title="Library"
    )

    folder = f'{output_dir}/benchmark/{figure_of_merit.lower().replace(" ", "")}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/top_{fixed_protocol.name.lower()}_{points}_min_{figure_of_merit.lower().replace(" ", "")}.{file_format}')
    plt.close()