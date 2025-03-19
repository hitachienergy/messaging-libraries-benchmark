import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from bml_optimizer.config.plots import PLOT_SETTINGS

from copy import deepcopy
import colorsys

from bml_optimizer.utils.logger import get_logger
logger = get_logger(__name__)

"""
Plot settings and utils; paper plots:
- have specific dimensions, proportional to the paper linewidth
- do not have a title
- are saved as a .pdf in a different folder
"""

def set_palette(type: str = "normal"):
    """
    Set and return the color palette for the plots
        - Pair palette for comparing different configs for the same lib (e.g. zcp vs. no-zcp)
        - Normal palette for comparing libraries (zeromq vs nanomsg vs nng)
    """
    basic_palette = sns.color_palette("hls", 4)
    basic_palette[2], basic_palette[3] = basic_palette[3], 'black' # consistent with old plots

    if type == "normal":
        sns.set_palette(basic_palette)

    elif type == "pair":
        pair_palette = []
        base_colors = basic_palette
        for i in range(len(base_colors)):
            base_col = deepcopy(base_colors[i])
            pair_palette.append(base_col)
            h, l, s = colorsys.rgb_to_hls(*base_col)
            pair_palette.append(colorsys.hls_to_rgb(h, l*0.8, s))
            sns.set_palette(pair_palette)
    else:
        raise ValueError(f"Invalid palette type: {type}. Allowed: 'normal' or 'pair'")

    return basic_palette if type == "normal" else pair_palette

"""
Plot settings and utils; paper plots:
- have specific dimensions, proportional to the paper linewidth
- do not have a title
- are saved as a .pdf in a different folder
"""

def get_plot_config(plot_mode: str = 'normal'):
    """
    Possible modes:
    - paper: for paper plots
    - normal: for normal plots
    """
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    formatter.set_scientific(True)

    plt.rcParams['savefig.dpi'] = PLOT_SETTINGS[plot_mode]['dpi']

    figsize_bar = PLOT_SETTINGS[plot_mode]['figsize_bar']
    figsize_scatter = PLOT_SETTINGS[plot_mode]['figsize_scatter']
    figsize_optimal = PLOT_SETTINGS[plot_mode]['figsize_optimal']
    figsize_gp = PLOT_SETTINGS[plot_mode]['figsize_gp']
    file_format = PLOT_SETTINGS[plot_mode]['format']
    output_dir = PLOT_SETTINGS[plot_mode]['output_dir']

    return formatter, figsize_bar, figsize_scatter, figsize_optimal, figsize_gp, file_format, output_dir

"""
Useful dictionaries for the plots
"""

contourf_cmaps = {
    'zeromq': 'rocket', 
    'nanomsg': 'mako', 
    'nng': 'viridis', 
    'optimized': 'cividis'
}

"""
For each unit, it maps 
    .csv name -> (plots name, unit)
The unit is None if the value is unitless
"""
fom_dict: dict[str, tuple[str, str]] = {
    'Payload Length': ('Message Size', 'bytes'),
    'Messages Sent': ('Sent Messages', None),
    'Pub Interval': ('Pub Interval', 'μs'),
    'Total Time': ('Time', 's'),
    'Messages Received': ('Received Messages', None),
    'Throughput': ('Throughput', 'msg/s'),
    'Payload Throughput': ('Throughput', 'MB/s'),
    'Min Latency': ('Minimum Latency', 'ns'),
    'Avg Latency': ('Average Latency', 'ns'),
    'P90 Latency': ('P90 Latency', 'ns'),
    'P99 Latency': ('P99 Latency', 'ns'),
    'Max Latency': ('Maximum Latency', 'ns'),
    'Mean Jitter': ('Mean Jitter', 'ns'),
    'Dev Latency': ('Latency Deviation', 'ns'),
    'Median CPU': ('Median CPU Usage Percentage', None),
    'Median MEM': ('Median Memory Usage Percentage', None),
    'Num Publishers': ('Number of Publishers', None),
    'Num Subscribers': ('Number of Subscribers', None),
    'Steps': ('Steps', None),
}

def get_metric_plot_info(metric: str) -> str:
    """
    Get the plot name and the unit for the metric as a string
        "{plot name} ({unit})"
    """
    try:
        return f"{fom_dict[metric][0]} ({fom_dict[metric][1]})" if fom_dict[metric][1] is not None else fom_dict[metric][0]
    except KeyError:
        logger.error(f"Invalid metric: {metric}")
        return metric

upper_camel_case_bml = {
    "zeromq": "ZeroMQ",
    "nanomsg": "NanoMsg",
    "nng": "NNG",
    "ZEROMQ": "ZeroMQ",
    "NANOMSG": "NanoMsg",
    "NNG": "NNG"
}

upper_camel_case_protocol = {
    "inproc": "In-Process",
    "ipc": "Inter-Process",
    "tcp": "TCP",
    "INPROC": "In-Process",
    "IPC": "Inter-Process",
    "TCP": "TCP"
}