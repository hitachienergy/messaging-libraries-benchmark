import math
import os
from num2tex import num2tex, configure as num2tex_configure

from bml_optimizer.simulator.simulator import Protocol
from bml_optimizer.plots.plot_utils import set_palette, get_plot_config, upper_camel_case_protocol
from bml_optimizer.utils.logger import get_logger

import matplotlib.pyplot as plt

from scipy.optimize import OptimizeResult
from skopt.plots import plot_objective, plot_convergence

logger = get_logger(__name__)
set_palette(type='normal')
plot_type = 'normal'
formatter, figsize_bar, figsize_scatter, figsize_optimal, figsize_gp, file_format, output_dir = get_plot_config(plot_type)
num2tex_configure(exp_format='cdot', help_text=False)


def get_messages_from_bytes(bytes_to_send: int, payload_length: int) -> int:
    """
    Compute the number of messages to send given the number of bytes to send and the payload length.
    """
    return math.ceil(bytes_to_send / payload_length)


"""
Plot functions for Gaussian Process optimization
"""

def produce_gp_plots(result: OptimizeResult, bytes_to_send: int,
                        max_time: int, gp_shots: int, gp_initial_points: int, runs: int,
                        protocol: Protocol, pub_delay: int, num_subscribers: int, 
                        constraints, plot_id: str = ""):
    """
    Produce plots for the Gaussian Process optimization.
    """
    folder = f'{output_dir}/optimize'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if plot_type == 'normal':
        fig = plt.figure(figsize=figsize_gp[0])

        ax_conv = fig.add_subplot(1, 2, 1)
        ax_obj = fig.add_subplot(1, 2, 2)
        
        plot_convergence(result, ax=ax_conv)
        plot_objective(result, ax=ax_obj)

        fig.suptitle(f"Results of Bayesian optimization with {runs} Evaluations for each {gp_initial_points} Initial Points and {gp_shots} Total Shots\n"
                        f"{upper_camel_case_protocol[protocol.name.lower()]} Communication, {num_subscribers} subscribers\n"
                        f"Max. Time ${num2tex(max_time, precision=2)}$μs, Max. CPU {constraints.cpu_max}%, Max. Memory {constraints.mem_max}%\n" 
                        f"Best result: ${num2tex(result.fun, precision=2)}$ - {result.x[2]} - "
                        f"sending ${num2tex(get_messages_from_bytes(bytes_to_send, int(result.x[0])), precision=2)}$ messages of ${num2tex(result.x[0], precision=2)}$B every ${num2tex(result.x[1], precision=2)}$μs")

        plt.subplots_adjust(top=0.78, wspace=0.25)
        plt.tight_layout()

        fig.savefig(f"{folder}/gp_plots{plot_id}.{file_format}")
        plt.close(fig)
    
    else:
        fig_conv = plt.figure(figsize=figsize_gp[0])
        ax_conv = fig_conv.add_subplot(1, 1, 1)
        plot_convergence(result, ax=ax_conv)
        ax_conv.xaxis.set_major_formatter(formatter)
        ax_conv.yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        fig_conv.savefig(f"{folder}/gp_convergence{plot_id}.{file_format}")
        plt.close(fig_conv)

        fig_obj = plt.figure(figsize=figsize_gp[1])
        ax_obj = fig_obj.add_subplot(1, 1, 1)
        plot_objective(result, ax=ax_obj)
        plt.tight_layout()
        fig_obj.savefig(f"{folder}/gp_objective{plot_id}.{file_format}")
        plt.close(fig_obj)