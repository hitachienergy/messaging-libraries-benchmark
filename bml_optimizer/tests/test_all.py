import pytest
import os
import pandas as pd

from bml_optimizer.plots.paper_plots import produce_benchmark_plots
from bml_optimizer.simulator.simulator import Workload, Simulator, SimResults, Configuration, BML, Protocol
from bml_optimizer.scripts.bruteforcer import bruteforce
from bml_optimizer.scripts.optimizer import gaussian_optimization, dump_dir, Constraints
from bml_optimizer.utils.utils import cleanup, setup
from bml_optimizer.utils.logger import get_logger
from bml_optimizer.scripts.classifier import transform_csv

logger = get_logger(__name__)

baseline = {
    'library': BML.ZEROMQ,
    'protocol': Protocol.INPROC,
    'pub_interval': 500,
    'pub_delay': 1000,
    'subscribers': 1,
    'message_count': 5000,
    'payload_length': 100,
    'runs': 2
}

def get_baseline(avoid: list[str] = []) -> dict:
    return {k: v for k, v in baseline.items() if k not in avoid}

def setup_function():
    setup()

def teardown_function():
    cleanup()

def wrapper_simulation(library, protocol, pub_interval, pub_delay, subscribers, message_count, payload_length, runs,
                       dump_latencies=False):
    """
    Wrapper function to run the simulator
    """
    simulator: Simulator = Simulator()
    config: Configuration = Configuration(pub_interval=pub_interval, pub_delay=pub_delay, subscribers=subscribers)
    workload: Workload = Workload(message_count=message_count, payload_length=payload_length)

    simulator.setup_sim(library=library, protocol=protocol, config=config, workload=workload, dump_latencies=dump_latencies)
    
    results: list[SimResults] = simulator.run_sim(runs=runs)
    assert results is not None and len(results) <= runs
    if len(results) < runs:
        logger.warning("Some runs have failed")

@pytest.mark.parametrize("runs", [2, 5])
def test_simulator(runs):
    """
    Single test to check if the simulator runs without errors using the baseline configuration
    """
    wrapper_simulation(**(get_baseline(avoid=['runs'])), runs=runs)

def test_dump_latencies():
    bin_files_before = [f for f in os.listdir("data") if f.endswith(".bin")]
    wrapper_simulation(**get_baseline(avoid=['runs']), runs=1, dump_latencies=True)
    bin_files_after = [f for f in os.listdir("data") if f.endswith(".bin")]
    logger.debug(f"Before: {bin_files_before}, After: {bin_files_after}")
    assert len(bin_files_after) == len(bin_files_before) + 1, "One more .bin file should be created."

@pytest.mark.parametrize("pub_interval", [0, 500])
@pytest.mark.parametrize("pub_delay", [0, 1000])
@pytest.mark.parametrize("message_count", [100, 5000])
@pytest.mark.parametrize("payload_length", [24, 10000])
def test_configurations(pub_interval, pub_delay, message_count, payload_length):
    """
    Single test to check if the simulator runs without errors using different configurations
    """
    wrapper_simulation(**get_baseline(avoid=['pub_interval', 'pub_delay', 'message_count', 'payload_length']), 
                       pub_interval=pub_interval, pub_delay=pub_delay, message_count=message_count, payload_length=payload_length)

@pytest.mark.parametrize("protocol", [Protocol.INPROC, Protocol.IPC, Protocol.TCP])
@pytest.mark.parametrize("subscribers", [3])
def test_multiple_sub(protocol: Protocol, subscribers: int):
    """
    Single test to check if the simulator runs without errors using multiple subscribers
    """
    wrapper_simulation(**get_baseline(avoid=['protocol', 'subscribers']), protocol=protocol, subscribers=subscribers)


def test_bruteforcer():
    """
    Check if the bruteforcer runs without errors
    """
    for r in bruteforce(libraries=[BML.ZEROMQ, BML.NANOMSG, BML.NNG], protocols=[Protocol.INPROC, Protocol.IPC, Protocol.TCP], pub_intervals=[500], pub_delays=[1000], num_subscribers=[1], 
               message_counts=[5000], payload_lengths=[100], runs=2):
        logger.info(f"Generator produced {r}")


def test_optimizer():
    """
    Check if the optimizer runs without errors
    """
    simulator: Simulator = Simulator()
    constraints: Constraints = Constraints(
        cpu_max=20, cpu_weight=1.0,
        mem_max=0.1, mem_weight=1.0
    )

    if not os.path.exists(dump_dir): os.makedirs(dump_dir)
    initial_files = len(os.listdir(dump_dir))

    gaussian_optimization(simulator=simulator, bytes_to_send=10**6, gp_shots=3, gp_initial_points=2, runs=2, protocol=Protocol.INPROC,
                          pub_delay=1000, num_subscribers=1, constraints=constraints, latency_weight=1.0,  max_time=10**7, store_results=True)

    final_files = len(os.listdir(dump_dir))
    assert final_files - initial_files == 1, "Exactly one new result file should be created."


def test_classifier():
    """
    Test the classifier
    """
    df = pd.DataFrame({
        "Avg Latency": [10, 5, 15, 20, 25, 30],
        "Throughput":  [10, 5, 15, 20, 25, 30],
        "Library": ["zeromq", "nanomsg", "nng", "zeromq", "nanomsg", "nng"],
        'Protocol': ["inproc"]*6,
        'Messages Sent': [5000]*6,
        'Payload Length': [100]*6,
        'Pub Delay': [1000]*6,
        'Pub Interval': [500]*6,
        'Num Subscribers': [1]*6
    })
    result = transform_csv(df, "Avg Latency")
    assert result["Best Library"].iloc[0] == "nanomsg", "The best library should be nanomsg."
    result = transform_csv(df, "Throughput")
    assert result["Best Library"].iloc[0] == "nng", "The best library should be nanomsg."


def test_plot_lib():
    """
    Check if the plotting runs without errors
    """
    produce_benchmark_plots(results_file="results/results.csv", plots_to_produce=["pub-int-0"])
    assert os.path.exists("figures-normal") or os.path.exists("figures-paper"), "Paper plots should be created."