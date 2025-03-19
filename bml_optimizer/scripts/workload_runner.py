from argparse import ArgumentParser
from collections import defaultdict
import csv
import itertools
from typing import Generator, Union
from pprint import pformat

from bml_optimizer.simulator.simulator import Workload, Simulator, SimResults, Configuration, BML, Protocol, parse_bml, parse_protocol, single_test
from bml_optimizer.config.workloads import get_workloads
from bml_optimizer.scripts.bruteforcer import single_test
from bml_optimizer.utils.utils import cleanup, setup
from bml_optimizer.utils.logger import get_logger
from bml_optimizer.scripts.classifier import predict_best_library
from bml_optimizer.plots.plot_lib import plot_workloads

logger = get_logger(__name__)

class WorkloadSimResult:
    """
    For a single step in the simulation, we store the following attributes:
        A. the five parameters in input: 
            (message_count, payload_length, pub_interval, pub_delay, subscribers). 
        B. the library used (for the classifier, the one predicted) 
        C. the current time and figure of merit
    """
    def __init__(self, step: int, 
                 message_count: int, payload_length: int, 
                 pub_interval: int, pub_delay: int, subscribers: int, 
                 library: tuple[str, BML]) -> None:
        self.step = step
        self.message_count = message_count
        self.payload_length = payload_length
        self.pub_interval = pub_interval
        self.pub_delay = pub_delay
        self.subscribers = subscribers
        self.library = library
        self.current_time = None
        self.current_fom = None

    def set_current_time(self, current_time: float) -> None:
        self.current_time = current_time

    def set_current_fom(self, current_fom: float) -> None:
        self.current_fom = current_fom

    def __repr__(self):
        return (f"WorkloadSimResult(message_count={self.message_count}, payload_length={self.payload_length}, pub_interval={self.pub_interval}, "
                f"pub_delay={self.pub_delay}, subscribers={self.subscribers}, library={self.library}, "
                f"current_time={self.current_time}, current_fom={self.current_fom})")


def dump_results_to_csv(results: list[WorkloadSimResult], protocol: Protocol, filename: str) -> None:
    """
    Dumps a list of results objects into a csv file, including the csv header    
    """
    header = [
        "Simulation", "Library", "Protocol", 
        "Messages Sent", "Payload Length", 
        "Pub Delay", "Pub Interval", "Num Subscribers", 
        "Step",
        "Cumulative Time", "Cumulative Avg Latency"
    ]
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        for result in results:
            writer.writerow([
                result.library[0],
                result.library[1].name.lower(), 
                protocol.name.lower(),
                result.message_count, result.payload_length, 
                result.pub_delay, result.pub_interval, result.subscribers, 
                result.step,
                round(result.current_time, 5), round(result.current_fom, 2)
            ])
            

def load_results_from_csv(filename: str) -> tuple[Protocol, list[WorkloadSimResult]]:
    """
    Loads a list of workload results objects from a csv file
    """
    results = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        _ = next(reader, None)  # skip header
        
        for row in reader:
            sim_type = row[0]  # "Fixed" or "Predicted"
            lib_str, protocol_str = row[1], row[2]
            msg_sent, payload_len, pub_delay, pub_interval = \
                int(row[3]), int(row[4]), int(row[5]), int(row[6])
            
            subs, step = int(row[7]), int(row[8])

            library_obj = parse_bml(lib_str)
            protocol_obj = parse_protocol(protocol_str)

            wl_result = WorkloadSimResult(step=step, message_count=msg_sent, payload_length=payload_len,
                                          pub_interval=pub_interval, pub_delay=pub_delay, subscribers=subs,
                                          library=(sim_type, library_obj))
            
            current_time, current_fom = float(row[9]), float(row[10])
            wl_result.set_current_time(current_time)
            wl_result.set_current_fom(current_fom)

            results.append(wl_result)
    
    return protocol_obj, results


def get_current_time(runs_results: list[SimResults]) -> float:
    """
    Returns the cumulative time of the current simulation
    """
    return sum([r.get_attr('time') for r in runs_results])


def get_current_fom(runs_results: list[SimResults]) -> float:
    """
    Returns the value for the current figure of merit
    Currently, it is defined as the average latency considering the past worklaods
    """
    latencies_lists: list[list[float]] = [r.get_attr('latencies') for r in runs_results]
    all_latencies: list[list[float]] = list(itertools.chain(*latencies_lists))
    curr_avg_latency = round(sum(all_latencies) / len(all_latencies), 5) 
    logger.info(f"Average over {len(all_latencies)} latencies: {curr_avg_latency}μs")
    return curr_avg_latency


def single_lib_run(workloads: dict[int, dict[str, Union[Workload, Configuration]]], library: BML, protocol: Protocol, runs: int) -> Generator[tuple[WorkloadSimResult, SimResults], None, None]:
    """
    We return a generator that yields (WorkloadSimResult, SimResults) for each workload.
    This approach is memory efficient and allows streaming of results.
    """
    for i, workload in workloads.items():

        r: tuple[BML, Protocol, Configuration, Workload, SimResults] = \
            single_test(pub_delay=workload['configuration']._pub_delay, 
                        message_count=workload['workload']._message_count, 
                        payload_length=workload['workload']._payload_length, 
                        pub_interval=workload['configuration']._pub_interval, 
                        subscribers=workload['configuration']._subscribers, 
                        library=library, 
                        protocol=protocol, 
                        runs=runs,
                        dump_latencies=True)
        
        wl: WorkloadSimResult = WorkloadSimResult(step=i,
                                message_count=workload['workload']._message_count, 
                                payload_length=workload['workload']._payload_length, 
                                pub_interval=workload['configuration']._pub_interval, 
                                pub_delay=workload['configuration']._pub_delay, 
                                subscribers=workload['configuration']._subscribers, 
                                library=("Fixed", library))
        
        logger.info(f"Step ({i+1}/{len(workloads)}), Library: {library}")        
        yield wl, r[-1]


def optimized_run(workloads: dict[int, dict[str, Union[Workload, Configuration]]], protocol: Protocol, runs: int, model_folder: str) -> Generator[tuple[WorkloadSimResult, SimResults], None, None]:
    """
    We return a generator that yields (WorkloadSimResult, SimResults) for each workload step,
    making it easy to iterate without storing all results at once.
    """
    for i, workload in workloads.items():

        library: BML = predict_best_library(dump_folder=model_folder, protocol=protocol,
                                   messages_sent=workload['workload']._message_count, 
                                   payload_length=workload['workload']._payload_length,
                                   pub_delay=workload['configuration']._pub_delay, 
                                   pub_interval=workload['configuration']._pub_interval, 
                                   num_subscribers=workload['configuration']._subscribers)
               
        r: tuple[BML, Protocol, Configuration, Workload, SimResults] = \
            single_test(pub_delay=workload['configuration']._pub_delay, 
                        message_count=workload['workload']._message_count, 
                        payload_length=workload['workload']._payload_length, 
                        pub_interval=workload['configuration']._pub_interval, 
                        subscribers=workload['configuration']._subscribers, 
                        library=library,
                        protocol=protocol, 
                        runs=runs,
                        dump_latencies=True)
        
        wl: WorkloadSimResult = WorkloadSimResult(step=i,
                                message_count=workload['workload']._message_count, 
                                payload_length=workload['workload']._payload_length, 
                                pub_interval=workload['configuration']._pub_interval, 
                                pub_delay=workload['configuration']._pub_delay, 
                                subscribers=workload['configuration']._subscribers, 
                                library=("Predicted", library))

        logger.info(f"Step ({i+1}/{len(workloads)}), Predicted Best Library: {library}")        
        yield wl, r[-1]


if __name__ == "__main__":
    """
    Run a sequence of workloads with the given parameters
    
    We store all the results in a dictionary, mapping
        Protocol -> [list of WorkloadSimResult objects]
    """
    parser = ArgumentParser(description="Run a sequence of workloads")
    parser.add_argument('--libraries', nargs='+', type=str, default=["zeromq", "nanomsg", "nng"], help='List of libraries to test [zeromq, nanomsg, nng]')
    parser.add_argument('--protocols', nargs='+', type=str, default=["inproc", "ipc", "tcp"], help='List of protocols to test [inproc, ipc, tcp]')
    parser.add_argument('--steps', type=int, default=16, help='Number of different workloads in the simulation')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs for each test')
    parser.add_argument('--random_workload', action='store_true', default=False, help='Use random workloads instead of fixed ones')
    parser.add_argument('--model_folder', type=str, default="model-lat-33k", help='Folder containing the model used for predictions')
                        
    args = parser.parse_args()

    runs: int = args.runs
    libraries: list[BML] = [parse_bml(l) for l in args.libraries]
    protocols: list[Protocol] = [parse_protocol(p) for p in args.protocols]

    workloads: dict[int, dict[str, Union[Workload, Configuration]]] = \
        get_workloads(random=args.random_workload, steps=args.steps)

    logger.info(f"Using the following workloads:\n{pformat(workloads)}")

    with open("results/workloads.log", "a") as f:
        f.write(f"Using the following workloads:\n{pformat(workloads)}\n")
    
    collective_results: dict[Protocol, list[WorkloadSimResult]] = defaultdict(list)

    try:
        setup()
        simulator: Simulator = Simulator()
        
        for protocol in protocols:
            logger.info(f"Investigating {protocol} communication")

            sim_results: list[SimResults] = []
            for (wl, sr) in optimized_run(workloads, protocol, runs, args.model_folder):
                sim_results.append(sr)
                wl.set_current_time(get_current_time(sim_results))
                wl.set_current_fom(get_current_fom(sim_results))
                collective_results[protocol].append(wl)

            for library in libraries:
                sim_results: list[SimResults] = []

                for (wl, sr) in single_lib_run(workloads, library, protocol, runs):
                    sim_results.append(sr)
                    wl.set_current_time(get_current_time(sim_results))
                    wl.set_current_fom(get_current_fom(sim_results))
                    collective_results[protocol].append(wl)

            plot_workloads(collective_results[protocol], protocol)
            dump_results_to_csv(collective_results[protocol], protocol, f"results/workloads_{protocol.name.lower()}.csv")
    
    except KeyboardInterrupt:
        logger.warning("Bruteforce simulation interrupted by user!")
    finally:
        cleanup()