from itertools import product
from argparse import ArgumentParser
from bml_optimizer.simulator.simulator import Workload, SimResults, Configuration, BML, Protocol, parse_bml, parse_protocol, single_test
from bml_optimizer.utils.utils import cleanup, setup
from bml_optimizer.utils.logger import get_logger
from bml_optimizer.plots.plot_lib import PlotResult
from typing import Generator

logger = get_logger(__name__)


def bruteforce(libraries: list[BML], protocols: list[Protocol], 
               pub_intervals: list[int], pub_delays: list[int], 
               num_subscribers: list[int],
               message_counts: list[int], payload_lengths: list[int], 
               runs: int) -> Generator[tuple[BML, Protocol, Configuration, Workload, SimResults], None, None]:
    """
    Run a brute-force simulation with the given parameters
    Returns the average results for each combination
    """
    num_combs = len(pub_delays) * len(message_counts) * len(num_subscribers) * len(protocols) * len(pub_intervals) * len(payload_lengths) * len(libraries)
    for idx, comb in enumerate(product(pub_delays, message_counts, num_subscribers, protocols, pub_intervals, payload_lengths, libraries)):
        logger.info(f"Testing combination ({idx+1}/{num_combs}): {comb}")
        try:
            yield single_test(*comb, runs=runs, dump_latencies=False)
        except Exception as e:
            logger.error(e)
            continue

if __name__ == "__main__":
    """
    Run a brute-force simulation with the given parameters
    """
    parser = ArgumentParser(description="Bruteforce the simulator")
    parser.add_argument('--libraries', nargs='+', type=str, default=["zeromq", "nanomsg", "nng"], help='List of libraries to test [zeromq, nanomsg, nng]')
    parser.add_argument('--protocols', nargs='+', type=str, default=["inproc", "ipc", "tcp"], help='List of protocols to test [inproc, ipc, tcp]')
    parser.add_argument('--pub_intervals', nargs='+', type=int, default=[0, 500], help='List of publication intervals to test')
    parser.add_argument('--pub_delays', nargs='+', type=int, default=[1000], help='List of publication delays to test')
    parser.add_argument('--subscribers', nargs='+', type=int, default=[1], help='List of subscribers to test')
    parser.add_argument('--message_counts', nargs='+', type=int, default=[5000], help='List of message counts to test')
    parser.add_argument('--payload_lengths', nargs='+', type=int, default=[1000*i for i in range(160, 201, 20)], 
                        help='List of payload lengths to test')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs for each test')

    args = parser.parse_args()

    runs: int = args.runs
    message_counts: list[int] = args.message_counts
    payload_lengths: list[int] = args.payload_lengths
    num_subscribers: list[int] = args.subscribers
    pub_delays: list[int] = args.pub_delays
    pub_intervals: list[int] = args.pub_intervals
    libraries: list[BML] = [parse_bml(l) for l in args.libraries]
    protocols: list[Protocol] = [parse_protocol(p) for p in args.protocols]

    try:
        setup()
        results = []
        for r in bruteforce(libraries, protocols, pub_intervals, pub_delays, num_subscribers, message_counts, payload_lengths, runs):
            logger.info(r)
            results.append(PlotResult(*r))
        
    except KeyboardInterrupt:
        logger.warning("Bruteforce simulation interrupted by user!")
    finally:
        cleanup()