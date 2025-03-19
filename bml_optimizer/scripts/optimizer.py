import math
import os
import shortuuid
from typing import Optional

from bml_optimizer.simulator.simulator import Workload, Simulator, SimResults, Configuration, BML, parse_bml, Protocol, parse_protocol
from bml_optimizer.utils.utils import cleanup, setup
from bml_optimizer.utils.logger import get_logger
from bml_optimizer.plots.plot_gp import produce_gp_plots, get_messages_from_bytes

from argparse import ArgumentParser

from scipy.optimize import OptimizeResult
from skopt import gp_minimize, dump, load
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args

logger = get_logger(__name__)
dump_dir = "./results/optimize_dump"


def get_percentage_difference(a: float, b: float) -> float:
    """
    Return the percentage difference between a and b.
    """
    return 100 * math.fabs(a - b) / b


class Constraints:
    """
    Type to store the constraints for the optimization.
    """
    def __init__(self,  cpu_max: float, cpu_weight: float, 
                        mem_max: float, mem_weight: float):
        self.cpu_max: float = cpu_max
        self.cpu_weight: float = cpu_weight
        self.mem_max: float = mem_max
        self.mem_weight: float = mem_weight

    def get_attr(self, attr: str):
        if not hasattr(self, f"_{attr}"): raise AttributeError(f"Attribute {attr} not found")
        return getattr(self, f"_{attr}")

    def get_penalty(self, results: SimResults) -> float:
        """
        Returns a percentage of the penalty for the given results.
        Considering the different runs, we take the highest median value for all the constrained metrics.
        We give a penalty, in percentage, equivalent to 
            - the percentage difference between the highest median value and the constraint,
            - multiplied by the weight of the constraint.

        Formally, in this scoring approach, we consider the average latency, denoted by \(\Lambda(x)\), and add penalties for violating constraints. Formally, the score is:
        \[
        f(x) = \omega_{l}\,\Lambda(x) + \Bigl(\omega_{c}\,\rho_{c} + \omega_{m}\,\rho_{m}\Bigr)\,\Lambda(x),
        \]
        where \(\rho_{c}\) and \(\rho_{m}\) are penalty ratios for CPU and memory, and \(\omega_{l}, \omega_{c}, \omega_{m}\) are their respective weights.
        """
        penalty: float = 0
        logger.info( "Computing penalty percentage "
                    f"for constraints (cpu, mem): ({self.cpu_max}%, {self.mem_max}%), "
                    f"and weights ({self.cpu_weight}, {self.mem_weight})")

        sim_cpu_max = max([r.get_attr("median_cpu") for r in results])  
        if sim_cpu_max > self.cpu_max:
            diff = get_percentage_difference(sim_cpu_max, self.cpu_max)
            penalty += self.cpu_weight * diff
            logger.info(f"CPU constraint exceeded: {sim_cpu_max}% vs. {self.cpu_max}% -> {diff}%")

        sim_mem_max = max([r.get_attr("median_mem") for r in results])  
        if sim_mem_max > self.mem_max:
            diff = get_percentage_difference(sim_mem_max, self.mem_max) 
            penalty += self.mem_weight * diff
            logger.info(f"Memory constraint exceeded: {sim_mem_max}% vs. {self.mem_max}% -> {diff}%")
        
        logger.info(f"Percentage of penalty: {penalty:.2f}%")
        return penalty

    def __str__(self):
        return (f"Constraints("
                f"cpu_max={self.cpu_max}, "
                f"cpu_weight={self.cpu_weight}, "
                f"mem_max={self.mem_max}, "
                f"mem_weight={self.mem_weight})")
    
    def get_short_str(self):
        return f"{self.cpu_max}_{self.cpu_weight}_{self.mem_max}_{self.mem_weight}"


def get_score(simulator: Simulator, results: list[SimResults], latency_weight: float, 
                max_time: float, constraints: Constraints) -> Optional[float]:
    """
    Return the score of the simulation.
    """
    try: 
        avg_results: SimResults = simulator.average_results(results)
        latency: float = avg_results.get_attr("avg_latency")
        penalty_percentage: float = constraints.get_penalty(results)
        penalty: float = penalty_percentage * latency / 100

        time_taken: float = avg_results.get_attr("time") * 10**6 # microseconds
        logger.info(f"Simulation took {time_taken:.2e} microseconds")
        if time_taken  > max_time:
            logger.info(f"Max. time exceed (took {time_taken} microseconds), returning null score")
            return 1e10
        
        logger.info(f"Average latency: {latency:.2e}ns, penalty: {penalty:.2e}ns")
        return (latency_weight * latency) + penalty
    
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(f"Yielding null score")
        return 1e10

def run_simulation(simulator: Simulator, library: BML, protocol: Protocol, 
                    pub_interval: int, pub_delay: int, subscribers: int,
                    message_count: int, payload_length: int, runs: int) -> list[SimResults]:
    """
    Run a single simulation with the given parameters.
    """
    simulator.setup_sim(library=library, protocol=protocol, 
                        config=Configuration(pub_interval=pub_interval, pub_delay=pub_delay, subscribers=subscribers),
                        workload=Workload(message_count=message_count, payload_length=payload_length))
    
    results: list[SimResults] = simulator.run_sim(runs=runs)
    return results


def objective_latency_score(space, simulator: Simulator, bytes_to_send: int, max_time: float,
                            shot_count, gp_shots: int, runs: int, 
                            protocol: Protocol, pub_delay: int, num_subscribers: int, 
                            constraints: Constraints, latency_weight: float) -> float:
    """
    Objective function, consider the whole space of actions,
        returning the score of the single simulation.
    """
    shot_count[0] += 1

    library_short: str = space["Library"]
    library: BML = parse_bml(library_short)

    payload_length: int = space["Payload Length"]
    pub_interval: int = space["Publishing Interval"]

    message_count = get_messages_from_bytes(bytes_to_send, payload_length)
    
    parameters = {
        "library": library,
        "protocol": protocol,
        "pub_interval": pub_interval,
        "pub_delay": pub_delay,
        "subscribers": num_subscribers,
        "message_count": message_count,
        "payload_length": payload_length,
        "runs": runs
    }
    
    logger.info(f"Shot {shot_count[0]}/{gp_shots}, using "
                f"Library: {library_short}, "  
                f"Payload length: {payload_length}B, "
                f"Publishing interval: {pub_interval}us, "
                f"Message count: {message_count}")
    
    sim_results: list[SimResults] = run_simulation(simulator, **parameters)
    score = get_score(simulator, sim_results, latency_weight, max_time, constraints)

    if score is None:
        raise Exception("Error in simulation")
    return score


def log_current_best(result: OptimizeResult, bytes_to_send: int):
    payload_length: int = int(result.x[0])
    message_count: int = get_messages_from_bytes(bytes_to_send, payload_length)
    pub_interval: int = result.x[1]
    library: str = result.x[2]
    score: float = result.fun
    logger.info(f"Current best setup: Library: {library}, Payload length: {payload_length}B, Message count: {message_count}, Publishing interval: {pub_interval}us")
    logger.info(f"Current best result: {score:.2e}\n")
    

def optimization_callback(bytes_to_send: int):
    """
    Callback function to do something during the optimization.
    Maybe it will be needed to (i) store some intermediate result or (ii) do some runtime plotting (iii) stop the sim. if some condition is met.
    """
    return lambda result: log_current_best(result, bytes_to_send)


def dump_gp_results(result: OptimizeResult,
                    bytes_to_send: int,
                    max_time: int,
                    gp_shots: int,
                    gp_initial_points: int,
                    runs: int,
                    protocol: Protocol,
                    pub_delay: int,
                    num_subscribers: int,
                    constraints: Constraints) -> None:
    """
    Dump the results of the Gaussian Process optimization.    
    """
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    short_str = constraints.get_short_str()
    filename = (f"{dump_dir}/"
                f"gp_results_{bytes_to_send}_{max_time}_{protocol.name}_{gp_shots}_"
                f"{gp_initial_points}_{runs}_{pub_delay}_{num_subscribers}_{short_str}_{shortuuid.uuid()[0:8]}.pkl")
    logger.info(f"Storing results to {filename}")

    del result.specs['args']['func']
    del result.specs['args']['callback']
    dump(result, filename, store_objective=False)


def load_gp_results(filename: str) -> OptimizeResult:
    """
    Load the results of the Gaussian Process optimization.
    """
    return load(filename)


def gaussian_optimization(simulator: Simulator, bytes_to_send: int,
                          gp_shots: int, gp_initial_points: int, runs: int, 
                          protocol: Protocol, pub_delay: int, num_subscribers: int, 
                          constraints: Constraints, latency_weight: float,
                          max_time: int,
                          store_results: bool = True) -> None:
    """
    This function is used to run the Gaussian Process optimization.
    """
    logger.warning(f"Bytes to send: {bytes_to_send:.2e}B")
    logger.warning(f"Gaussian process with {gp_shots} evaluations "
                f"and {gp_initial_points} initial points\n")

    min_length: int = 25
    max_length: int = bytes_to_send // 10**4
    max_interval: int = max_time // (bytes_to_send // min_length)

    logger.warning(f"Investigating input space: "
                   f"Payload length: [{min_length}, {max_length}]B, "
                   f"Message count: [{bytes_to_send // max_length}, {bytes_to_send // min_length}], "
                   f"Publishing interval: [0, {max_interval}]us")

    space = [
        Integer(min_length, max_length, name='Payload Length'), # bytes 
        Integer(0, max_interval, name='Publishing Interval'), # microseconds
        Categorical(["0MQ", "NMsg", "NNG"], name='Library'),
    ]
    shot_count = [0]
    
    @use_named_args(space)
    def wrapped_objective(**space_params):
        return objective_latency_score(
            space=space_params,
            simulator=simulator,
            bytes_to_send=bytes_to_send, 
            max_time=max_time,
            shot_count=shot_count,
            gp_shots=gp_shots,
            runs=runs,
            protocol=protocol,
            pub_delay=pub_delay,
            num_subscribers=num_subscribers,
            constraints=constraints,
            latency_weight=latency_weight
        )

    result: OptimizeResult = gp_minimize(
        wrapped_objective,
        space,
        n_calls=gp_shots,
        n_initial_points=gp_initial_points,
        callback=[optimization_callback(bytes_to_send)],
        acq_func='gp_hedge',
        noise="gaussian",
    )
    
    if store_results:
        dump_gp_results(result, bytes_to_send, max_time, gp_shots, gp_initial_points, runs, protocol, pub_delay, num_subscribers, constraints)
        produce_gp_plots(result=result, bytes_to_send=bytes_to_send, max_time=max_time, gp_shots=gp_shots, gp_initial_points=gp_initial_points, runs=runs, 
                         protocol=protocol, pub_delay=pub_delay, num_subscribers=num_subscribers, constraints=constraints)

    return result


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--bytes_to_send", type=int, default=10**6, help="Number of bytes to send")

    parser.add_argument("--gp_shots", type=int, default=100, help=("Number of shots for Gaussian Process optimization"))
    parser.add_argument("--gp_initial_points", type=int, default=10, help=("Number of initial points for Gaussian Process optimization"))
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for each simulation")

    parser.add_argument("--protocol", type=str, default="ipc", help="Protocol to use (inproc, ipc, tcp)")
    parser.add_argument('--pub_delay', type=int, default=1000, help='Delay of the publisher')
    parser.add_argument("--num_subscribers", type=int, default=1, help="Number of subscribers listening")

    parser.add_argument("--max_time", type=int, default=10**7, help="Constraint for the maximum time for the simulation (in microseconds)")
    
    parser.add_argument("--cpu_max", type=float, default=20, help="Constraint for the maximum CPU usage")
    parser.add_argument("--mem_max", type=float, default=0.05, help="Constraint for the maximum memory usage")

    parser.add_argument("--latency_weight", type=float, default=1.0, help=("Weight applied to average latency in scoring"))
    parser.add_argument("--cpu_weight", type=float, default=1.0, help="Weight for the CPU constraint")
    parser.add_argument("--mem_weight", type=float, default=1.0, help="Weight for the memory constraint")

    args = parser.parse_args()
    simulator: Simulator = Simulator()

    bytes_to_send: int = args.bytes_to_send
    max_time: int = args.max_time
    gp_shots, gp_initial_points = args.gp_shots, args.gp_initial_points
    runs: int = args.runs

    protocol: Protocol = parse_protocol(args.protocol)
    pub_delay: int = args.pub_delay
    num_subscribers: int = args.num_subscribers

    constraints: Constraints = Constraints(
        cpu_max=args.cpu_max, cpu_weight=args.cpu_weight,
        mem_max=args.mem_max, mem_weight=args.mem_weight
    )
    latency_weight: float = args.latency_weight

    try:
        setup()
        gaussian_optimization(
            simulator=simulator,
            bytes_to_send=bytes_to_send, 
            gp_shots=gp_shots, 
            gp_initial_points=gp_initial_points, 
            runs=runs, 
            protocol=protocol, 
            pub_delay=pub_delay, 
            num_subscribers=num_subscribers, 
            constraints=constraints,
            latency_weight=latency_weight,
            max_time=max_time
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user!")
    finally:
        cleanup()