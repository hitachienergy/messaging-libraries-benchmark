import csv
import os
import signal
from bml_optimizer.utils.logger import get_logger
import enum
import subprocess as sp
import shortuuid
import json
import psutil
import time 
import statistics
import struct

logger = get_logger(__name__)

class BML(enum.Enum):
    ZEROMQ = 1
    NANOMSG = 2
    NNG = 3

class Protocol(enum.Enum):
    INPROC = 1
    IPC = 2
    TCP = 3

def parse_bml(library: str) -> BML:
    """
    Also handles short names for the libraries
    """
    try:
        if library == "0MQ":
            return BML.ZEROMQ
        elif library == "NMsg":
            return BML.NANOMSG
        elif library == "NNG":
            return BML.NNG
        else:
            return BML[library.upper()]
    except KeyError:
        raise ValueError(f"Invalid BML value: {library}. Allowed: {[m.name for m in BML]}")


def sort_bml(bml_list: list[BML]) -> list[BML]:
    """
    Sort a list of BML enum members by their integer values.
    """
    return sorted(bml_list, key=lambda bml: bml.value)


def parse_protocol(value: str) -> Protocol:
    try:
        return Protocol[value.upper()]
    except KeyError:
        raise ValueError(f"Invalid Protocol value: {value}. Allowed: {[p.name for p in Protocol]}")

class Configuration:
    def __init__(self, pub_interval: int, pub_delay: int, subscribers: int):
        self._pub_interval = pub_interval
        self._pub_delay = pub_delay
        self._subscribers = subscribers

    def __repr__(self):
        return (f"Configuration(pub_interval={self._pub_interval}, "
                f"pub_delay={self._pub_delay}, "
                f"subscribers={self._subscribers})")

class Workload:
    def __init__(self, message_count: int, payload_length: int):
        self._message_count = message_count
        self._payload_length = payload_length

    def __repr__(self):
        return (f"Workload(message_count={self._message_count}, "
                f"payload_length={self._payload_length})")

class SimResults:
    def __init__(self, message_received: int, time: float, 
                 throughput: float, payload_throughput: float,
                 min_latency: int, avg_latency: int, max_latency: int, 
                 p90_latency: int, p99_latency: int, mean_jitter: int, 
                 median_cpu: float = -1, median_mem: float = -1,
                 latencies_list: list[float] = []):
        self._message_received = message_received
        self._time = time
        self._throughput = throughput
        self._payload_throughput = payload_throughput
        self._min_latency = min_latency
        self._avg_latency = avg_latency
        self._max_latency = max_latency
        self._p90_latency = p90_latency
        self._p99_latency = p99_latency
        self._mean_jitter = mean_jitter
        self._median_cpu = median_cpu
        self._median_mem = median_mem
        self._latencies = latencies_list

        self._data_repr = {"_message_received": self._message_received, "_time": self._time, 
                            "_throughput": self._throughput, "_payload_throughput": self._payload_throughput,
                            "_min_latency": self._min_latency, "_avg_latency": self._avg_latency, "_max_latency": self._max_latency,
                            "_p90_latency": self._p90_latency, "_p99_latency": self._p99_latency, "_mean_jitter": self._mean_jitter,
                            "_median_cpu": self._median_cpu, "_median_mem": self._median_mem}

    def get_dict(self):
        return self._data_repr

    def get_attr(self, attr: str):
        if not hasattr(self, f"_{attr}"): raise AttributeError(f"Attribute {attr} not found")
        return getattr(self, f"_{attr}")

    def __repr__(self):
        return json.dumps(self._data_repr, indent=2)

    def __str__(self):
        return json.dumps(self._data_repr, indent=2)


def get_json_data(json_file: str) -> dict:
    """
    Read a JSON file and return a dictionary with the data.
    Returns None if the JSON contains an "error" key.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            return data if "error" not in data else None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading JSON file {json_file}: {e}")
        return None
    

def get_binary_data(binary_file: str) -> list[float]:
    """
    Read a binary file, 
        containing the dumped list of latencies (uint64_t) 
        and return a list with the unsigned long long elements.
    Returns None if the file does not exist.
    """
    element_size = 8
    latencies: list[float] = []
    try:
        with open(binary_file, "rb") as f:
            while True:
                data = f.read(element_size)
                if not data: break
                latencies.append(struct.unpack('Q', data)[0])
        return latencies                

    except (FileNotFoundError) as e:
        return []


def single_test(pub_delay: int, message_count: int, subscribers: int, protocol: Protocol, pub_interval: int, 
                payload_length: int, library: BML, runs: int,
                dump_latencies: bool = False) -> tuple[BML, Protocol, Configuration, Workload, SimResults]:
    """
    Run a single test with the given parameters
    The parameters follow the order of the function call
    Returns the average results and dumps them on the results file
    """
    simulator: Simulator = Simulator()
    config: Configuration = Configuration(pub_interval=pub_interval, pub_delay=pub_delay, subscribers=subscribers)
    workload: Workload = Workload(message_count=message_count, payload_length=payload_length)
    
    simulator.setup_sim(library=library, protocol=protocol, config=config, workload=workload, dump_latencies=dump_latencies)
    results: list[SimResults] = simulator.run_sim(runs=runs)
    
    simulator.dump_results(results)
    avg_results: SimResults = simulator.average_results(results)
    return (library, protocol, config, workload, avg_results)


class Simulator:
    _library: BML
    _protocol: Protocol
    _config: Configuration
    _workload: Workload
    _runs: int
    _port_counter: int
    _real_time_cpus: str = "2,3"
    _desired_usages_samples: int = 10
    _max_time: int
    _dump_latencies: bool = True

    TASKSET_CMD: list[str] = ["taskset", "-ac", _real_time_cpus]
    BINARY_PATH: str = "./bml/bml-pub-sub"
    DATA_PATH: str = "./data"
    RESULTS_PATH: str = "./results"

    def __init__(self):
        """
        Initialize the simulator
        The port counter is used to assign a port to the TCP protocol
        """
        self._port_counter: int = 5555
        pass


    def setup_sim(self, library: BML, protocol: Protocol, config: Configuration, workload: Workload, max_time: int = None, dump_latencies: bool = False) -> None:
        """
        Set the simulation parameters
        max_time refers to the maximum time the simulation should run (in microseconds)
            does not include the publisher delay 
            if set, the simulation will run for at most that time
        """
        self._library = library
        self._protocol = protocol
        self._config = config
        self._workload = workload
        self._max_time = max_time
        self._dump_latencies = dump_latencies


    def average_latencies_lists(self, latencies_lists: list[list[float]]) -> list[float]:
        """
        Return a list with the average of the latencies lists using index-based averaging
            for missing elements, the average is calculated without them
        """
        if not latencies_lists: 
            logger.warning("No latencies lists to average")
            return []
        if len(latencies_lists) == 1: return latencies_lists[0]
        
        max_len = max([len(latency_list) for latency_list in latencies_lists])
        latencies_list = []

        for i in range(max_len):
            latencies = [latency_list[i] for latency_list in latencies_lists if i < len(latency_list)]
            latencies_list.append(round(statistics.mean(latencies), 5))

        return latencies_list


    def average_results(self, results: list[SimResults]) -> SimResults:
        """
        Return a single SimResults object averaging all the attributes of the given results
        """
        if not results: raise ValueError("No results to average")
        avg_results = {
                            k: round(sum([getattr(r, k) for r in results]) / len(results), 5)
                            for k in results[0].get_dict().keys()
        }

        if self._dump_latencies and any([result.get_attr("latencies") for result in results]):
            avg_results["_latencies"] = self.average_latencies_lists(
                [result.get_attr("latencies") for result in results])
            
        return SimResults(*avg_results.values())


    def dump_results(self, results: list[SimResults]) -> None:
        """
        Dump the results to a file
        """
        avg_results: SimResults = self.average_results(results)
        row = {
            "library": self._library.name.lower(),
            "protocol": self._protocol.name.lower(),
            "message_count": self._workload._message_count,
            "payload_length": self._workload._payload_length,
            "pub_delay": self._config._pub_delay,
            "pub_interval": self._config._pub_interval,
            "subscribers": self._config._subscribers,
            "message_received": avg_results.get_attr("message_received"),
            "time": avg_results.get_attr("time"),
            "throughput": avg_results.get_attr("throughput"),
            "payload_throughput": avg_results.get_attr("payload_throughput"),
            "min_latency": avg_results.get_attr("min_latency"),
            "avg_latency": avg_results.get_attr("avg_latency"),
            "p90_latency": avg_results.get_attr("p90_latency"),
            "p99_latency": avg_results.get_attr("p99_latency"),
            "max_latency": avg_results.get_attr("max_latency"),
            "mean_jitter": avg_results.get_attr("mean_jitter"),
            "median_cpu": avg_results.get_attr("median_cpu"),
            "median_mem": avg_results.get_attr("median_mem")
        }
        file_path = f"{self.RESULTS_PATH}/results.csv"
        file_exists = os.path.exists(file_path)
        try:
            with open(file_path, "a", newline="") as csvfile:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing results to file: {e}")
            logger.error(f"Make sure to setup the results file {self.RESULTS_PATH}/results.csv before running the simulation")

    
    def get_sim_results(self, data: dict, 
                        median_cpu: float, median_mem: float, 
                        latencies_list: list[float] = None) -> SimResults:
        """
        Get the SimResults object from the given data
        """
        if not data: return None
        return SimResults(**data, median_cpu=median_cpu, median_mem=median_mem, latencies_list=latencies_list)


    def get_sub_id(self) -> str:
        """
        Return a random shortuuid combined with the library and protocol names
        """
        return f"{self._library.name.lower()}_{self._protocol.name.lower()}_{shortuuid.uuid()}"


    def check_termination(self, processes: list[sp.Popen], start_time: float) -> bool:
        """
        Return True if the simulation should be terminated
            and send termination signals to the processes
        """
        if self._max_time is not None:
            elapsed = (time.time_ns() - start_time) / 10**3 # convert ns to microseconds
            if elapsed >= self._max_time:
                logger.info("Max simulation time exceeded. Sending SIGUSR1 to remaining processes.")
                for process in processes:
                    if process.poll() is None: os.kill(process.pid, signal.SIGUSR1)
                return True
        return False


    def collect_cpu_memory_usage(self, processes: list[sp.Popen]) -> tuple[float, float]:
        """
        Return median CPU and Mem usages

        For the memory, we collect the "uss" (Unique Set Size): This is the amount of memory that is unique to the process and not shared with any other processes. USS is a good indicator of the memory usage that is solely attributable to the process.

        While the pub and subs are running, we capture the CPU and memory usages
            poll() checks if the process is finished, to know when to stop collecting

        We set interval to None to return an instantaneous measurement relative to the last time it was called. 
        Because of this, the first call you make usually returns 0.0 (or a cached value), and you need to make a second call—after some duration—to get a meaningful result. 
        
        The typical pattern is:
            Call cpu_percent(interval=None) and discard the result. This "primes" psutil's internal sampling.
            Wait for a specific interval.
            Call cpu_percent(interval=None) again to get an accurate percentage for that period.
        This effectively simulates the same behavior as interval=N but without blocking the current thread for N seconds.

        We collect the usages of the different processes and store them in lists
        We return the median of the samples for the aggregated CPU and memory usages

        New! If max_time is not None, we should stop the process after max_time
                -> set a timer of max_time microseconds
                -> after max_time, stop all processes by sending "kill -SIGUSR1 <pid>" for each pid
        """
        # derive the expected time and the sampling interval
        expected_time: float = (self._config._pub_interval * self._workload._message_count / 10**6)
        MIN_SAMPLING_INT: float = 0.1
        MAX_SAMPLING_INT: float = 2.0
        sampling_interval: float = min(max((expected_time / self._desired_usages_samples), MIN_SAMPLING_INT), MAX_SAMPLING_INT)
        if expected_time < 0.1 and self._config._pub_interval > 0:
            logger.warning(f"Expected time is {expected_time} seconds. "
                            "The simulation may not produce usage data.")
        else:
            logger.debug(f"Expected time: {expected_time}, Sampling interval: {sampling_interval}")

        # initialize the monitor and the lists to store the usages
        monitors: list[psutil.Process] = [psutil.Process(pid=process.pid) for process in processes]
        
        # discard the first sample, sleep for the pub. delay...
        for monitor in monitors: monitor.cpu_percent(interval=None)
        time.sleep(self._config._pub_delay / 10**3)

        cpu_samples: list[float] = []
        mem_samples: list[float] = []

        start_time: float = time.time_ns()
        # ...and then start collecting the cpu and memory usages
        while any([process.poll() is None for process in processes]):
            time.sleep(sampling_interval)

            # checking if the simulation should be terminated
            if self.check_termination(processes, start_time): break

            # adding up the cpu and memory usages of the different processes
            cpu_sample, mem_sample = 0.0, 0.0
            for idx, (monitor, process) in enumerate(zip(monitors, processes)):
                if process.poll() is None:
                    process_cpu, process_mem = monitor.cpu_percent(interval=None), monitor.memory_percent(memtype="uss")
                    cpu_sample += process_cpu
                    mem_sample += process_mem
            
            logger.debug(f"CPU/Mem Usages: ( {cpu_sample} / {mem_sample} )")
            if cpu_sample > 0.0: cpu_samples.append(cpu_sample)
            if mem_sample > 0.0: mem_samples.append(mem_sample)

        # return the median of the samples
        if not cpu_samples: cpu_samples.append(0.0)
        if not mem_samples: mem_samples.append(0.0)
        return statistics.median(cpu_samples), statistics.median(mem_samples)


    def run_inproc(self, sub_ids: list[str]) -> tuple[float, float]:
        """
        Run the simulation with the inproc protocol
        Each subscriber will write to a file with an identifier
        """
        endpoint: str = f"inproc://{shortuuid.uuid()[0:8]}"
        num_subs: int = self._config._subscribers

        cmd: list[str] = [
            *self.TASKSET_CMD,
            self.BINARY_PATH,
            self._library.name.lower(),
            "--pub",
            *(f"--sub" for _ in range(num_subs)),
            "--sub_ids",
            *sub_ids,
            "--rate", f"{self._config._pub_interval}",
            "--delay", f"{self._config._pub_delay}",
            "--count", f"{self._workload._message_count}",
            "--dp-len", f"{self._workload._payload_length}",
            "--endpoint", endpoint
        ]
        if self._dump_latencies: cmd.append("--dump_latencies")

        logger.debug(f"Running command {' '.join(cmd)}")
        process = sp.Popen(cmd, stdout=sp.DEVNULL)
        logger.debug(f"Pub and subs running with pid {process.pid}")
        median_cpu, median_mem = self.collect_cpu_memory_usage([process])
        process.wait()
        return median_cpu, median_mem


    def run_processes(self, endpoint: str, num_subs: int, sub_ids: list[str]) -> tuple[float, float]:
        """
        Start one process for the publisher and one for each subscriber (with the respective id)
        """
        commands: list[list[str]] = []

        # a single process for the publisher with all the parameters
        commands.append([
            *self.TASKSET_CMD,
            self.BINARY_PATH,
            self._library.name.lower(),
            "--pub",
            "--rate", f"{self._config._pub_interval}",
            "--delay", f"{self._config._pub_delay}",
            "--count", f"{self._workload._message_count}",
            "--dp-len", f"{self._workload._payload_length}",
            "--endpoint", endpoint
        ])

        # num_sub processes for the subscribers with the respective id
        assert num_subs == len(sub_ids)
        for sub_id in sub_ids:
            cmd: list[str] = [
                *self.TASKSET_CMD,
                self.BINARY_PATH,
                self._library.name.lower(),
                "--sub",
                "--sub_ids", sub_id,
                "--count", f"{self._workload._message_count}",
                "--endpoint", endpoint
            ]
            if self._dump_latencies: cmd.append("--dump_latencies")
            commands.append(cmd)

        # run all the processes
        processes: list[sp.Popen] = []
        for cmd in commands:
            logger.debug(f"Running command {' '.join(cmd)}")
            process = sp.Popen(cmd, stdout=sp.DEVNULL)
            pid: int = process.pid
            logger.debug(f"Process running with pid {pid}")
            processes.append(process)

        # collect the cpu and memory usage
        median_cpu, median_mem = self.collect_cpu_memory_usage(processes)

        # wait for all the processes to finish
        for process in processes:
            process.wait()

        return median_cpu, median_mem
    

    def run_ipc(self, sub_ids: list[str]) -> tuple[float, float]:
        """
        Run the simulation with the ipc protocol
        Runs pub and subs on different processes: one for the publisher (--pub) and g.e. than one for the subscriber (--sub)
        """
        endpoint: str = f"ipc:///tmp/{shortuuid.uuid()[0:8]}"
        num_subs: int = self._config._subscribers
        median_cpu, median_mem = self.run_processes(endpoint, num_subs, sub_ids)
        return median_cpu, median_mem


    def run_tcp(self, sub_ids: list[str]):
        """
        Run the simulation with the tcp protocol
        Runs two processes: one for the publisher (--pub) and one for the subscriber (--sub)
        """
        self._port_counter += 1
        endpoint = "tcp://127.0.0.1:" + str(self._port_counter)
        num_subs: int = self._config._subscribers
        median_cpu, median_mem = self.run_processes(endpoint, num_subs, sub_ids)
        return median_cpu, median_mem
    

    def run_sim(self, runs: int) -> list[SimResults]:
        """
        Run the simulation with the given parameters
        The endpoint is a string '<transport>://<address>, based on the protocol
            If the protocol is inproc, the endpoint is a random string (the sim shortuuid)
            If the protocol is ipc, the endpoint is a pathname in /tmp
            If the protocol is tcp, the endpoint is a tcp address
        Here, we average the results of each subscriber, and return a list of the averaged results (one for each run)
        """
        all_results: list[SimResults] = []
        for run in range(runs):
            """
            Each iteration is a single run,
                it produces a list of sub_ids (one for each subscriber)
                which will be used to identify the subscribers' results
                which are averaged and stored in the all_results list
            """
            logger.debug(f"Running simulation ({run+1}/{runs})")
            sub_ids: list[str] = [self.get_sub_id() for _ in range(self._config._subscribers)]

            if self._protocol == Protocol.INPROC:
                median_cpu, median_mem = self.run_inproc(sub_ids)
            elif self._protocol == Protocol.IPC:
                median_cpu, median_mem = self.run_ipc(sub_ids)
            elif self._protocol == Protocol.TCP:
                median_cpu, median_mem = self.run_tcp(sub_ids)
            else:
                raise ValueError(f"Invalid protocol: {self._protocol}")

            subscribers_data: list[SimResults] = []

            for id in sub_ids:
                metrics_data: dict = get_json_data(f"{self.DATA_PATH}/{id}.json")
                latencies_list: list[float] = get_binary_data(f"{self.DATA_PATH}/{id}.bin")

                if self._dump_latencies and not latencies_list:
                    logger.warning(f"Warning: run aborted, subscriber {id} did not yield latencies")
                    break

                elif not self._dump_latencies and latencies_list:
                    logger.warning(f"Warning: latencies were dumped without the flag")

                sub_results: SimResults = self.get_sim_results(data=metrics_data,
                                                               latencies_list=latencies_list,
                                                               median_cpu=median_cpu, 
                                                               median_mem=median_mem)
                if not sub_results:
                    logger.warning(f"Warning: run aborted, subscriber {id} did not yield a result")
                    break
                subscribers_data.append(sub_results)
            else:            
                run_data_averaged: SimResults = self.average_results(subscribers_data)
                all_results.append(run_data_averaged)

        return all_results