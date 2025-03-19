from bml_optimizer.simulator.simulator import Configuration, Workload
from typing import Union
import random as rand

fixed_subscribers: int = 1

def get_workloads(random: bool = False, steps: int = 16) -> dict[int, dict[str, Union[Workload, Configuration]]]:
    """
    Returns a dictionary of workloads with the given number of steps. If random is set to True, the workloads will be generated randomly.
    """
    if random:
        random_param_list: list[tuple[int, int, int, int, int]] = [
            (
                5000,                           # message_count
                rand.randint(1, 20),            # payload_length power-of-two
                rand.randint(0, 1000),          # pub_interval
                1000,                           # pub_delay
                fixed_subscribers
            )
            for _ in range(steps)
        ]
        
        workloads: dict[int, dict[str, Union[Workload, Configuration]]] = {
            i: {
                'workload': Workload(message_count=mc, payload_length=(2**pl)),
                'configuration': Configuration(pub_interval=pi, pub_delay=pd, subscribers=s)
            }
            for i, (mc, pl, pi, pd, s) in enumerate(random_param_list)
        }

        return workloads
    
    else:
        param_list: list[tuple[int, int, int, int, int]] = \
        [
            # (count,          length,      rate,      delay,  subscribers).
            (5000 - i * 200,   2**(i+4),    i * 50,    1000,   fixed_subscribers)
            for i in range(steps)
        ]

        workloads: dict[int, dict[str, Union[Workload, Configuration]]] = {
            i: {
                'workload': Workload(message_count=mc, payload_length=pl),
                'configuration': Configuration(pub_interval=pi, pub_delay=pd, subscribers=s)
            }
            for i, (mc, pl, pi, pd, s) in enumerate(param_list)
        }
        return workloads