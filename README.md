# Introduction 
This project benchmarks various messaging libraries to compare their performance in different communication scenarios. 

# Required Libraries

You can either use ```install_bml.sh``` or follow these instructions.

ZeroMQ:
```
sudo apt-get install libzmq3-dev && sudo apt-get install libczmq-dev
```

NanoMsg:
```
sudo apt-get install libnanomsg-dev
```

NanoMsg Next Generation:
```
sudo apt-get install cmake ninja-build libmbedtls-dev
mkdir -p nng && cd nng
wget https://github.com/nanomsg/nng/archive/refs/tags/v1.9.0.tar.gz
tar -xzf v1.9.0.tar.gz
cd nng-1.9.0
mkdir build && cd build
cmake -G Ninja ..
ninja
sudo ninja install
cd ../../..
sudo rm -r nng
```


# Compilation

To compile the project, just ```make```, or ```make -C bml``` from the root of the project.

# Running the Program
```
Usage: ./bml/bml-pub-sub <zeromq|nanomsg|nng> [OPTIONS]
Examples:
  In-process:           ./bml/bml-pub-sub zeromq --log --sub --sub --sub_ids sub1 sub2 --pub --count 100 --rate 500 --dp-len 100 --delay 1000 --endpoint inproc://benchmark
  Inter-process:
    Publisher:  ./bml/bml-pub-sub zeromq --log --pub --count 100 --rate 500 --dp-len 100 --delay 1000 --endpoint ipc://127.0.0.1:5555
    Subscriber: ./bml/bml-pub-sub zeromq --log --sub --sub_ids sub1 --count 100 --endpoint ipc://127.0.0.1:5555

  --help                         Show this menu
  -z, --conf <file>              Use config file for options (config has priority)
  -l, --log                      Enable logging (debugging)
  -c, --count <number>           Total number of messages to be published (default: 2147483647)
  -f, --rate <microseconds>      Publishing time interval in microseconds (default: 0)
  -d, --dp-len <length>          Length of the datapoint (payload) (default: 100)
  -y, --delay <milliseconds>     Initial delay before publisher starts (default: 0)
  -e, --endpoint <endpoint>      Endpoint for publisher and subscriber (default: inproc://benchmark)
  --pub                          Run as publisher (inter-process communication only)
  --sub                          Run as subscriber (inter-process communication only)
  --sub_ids <id1> <id2> ...      Provide IDs for each subscriber if multiple subs are used
```

# Benchmarking Suite (Paper)

Required to run all the components of the suite, you need to install python packages via pip:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

To reproduce the plots of the paper by plotting the numerics in `results/results.csv`, you can run:
```
 python3 -m bml_optimizer.plots.paper_plots --use_log_scale
```

To run your own full simulation (as the one described in the paper)
```
 chmod +x ./results/runner.sh
 ./results/runner.sh
```
Mind that you will write/overwrite `results/results.csv`.

To run a custom simulation, you can run:
```
python3 -m bml_optimizer.scripts.bruteforcer
```

with custom parameters:
```
usage: bruteforcer.py [-h] [--libraries LIBRARIES [LIBRARIES ...]] [--protocols PROTOCOLS [PROTOCOLS ...]] [--pub_intervals PUB_INTERVALS [PUB_INTERVALS ...]] [--pub_delays PUB_DELAYS [PUB_DELAYS ...]]
                      [--subscribers SUBSCRIBERS [SUBSCRIBERS ...]] [--message_counts MESSAGE_COUNTS [MESSAGE_COUNTS ...]] [--payload_lengths PAYLOAD_LENGTHS [PAYLOAD_LENGTHS ...]] [--runs RUNS]

Bruteforce the simulator

options:
  -h, --help            show this help message and exit
  --libraries LIBRARIES [LIBRARIES ...]
                        List of libraries to test [zeromq, nanomsg, nng]
  --protocols PROTOCOLS [PROTOCOLS ...]
                        List of protocols to test [inproc, ipc, tcp]
  --pub_intervals PUB_INTERVALS [PUB_INTERVALS ...]
                        List of publication intervals to test
  --pub_delays PUB_DELAYS [PUB_DELAYS ...]
                        List of publication delays to test
  --subscribers SUBSCRIBERS [SUBSCRIBERS ...]
                        List of subscribers to test
  --message_counts MESSAGE_COUNTS [MESSAGE_COUNTS ...]
                        List of message counts to test
  --payload_lengths PAYLOAD_LENGTHS [PAYLOAD_LENGTHS ...]
                        List of payload lengths to test
  --runs RUNS           Number of runs for each test
  ```