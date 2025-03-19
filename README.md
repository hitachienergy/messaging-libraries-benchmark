# Introduction 
This project benchmarks various messaging libraries to compare their performance in different communication scenarios. 

In the following, we state our goals for the implementation.

### Libraries
- **ZeroMQ** [done]
- **NanoMsg** [done]
- **NNG** [done]

### Transports
- **In-process** communication (inproc) [done]
- **Inter-process** communication (ipc) [done]
- **Inter-device** communication (tcp) [done]

### Features
- **Buffering** [done]
- **Polling** [done, not in NNG]
- **Zero-copy** [done, but inefficient]

### Metrics Collected
- **Latency** [done]
- **Throughput** [done]
- **Messages lost** [done]
- **CPU and memory usage** [done]


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

# BML Optimizer

TODO: add description and link to the suite

Required to run all the components of the suite, you need to install python packages via pip:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```
