#!/bin/bash

sudo apt-get update

# ZeroMQ:
sudo apt-get install libzmq3-dev libczmq-dev

# NanoMsg
sudo apt-get install libnanomsg-dev

# NNG
NNG_DIR=nng
sudo apt-get install cmake ninja-build libmbedtls-dev

mkdir -p $NNG_DIR && cd $NNG_DIR
wget https://github.com/nanomsg/nng/archive/refs/tags/v1.9.0.tar.gz
tar -xzf v1.9.0.tar.gz
cd nng-1.9.0
mkdir build && cd build
cmake -G Ninja ..
ninja
sudo ninja install
cd ../../..
sudo rm -r $NNG_DIR