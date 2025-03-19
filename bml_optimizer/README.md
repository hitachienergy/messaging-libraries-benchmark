# BML Optimizer

```
└── bml
|   ├── pub-sub.c
|   ├── bml-pub-sub
|   ├── (other scripts for running the BML)
└── bml_optimizer
|   ├── simulator
|       ├── __init__.py
|       └── simulator.py
|   ├── scripts
|       ├── __init__.py
|       ├── bruteforcer.py
|       └── optimizer.py
|   ├── tests
|       ├── __init__.py
|       └── test_bml.py
|   ├── plots
|       ├── __init__.py
|       └── plot_lib.py
|   └── README.md
└── data
|   ├── zeromq_ipc_q2GuELaXisGr8atCHqEWmh.json
|   ├── (other simulation result files)
└── requirements.txt
```

## Overview
This package provides tools and scripts to run simulations and optimizations on messaging libraries.  
Use the `simulator` module to run tests, and the scripts under `scripts` to bruteforce or optimize over the libraries. 

## Tests
The `tests` module provides `pytest` tests for all the other modules.

Install requirements, then run:
```
pytest -v
```

## Plots
The `plots` module provides functionality to parse and plot simulation results. The main classes used are `PlotResult` and `PlotResults`.

`PlotResult` represents a single simulation result with its configuration and workload, whereas PlotResults is a collection of `PlotResult` objects that provides a method to convert the results to a pandas DataFrame for plotting.

Plot settings such as DPI, figure sizes, and file formats are configured in a separate configuration file. There are two sets of parameters: one for normal plotting in PNG format and one for paper-quality plots in PDF format.
