# Fruit Quality Check Service

This is the final project of Hanna Kondrashova at the University of London.

This repository contains an HTTP service that is capable of processing pictures
and outputting the coordinates of several types of objects:
- Rotten apple
- Good apple
- Storage containing a large amount of apples that are difficult to distinguish

## Prerequisites

To run the backend application perform the following steps:

1. Setup CUDA >= 11.0 following [the official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
Ultimately, you need an AWS or Google Cloud instance with a GPU having ~6GB of memory to run the service. Inference on CPU is much slower, but it can still be used for applications that do not have strict latency restrictions.

2. Clone the repository:

```bash
$ git clone git@github.com:hannakond/QualityCheck.git
```

After that, clone submodules and LFS objects:

```bash
$ git lfs pull
$ git submodule update
```

3. Activate virtual environment. Under Ubuntu, run:

```bash
$ sudo apt-get install virtualenv
$ virtualenv venv
$ source venv/bin/activate
```

It is assumed that Python >= 3.8 is installed in the system. Otherwise, you need to specify python to `virtualenv`:

```bash
$ virtualenv --python=<path to python >= 3.8> venv
```

4. Using activated virtual environment, run:

```bash
$(venv) pip install -r requirements.txt
$(venv) pip install -r yolov5/requirements.txt
```

## Run service

Inside virtual environment:

```bash
$(venv) python main.py
```

## Run test visualization test

Inside virtual environment:

```bash
$(venv) python test_viz.py
```