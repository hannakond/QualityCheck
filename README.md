# Fruit Quality Check Service

This is the final project of Hanna Kondrashova at the University of London.

This repository contains HTTP sevice that is capable to process pictures and
output the coordinates of several types of objects:
- rotten apple,
- good apple,
- storage containing big amount of apples that are difficult to distinguish.

# Prerequisites

To run the backend application perform the following steps:

1. Setup CUDA >= 11.0 following [official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

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

# Run service

Inside virtual environment:

```bash
$(venv) python main.py
```

# Run test with visualization
