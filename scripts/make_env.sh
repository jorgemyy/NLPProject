#!/bin/bash

# load python
module purge
module load python/3.10

# define paths
PROJECT_DIR=/scratch/alpine/jemi2768/NLPProject
VENV_DIR=$PROJECT_DIR/venvs
cd $PROJECT_DIR

# create virtual environment
# if it already exists, remove it first
if [ -d "$VENV_DIR" ]; then
    echo "Removing old virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

echo "Creating new virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

# activate venv, install requirements
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -e .

echo "Virtual environment setup complete at $VENV_DIR"