#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=hello
#SBATCH --output=hello-%j.out
#SBATCH --error=stderr.%j.out
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100M
#SBATCH --partition=atesting --qos=testing

module purge
module load python/3.10

# activate your environment
PROJECT_DIR=/scratch/alpine/jemi2768/NLPProject/
source $PROJECT_DIR/venvs/nlpproject/bin/activate
cd $PROJECT_DIR

python src/main.py

echo "Task complete"