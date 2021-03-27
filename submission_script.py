#!/usr/bin/env python

import os


def mkdir_p(dir):
    os.makedirs(dir, exist_ok=True)
job_directory = f"{os.environ['HOME']}/job"

data_dir = os.path.join(os.environ['HOME'], '/project/mel-cnn')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)

job_file = os.path.join(job_directory, 'test.job')
with open(job_file) as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH --job-name=test.job\n")
    fh.writelines("#SBATCH --output=.out/test.out\n")
    fh.writelines("#SBATCH --error=.out/test.err\n")
    fh.writelines("#SBATCH --time=2-00:00\n")
    fh.writelines("#SBATCH --mem=12000\n")
    fh.writelines("#SBATCH --qos=normal\n")
    fh.writelines("#SBATCH --mail-type=ALL\n")
    fh.writelines("#SBATCH --mail-user=$USER@stanford.edu\n")
    fh.writelines("run ...")
os.system("sbatch %s" % job_file)
