#!/bin/bash
set -e
set -u

echo "-- Job start $(date) --"
echo "Running on host: $(hostname)"

tar -xzf mhx_env.tar.gz

# Set paths
PY_VER="python3.10"
export PATH=$PWD/bin:${PATH:-}
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PYTHONPATH=$PWD/lib/$PY_VER/site-packages:${PYTHONPATH:-}
for nv_dir in $PWD/lib/$PY_VER/site-packages/nvidia/*/lib; do
    if [ -d "$nv_dir" ]; then
        export LD_LIBRARY_PATH="$nv_dir:${LD_LIBRARY_PATH}"
    fi
done

export FONTCONFIG_PATH=/etc/fonts
export JAX_ENABLE_X64=1

echo "Python location: $(which python3)"
python3 -m pip list

echo "Import test"
python3 -c "
import numpy as np
import matplotlib.pyplot as plt
import jax
import mhx
print('Success')
"

python3 -u "$1"
echo "--- JOB FINISHED: $(date) ---"
