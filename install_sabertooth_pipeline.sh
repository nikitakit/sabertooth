#!/bin/bash
set -e

cd "$(dirname $0)/rust/sabertooth_pipeline"

# In a virtual environment or conda environment, "maturin develop" is enough
# to build the data pipeline. But with system python (like on TPU VMs), we
# have to build a wheel and install it.
RUSTFLAGS="-Ctarget-cpu=native" maturin build --release
pip uninstall -y sabertooth_pipeline

# maturin will build wheels for all detected python versions
# Iterate through them until a supported configuration is found
ls target/wheels/sabertooth_pipeline-*.whl | while read wheel;
do
    pip install "$wheel" || continue
    echo "Installed wheel: $wheel"
    break
done
