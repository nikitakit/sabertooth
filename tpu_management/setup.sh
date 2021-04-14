#!/bin/bash
# Setup script for TPU VMs
# "tpu.sh provision" and "pod.sh provision" will run this file on the TPU VMs

# Set up PATH
export PATH="$HOME/.local/bin:$PATH"

# Set up SSH across TPU vms
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Install key dependencies
pushd sabertooth && git checkout pushbranch && git checkout -b master && popd
pip3 install --user --upgrade -r sabertooth/requirements_tpu.txt

# Install Flax
git clone https://github.com/google/flax.git
pip install --user -e flax

# Install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install cmake (required by rust sentencepiece bindings)
sudo apt install -y cmake

# Build the pre-training input pipeline
pushd sabertooth
./install_sabertooth_pipeline.sh
popd

# JAX pod setup helper (only run in a multi-host TPU setup)
# Running the alias "jax_pod_setup" in all worker VMs will cause subsequent
# JAX-based programs to use all workers for multi-host training.
if [ -f ./jax_pod_setup.sh ]; then
    chmod +x ./jax_pod_setup.sh
    printf 'alias jax_pod_setup="eval \\\"$(~/jax_pod_setup.sh)\\\""' >> ~/.bash_aliases
fi

