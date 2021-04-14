#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

[ -z "$1" ] && die "Usage: tpu.sh create-vm NAME"
TPU_NAME=${TPU_NAME_PREFIX}$1
echo "TPU_NAME=${TPU_NAME?}"
echo "TPU_ZONE=${TPU_ZONE?}"
echo "TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE?}"
echo "TPU_RUNTIME_VERSION=${TPU_RUNTIME_VERSION?}"

# Create a TPU node
exec gcloud alpha compute tpus tpu-vm create ${TPU_NAME?} \
    --zone ${TPU_ZONE?} \
    --accelerator-type ${TPU_ACCELERATOR_TYPE?} \
    --version ${TPU_RUNTIME_VERSION?}
