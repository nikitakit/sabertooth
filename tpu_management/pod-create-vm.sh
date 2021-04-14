#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

[ -z "$1" ] && die "Usage: pod.sh create-vm NAME"
POD_NAME=${POD_NAME_PREFIX}$1
echo "POD_NAME=${POD_NAME?}"
echo "POD_ZONE=${POD_ZONE?}"
echo "POD_ACCELERATOR_TYPE=${POD_ACCELERATOR_TYPE?}"
echo "POD_RUNTIME_VERSION=${POD_RUNTIME_VERSION?}"

# Create a TPU node
exec gcloud alpha compute tpus tpu-vm create ${POD_NAME?} \
    --zone ${POD_ZONE?} \
    --accelerator-type ${POD_ACCELERATOR_TYPE?} \
    --version ${POD_RUNTIME_VERSION?}
