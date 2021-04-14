#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

[ -z "$1" ] && die "Usage: tpu.sh delete NAME"
TPU_NAME=${TPU_NAME_PREFIX}$1
echo "TPU_NAME=${TPU_NAME?}"
echo "TPU_ZONE=${TPU_ZONE?}"

# Delete a TPU node
exec gcloud alpha compute tpus tpu-vm delete ${TPU_NAME?} --zone ${TPU_ZONE?}
