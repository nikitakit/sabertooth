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
POD_NAME=${POD_NAME_PREFIX}$1
echo "POD_NAME=${POD_NAME?}"
echo "POD_ZONE=${POD_ZONE?}"

# Delete a TPU node
exec gcloud alpha compute tpus tpu-vm delete ${POD_NAME?} --zone ${POD_ZONE?}
