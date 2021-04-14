#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

[ -z "$1" ] && die "Usage: tpu.sh create NAME"

./tpu.sh create-vm "$@"
sleep 5
./tpu.sh config-ssh "$@"
