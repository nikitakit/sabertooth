#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

cmd="$1"
[ -z "$cmd" ] && die "Usage: tpu.sh <command> NAME"
shift

cd $(dirname $0)

# Load tpu management configuration from config.env
[ -f "./config.env" ] || die "config.env not found"
set -a
. ./config.env
set +a

# Call the actual command
exec ./tpu-$cmd.sh "$@"
