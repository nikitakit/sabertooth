#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

[ -z "$1" ] && die "Usage: run_glue.sh INIT_CHECKPOINT OUTPUT_DIR LR [LR ...]"
INIT_CHECKPOINT=$1
shift
[ -z "$1" ] && die "Usage: run_glue.sh INIT_CHECKPOINT OUTPUT_DIR LR [LR ...]"
OUTPUT_DIR=$1
shift
[ -z "$1" ] && die "Usage: run_glue.sh INIT_CHECKPOINT OUTPUT_DIR LR [LR ...]"
LEARNING_RATES="$@"

echo "INIT_CHECKPOINT=${INIT_CHECKPOINT?}"
echo "OUTPUT_DIR=${OUTPUT_DIR?}"
echo "LEARNING_RATES=${LEARNING_RATES?}"

[ ! -d "${INIT_CHECKPOINT?}" ] && die "Not a checkpoint directory: '${INIT_CHECKPOINT?}'"

set -x
for LR in $LEARNING_RATES
do
for TASK in cola mrpc qqp sst2 stsb mnli qnli rte
do
    mkdir -p "${OUTPUT_DIR?}/lr${LR?}/${TASK?}"
    python3 run_classifier.py \
        --config=configs/classifier.py \
        --config.init_checkpoint="${INIT_CHECKPOINT}" \
        --config.dataset_name="${TASK?}" \
        --config.learning_rate="${LR}" \
        --output_dir="${OUTPUT_DIR?}/lr${LR?}/${TASK?}"
done
done
