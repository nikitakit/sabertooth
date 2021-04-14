#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

HOST=$1
[ -z "$HOST" ] && die "Usage: tpu.sh provision NAME"
shift

REPO_NAME=$(git rev-parse --show-toplevel | xargs basename)
[ -z "$REPO_NAME" ] && die "Failed to find root of git repository"

echo "HOST=${HOST}"
echo "REPO_NAME=${REPO_NAME}"
cd $(dirname $0)
scp ${TPU_INSECURE_PUBLIC_KEY?} $HOST:~/.ssh/id_rsa.pub
scp ${TPU_INSECURE_PRIVATE_KEY?} $HOST:~/.ssh/id_rsa
scp ${TPU_GIT_CONFIG?} $HOST:~/.gitconfig
ssh "$HOST" -- "mkdir ${REPO_NAME} && cd ${REPO_NAME} && git init"
git remote get-url ${HOST} >/dev/null 2>/dev/null || git remote add ${HOST} ${HOST}:~/${REPO_NAME}/
git remote set-url ${HOST} ${HOST}:~/${REPO_NAME}/
git remote set-url --push ${HOST} ${HOST}:~/${REPO_NAME}/
git push ${HOST} HEAD:pushbranch

cat setup.sh | ssh "$HOST" -- "cat >setup.sh && bash setup.sh"

if [ -e "${TPU_POST_SETUP_SCRIPT}" ]; then
    cat "${TPU_POST_SETUP_SCRIPT}" | ssh "$HOST" -- "cat >post_setup.sh && bash post_setup.sh"
fi
