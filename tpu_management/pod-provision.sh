#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

HOST_PREFIX=$1
[ -z "$HOST_PREFIX" ] && die "Usage: pod.sh provision NAME"
shift

REPO_NAME=$(git rev-parse --show-toplevel | xargs basename)
[ -z "$REPO_NAME" ] && die "Failed to find root of git repository"

echo "HOST_PREFIX=${HOST_PREFIX}"
echo "REPO_NAME=${REPO_NAME}"
cd $(dirname $0)

for WORKER in 0 1 2 3
do
scp ${TPU_INSECURE_PUBLIC_KEY?} ${HOST_PREFIX}${WORKER}:~/.ssh/id_rsa.pub
scp ${TPU_INSECURE_PRIVATE_KEY?} ${HOST_PREFIX}${WORKER}:~/.ssh/id_rsa
scp ${TPU_GIT_CONFIG?} ${HOST_PREFIX}${WORKER}:~/.gitconfig
ssh "${HOST_PREFIX}${WORKER}" -- "mkdir ${REPO_NAME} && cd ${REPO_NAME} && git init"
done

git remote get-url ${HOST_PREFIX} >/dev/null 2>/dev/null && git remote remove ${HOST_PREFIX}
git remote add ${HOST_PREFIX} ${HOST_PREFIX}0:~/${REPO_NAME}/
git remote set-url --add --push ${HOST_PREFIX} ${HOST_PREFIX}0:~/${REPO_NAME}/
git remote set-url --add --push ${HOST_PREFIX} ${HOST_PREFIX}1:~/${REPO_NAME}/
git remote set-url --add --push ${HOST_PREFIX} ${HOST_PREFIX}2:~/${REPO_NAME}/
git remote set-url --add --push ${HOST_PREFIX} ${HOST_PREFIX}3:~/${REPO_NAME}/
git push -f ${HOST_PREFIX} HEAD:pushbranch

for WORKER in 0 1 2 3
do
cat <<"EOF" | ssh "${HOST_PREFIX}${WORKER}" -- "sudo tee /usr/local/lib/python3.6/dist-packages/jax_pod_setup.py > /dev/null"
import os
import requests

def get_metadata(key):
  return requests.get(
      'http://metadata.google.internal/computeMetadata'
      '/v1/instance/attributes/{}'.format(key),
      headers={
          'Metadata-Flavor': 'Google'
      }).text

worker_id = get_metadata('agent-worker-number')
accelerator_type = get_metadata('accelerator-type')
worker_network_endpoints = get_metadata('worker-network-endpoints')

os.environ['CLOUD_TPU_TASK_ID'] = worker_id
os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'

accelerator_type_to_host_bounds = {
    'v2-8': '1,1,1',
    'v2-32': '2,2,1',
    'v2-128': '4,4,1',
    'v2-256': '4,8,1',
    'v2-512': '8,8,1',
    'v3-8': '1,1,1',
    'v3-32': '2,2,1',
    'v3-128': '4,4,1',
    'v3-256': '4,8,1',
    'v3-512': '8,8,1',
    'v3-1024': '8,16,1',
    'v3-2048': '16,16,1',
}

os.environ['TPU_HOST_BOUNDS'] = accelerator_type_to_host_bounds[
    accelerator_type]
os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = worker_network_endpoints.split(
    ',')[0].split(':')[2] + ':8476'
os.environ['TPU_MESH_CONTROLLER_PORT'] = '8476'
EOF

cat <<"EOF" | ssh "${HOST_PREFIX}${WORKER}" -- "cat > ~/jax_pod_setup.sh"
#!/bin/bash
python3 -c 'import os, jax_pod_setup
for var in [
    "CLOUD_TPU_TASK_ID",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_MESH_CONTROLLER_ADDRESS",
    "TPU_MESH_CONTROLLER_PORT",
    ]:
  print(f"export {var}={os.environ[var]}")'
EOF
done

for WORKER in 0 1 2 3
do
cat setup.sh | ssh "${HOST_PREFIX}${WORKER}" -- "cat >setup.sh && bash setup.sh"
done

if [ -e "${TPU_POST_SETUP_SCRIPT}" ]; then
for WORKER in 0 1 2 3
do
cat "${TPU_POST_SETUP_SCRIPT}" | ssh "${HOST_PREFIX}${WORKER}" -- "cat >post_setup.sh && bash post_setup.sh"
done
fi
