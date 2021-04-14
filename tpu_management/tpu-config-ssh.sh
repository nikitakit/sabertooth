#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

[ -z "$1" ] && die "Usage: tpu.sh config-ssh NAME"

TPU_NAME=${TPU_NAME_PREFIX}$1
HOST=$1
echo "TPU_NAME=${TPU_NAME?}"
echo "HOST=${HOST?}"
echo "TPU_ZONE=${TPU_ZONE?}"

echo "Extracting SSH arguments from the gcloud tool..."
SSH_COMMAND=$(gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME?} --zone ${TPU_ZONE?} --dry-run)
echo ${SSH_COMMAND?}
IP_ADDR=$(echo ${SSH_COMMAND} | cut -d "@" -f 2)
VM_USER=$(echo ${SSH_COMMAND} | grep -o '[^\s]*@' | xargs | cut -d '@' -f 1)
IDENTITY_FILE=$(echo ${SSH_COMMAND} | grep -o -- '-i.* -o' | cut -d ' ' -f 2)
KNOWN_HOSTS_FILE=$(echo ${SSH_COMMAND} | grep -o -- 'UserKnownHostsFile=.*' | cut -d ' ' -f 1 | cut -d '=' -f 2)

# When a VM migrates to a different IP address after a restart, or when you
# re-create a new VM with the same name as a previous one, SSH would normally
# complain that "REMOTE HOST IDENTIFICATION HAS CHANGED". Work around this by
# deleting stale known hosts entries each time you re-configure SSH aliases.
ssh-keygen -f "${KNOWN_HOSTS_FILE}" -R "${TPU_NAME}"

cat <<EOF

Add the following lines to your SSH config (typically ~/.ssh/config)
 

Host ${HOST}
  HostName ${IP_ADDR}
  IdentityFile ${IDENTITY_FILE}
  CheckHostIP no
  HostKeyAlias ${TPU_NAME}
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile ${KNOWN_HOSTS_FILE}
  User ${VM_USER}
  ForwardAgent yes
EOF