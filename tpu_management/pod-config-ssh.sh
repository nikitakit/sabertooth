#!/bin/bash
set -e

function die
{
    local message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

[ -z "$1" ] && die "Usage: pod.sh config-ssh NAME"

POD_NAME=${POD_NAME_PREFIX}$1
HOST_PREFIX=$1
echo "POD_NAME=${POD_NAME?}"
echo "HOST_PREFIX=${HOST_PREFIX?}"
echo "POD_ZONE=${POD_ZONE?}"

echo "Extracting SSH arguments from the gcloud tool..."
SSH_COMMAND0=$(gcloud alpha compute tpus tpu-vm ssh ${POD_NAME?} --zone ${POD_ZONE?} --worker 0 --dry-run)
echo ${SSH_COMMAND0?}
SSH_COMMAND1=$(gcloud alpha compute tpus tpu-vm ssh ${POD_NAME?} --zone ${POD_ZONE?} --worker 1 --dry-run)
echo ${SSH_COMMAND1?}
SSH_COMMAND2=$(gcloud alpha compute tpus tpu-vm ssh ${POD_NAME?} --zone ${POD_ZONE?} --worker 2 --dry-run)
echo ${SSH_COMMAND2?}
SSH_COMMAND3=$(gcloud alpha compute tpus tpu-vm ssh ${POD_NAME?} --zone ${POD_ZONE?} --worker 3 --dry-run)
echo ${SSH_COMMAND3?}

IP_ADDR0=$(echo ${SSH_COMMAND0} | cut -d "@" -f 2)
IP_ADDR1=$(echo ${SSH_COMMAND1} | cut -d "@" -f 2)
IP_ADDR2=$(echo ${SSH_COMMAND2} | cut -d "@" -f 2)
IP_ADDR3=$(echo ${SSH_COMMAND3} | cut -d "@" -f 2)

# Configuration fields other than the IP address are assumed to be identical for all workers
VM_USER=$(echo ${SSH_COMMAND0} | grep -o '[^\s]*@' | xargs | cut -d '@' -f 1)
IDENTITY_FILE=$(echo ${SSH_COMMAND0} | grep -o -- '-i.* -o' | cut -d ' ' -f 2)
KNOWN_HOSTS_FILE=$(echo ${SSH_COMMAND0} | grep -o -- 'UserKnownHostsFile=.*' | cut -d ' ' -f 1 | cut -d '=' -f 2)

# When a VM migrates to a different IP address after a restart, or when you
# re-create a new VM with the same name as a previous one, SSH would normally
# complain that "REMOTE HOST IDENTIFICATION HAS CHANGED". Work around this by
# deleting stale known hosts entries each time you re-configure SSH aliases.
ssh-keygen -f "${KNOWN_HOSTS_FILE}" -R "${POD_NAME}_0"
ssh-keygen -f "${KNOWN_HOSTS_FILE}" -R "${POD_NAME}_1"
ssh-keygen -f "${KNOWN_HOSTS_FILE}" -R "${POD_NAME}_2"
ssh-keygen -f "${KNOWN_HOSTS_FILE}" -R "${POD_NAME}_3"

cat <<EOF

Add the following lines to your SSH config (typically ~/.ssh/config)


Host ${HOST_PREFIX}0
  HostName ${IP_ADDR0}
  IdentityFile ${IDENTITY_FILE}
  CheckHostIP no
  HostKeyAlias ${POD_NAME}_0
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile ${KNOWN_HOSTS_FILE}
  User ${VM_USER}
  ForwardAgent yes

Host ${HOST_PREFIX}1
  HostName ${IP_ADDR1}
  IdentityFile ${IDENTITY_FILE}
  CheckHostIP no
  HostKeyAlias ${POD_NAME}_1
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile ${KNOWN_HOSTS_FILE}
  User ${VM_USER}
  ForwardAgent yes

Host ${HOST_PREFIX}2
  HostName ${IP_ADDR2}
  IdentityFile ${IDENTITY_FILE}
  CheckHostIP no
  HostKeyAlias ${POD_NAME}_2
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile ${KNOWN_HOSTS_FILE}
  User ${VM_USER}
  ForwardAgent yes

Host ${HOST_PREFIX}3
  HostName ${IP_ADDR3}
  IdentityFile ${IDENTITY_FILE}
  CheckHostIP no
  HostKeyAlias ${POD_NAME}_3
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile ${KNOWN_HOSTS_FILE}
  User ${VM_USER}
  ForwardAgent yes
EOF