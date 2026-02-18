# Raspberry Pi Ansible Setup Scripts

This directory contains Ansible scripts to automate the setup and maintenance of the Raspberry Pi cluster (`alice`, `bob`, `charlie`, `diana`) for distributed MPC path planning.

## Prerequisites

1.  **Install Ansible** (on your local machine/controller):
    ```bash
    pip install ansible
    # or
    # sudo apt install ansible
    ```
    *Note: Ensure you have a recent version of Ansible (>= 2.12 recommended) to support Python 3 features.*

2.  **SSH Access**:
    Ensure you can SSH into all Pis. The inventory assumes you have SSH keys set up or can provide passwords.
    It's recommended to have entries in your `~/.ssh/config` for the hosts if you use custom hostnames, but the inventory file uses `ansible_host` (implied or explicit) matching your SSH config.

## Inventory

The inventory file `inventory.yml` defines the hosts and variables.
-   **Hosts**: `alice`, `bob`, `charlie`, `diana`
-   **User**: Defined per host (e.g., `ansible_user: alice`)
-   **Python Interpreter**: Forces `/usr/bin/python3` to avoid compatibility issues.
-   **SSH Args**: Disables strict host key checking to facilitate connecting to re-imaged or new hosts.

## Running the Playbook

To set up the cluster, run the `setup_pis.yml` playbook.

**Command:**
```bash
ansible-playbook -i scripts/inventory.yml scripts/setup_pis.yml -K
```

**Flags Explanation:**
-   `-i scripts/inventory.yml`: Specifies the inventory file.
-   `scripts/setup_pis.yml`: The playbook to run.
-   `-K` (capital K): **Crucial!** Prompts for the `BECOME` password (sudo password) on the remote hosts. This is required for installing apt packages.
-   `-k` (lowercase k): (Optional) Prompts for the SSH connection password if you haven't set up SSH keys.
-   `-v`: (Optional) Verbose mode for debugging.

## Common Issues & Troubleshooting

1.  **Missing Sudo Password (`Missing sudo password`)**:
    Ensure you use the `-K` flag.

2.  **ModuleNotFoundError (ansible.legacy.setup)**:
    This usually means Ansible is using the wrong Python interpreter on the remote host. The `inventory.yml` fixes this by setting `ansible_python_interpreter: /usr/bin/python3`.

3.  **SSH Connection Errors**:
    -   Check your VPN/Tailscale connection.
    -   Verify you can SSH manually: `ssh alice@alice`.
    -   Host key verification failures are suppressed by `StrictHostKeyChecking=no` in the inventory, but severe mismatches might still block connection.

4.  **Local Modifications Error (Git)**:
    If the playbook fails with "Local modifications exist in the destination", it means files on the Pi have been changed manually. You may need to commit/stash them on the Pi or delete the directory to let Ansible re-clone.

## Playbook Structure created

-   **Install Dependencies**: Installs `git`, `python3`, `cmake`, and other build tools via `apt`.
-   **Clone Repo**: Clones `aima-python-matsat` to `~/aima-python-matsat` on each Pi.
-   **Update Submodules**: Runs `git submodule update --init --recursive --remote` to pull the latest MP-SPDZ.
-   **Setup MP-SPDZ**: Executed the setup script to compile MP-SPDZ dependencies.
