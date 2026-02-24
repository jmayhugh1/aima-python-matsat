#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
import urllib.request
import tarfile

# Configuration
REPO_URL = "https://github.com/jmayhugh1/aima-python-matsat.git"
REPO_BRANCH = "log_encoding"

# Determine where the script was executed from. 
# Since this script lives in `scripts/local_setup.py`, the repo root is the parent directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DEST = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MPSPDZ_DIR = os.path.join(REPO_DEST, "MP-SPDZ")
BOOST_URL = "https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.bz2"
BOOST_DEST = os.path.join(MPSPDZ_DIR, "deps/libOTe/cryptoTools/thirdparty/boost_1_83_0.tar.bz2")

def run(cmd, cwd=None, exit_on_fail=True):
    print(f"\n---> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if result.returncode != 0 and exit_on_fail:
        print(f"Error executing command: {cmd}")
        sys.exit(result.returncode)
    return result.returncode

def main():
    print("=== Starting Local Raspberry Pi Setup ===")

    # 1. System Dependencies
    print("\n--- 1. Installing System Dependencies ---")
    run("sudo apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=5 update || true")
    run("sudo apt-get install -y git python3 python3-pip automake build-essential clang cmake libboost-dev "
        "libboost-thread-dev libclang-dev libgmp-dev libntl-dev libsodium-dev libssl-dev libtool vim "
        "gdb valgrind wget python3-packaging")

    # 2. Update Repository and Submodules
    print("\n--- 2. Setting up Repository ---")
    run("git submodule update --init --recursive --remote --force --jobs 4 --depth 1", cwd=REPO_DEST)

    # 3. MP-SPDZ Configuration
    print("\n--- 3. Configuring MP-SPDZ ---")
    config_mine = os.path.join(MPSPDZ_DIR, "CONFIG.mine")
    config_file = os.path.join(MPSPDZ_DIR, "CONFIG")
    
    if os.path.exists(config_mine):
        os.remove(config_mine)
        
    with open(config_mine, "w") as f:
        f.write("ARCH = -march=native\n")
        f.write("CXX = clang++\n")
        f.write("USE_NTL = 0\n")
        f.write("MY_CFLAGS += -I/usr/local/include\n")
        f.write("MY_LDLIBS += -Wl,-rpath -Wl,/usr/local/lib -L/usr/local/lib\n")

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            content = f.read()
        content = content.replace(" -Werror", "")
        with open(config_file, "w") as f:
            f.write(content)

    # 4. Python Dependencies
    print("\n--- 4. Installing Python Dependencies ---")
    if run("pip3 show ipython", exit_on_fail=False) != 0:
        run("pip3 install pip ipython --break-system-packages")

    # 5. MP-SPDZ Dependencies (libOTe)
    print("\n--- 5. Compiling MP-SPDZ Dependencies (libOTe & Boost) ---")
    run("git submodule update --init --recursive --remote --force --jobs 4 --depth 1 deps/libOTe", cwd=MPSPDZ_DIR)

    libote_static = os.path.join(MPSPDZ_DIR, "local/lib/liblibOTe.a")
    boost_dir = os.path.dirname(BOOST_DEST)
    
    if not os.path.exists(libote_static):
        os.makedirs(boost_dir, exist_ok=True)
        if not os.path.exists(BOOST_DEST):
            print(f"Downloading Boost from {BOOST_URL}...")
            urllib.request.urlretrieve(BOOST_URL, BOOST_DEST)
        else:
            print("Boost archive already exists. Skipping download.")
        
        run("make clean-deps boost libote ARM=1", cwd=MPSPDZ_DIR)
    else:
        print("libOTe is already compiled. Skipping Boost download and libOTe build.")

    # 6. MP-SPDZ Protocols
    print("\n--- 6. Compiling Protocols & Generating Keys ---")
    p0_key = os.path.join(MPSPDZ_DIR, "Player-Data", "P0.key")
    if not os.path.exists(p0_key):
        run("./Scripts/setup-ssl.sh", cwd=MPSPDZ_DIR)
        
    mascot_x = os.path.join(MPSPDZ_DIR, "mascot-party.x")
    shamir_x = os.path.join(MPSPDZ_DIR, "shamir-party.x")
    
    if not os.path.exists(mascot_x):
        run("make mascot-party.x ARM=1 -j2", cwd=MPSPDZ_DIR)
    else:
        print("mascot-party.x exists. Skipping compile.")
        
    if not os.path.exists(shamir_x):
        run("make shamir-party.x ARM=1 -j2", cwd=MPSPDZ_DIR)
    else:
        print("shamir-party.x exists. Skipping compile.")

    print("\n=== Setup Complete! ===")

if __name__ == "__main__":
    main()
