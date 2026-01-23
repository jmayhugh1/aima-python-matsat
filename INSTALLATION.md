# Installation & Verification

This repository uses **git submodules**. Follow the steps below to clone the repository and verify that all required submodules and files are present.

---

## 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules https://github.com/jmayhugh1/aima-python-matsat
cd aima-python-matsat
pip install -r requirements.txt
```


---

## 2. Verify submodules were cloned correctly

### Check `.gitmodules`

Run:

```bash
cat .gitmodules
```

It should look like this:

```ini
[submodule "aima-data"]
  path = aima-data
  url = https://github.com/aimacode/aima-data.git

[submodule "MP-SPDZ"]
  path = MP-SPDZ
  url = https://github.com/jmayhugh1/MP-SPDZ.git
  branch = adding-matsat-subproc
```

---

## 3. Verify MP-SPDZ remotes and compile CP


Navigate into the MP-SPDZ submodule:

```bash
cd MP-SPDZ
git remote -v
```

It should look like this:

```text
origin  https://github.com/jmayhugh1/MP-SPDZ.git (fetch)
origin  https://github.com/jmayhugh1/MP-SPDZ.git (push)
upstream        https://github.com/data61/MP-SPDZ.git (fetch)
upstream        https://github.com/data61/MP-SPDZ.git (push)
```

Now, compile the communcation protocls
```bash
make -j8 shamir-party.x
make -j8 mascot-party.x
```
---

## 4. Verify the correct branch

Run:

```bash
git branch --show-current
```

You should see:

```text
adding-matsat-subproc
```

---

## 5. Verify MatSat-related source files

Inside the `MP-SPDZ` directory, confirm the following files exist:

### Core MatSat logic

```text
Programs/Source/matsat_utils.py
```

This file contains the MatSat logic used by the programs below.

---

### MatSat programs

```text
Programs/Source/multiparty_matsat.py
```

Original MatSat code used by the Wumpus examples.

```text
Programs/Source/private_path_query.py
```

Logic for the privacy-preserving path query.

---

## 6. Examples and supporting code

The following files demonstrate usage and provide supporting utilities:

```text
private_path_query_utils.py
```

Contains logic and classes for paths and grids, as well as join computation logic.

```text
bayesien_paths.py
bayesien_paths.ipynb
```

Implements the Bayesian belief map and Alice/Bob classes, along with a notebook showcasing an example.

---

## verify logic works

run the following tests, 

```
pytest tests/test_bayesian_paths.py
pytest tests/test_private_path_query_utils.py 
```



## Troubleshooting

If submodules were not cloned correctly, from the repository root run:

```bash
git submodule update --init --recursive
```

---
