
# How to setup environment for LGU and baselines experiment

## 1. Install Miniconda (if not yet)
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

#  Reload shell
source ~/.bashrc
```

## 2. Create conda envirionment
```bash
# Del old env (if exsist)
conda env remove -n old_env_name -y
conda env remove -n pstree -y
conda env remove -n rils-rols -y

```bash
cd env_setup

# For LGO / PySR / Operon. 
# * PySR requires a different version of numpy than Operon. 
# A technical solution has been implemented for this issue.
conda env create -f env_setup/env_py310.yml
conda activate py310
cd ..

# Similarly, build env for PSTree and RILS-ROLS.

# For PSTree. * PSTree seems not wrok well on Python 3.10 or 3.11 
conda env create -f env_setup/env_pstree.yml
conda activate pstree

# For RILS-ROLS
conda env create -f env_setup/env_rils-rols.yml
conda activate rils-rols
```

# Also, can follow the official guilde on GitHub: 

- (PSTree) https://github.com/hengzhe-zhang/PS-Tree
- (RILS-ROLS) https://github.com/kartelj/rils-rols