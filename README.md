# Paper_DS

This repository contains code for the paper  
[**"Doubly Smoothed Density Estimation with Application to Miners' Unsafe Act Detection."**](https://doi.org/10.1080/00401706.2025.2582628) by Qianhan Zeng, Miao Han, Ke Xu, Feifei Wang, and Hansheng Wang (Technometrics, 2026). 

The repository is organized into modular components, including simulation scripts, model implementations, and utility functions. A brief overview of the folder structure is provided below:
```bash
.
├── README.md
├── install.sh                 # Shell script for setting up the required environment.
├── models/                    # Includes code for the CD, DS, and GPA methods.
│   ├── CD/                    # Includes code for the Classical nonparametric Density (CD) estimator.
│   ├── DS/                    # Includes code for the Doubly Smoothed (DS) estimator.
│   └── GPA/                   # Includes code for the Grid Point Approximation (GPA) method.
├── simulation/                # Contains all simulation scripts, notebooks, and result files for the simulation study (Section 4.2).
│   ├── Simulation.ipynb # Jupyter notebook used to reproduce the results of the simulation study (Section 4.2).
│   ├── mean-540.npy           # Mean image used for the simulation study.
│   ├── plot_simulation.R      # R script used to reproduce Figure 3.
│   ├── results/               # Folder containing simulation outputs (CSV files and plots).
│   └── simu_auxiliary.py      # Python script with auxiliary functions for simulation.
└── utils.py                   # Utility functions shared across simulation and model code.
```
---

## 🛠 Installation

- ⚠️ Note that there may be package compatibility issues. Please execute the following command in the terminal to create a virtual environment with compatible versions:
```bash
# Step 1: Create a new Conda virtual environment named "env" with Python 3.10
conda create -n env python=3.10 -y

# Step 2: Activate the newly created environment
conda activate env

# Step 3: Run the installation script to install required dependencies
sh install.sh

# Step 4: Register this environment as a Jupyter kernel 
# so it appears as "Python (env)" in Jupyter Notebook or VS Code
python -m ipykernel install --user --name env --display-name "Python (env)"
```

- ⚠️ Note that **TensorFlow 2.12.0** is compiled with **CUDA 11.8** and **cuDNN 8.6.0**. If your local CUDA/cuDNN version is lower (e.g., cuDNN 8.1.x), `AveragePooling2D` may raise a "DNN library is not found" or "UnimplementedError" due to version mismatch. In that case, you can switch to an alternative CPU-based TensorFlow or upgrade your local CUDA/cuDNN version.


## 📊 Part I. Simulation

- The folder [simulation/](./simulation) contains reproducible code and results for the simulation study (Section 4.2) of the main paper. The files and subfolders included in [simulation/](./simulation) are summarized below:

| File | Description |
|:--------------------------------------------------:|:--------------------------------------------------:|
| [Simulation.ipynb](./simulation/Simulation.ipynb) | Jupyter notebook used to reproduce the results of the simulation study (Section 4.2). |
| [mean-540.npy](./mean-540.npy) | Mean image used for the simulation study. |
| [results/](./simulation/results) | Folder containing simulation outputs (CSV files and plots). |
| [plot_simulation.R](./simulation/plot_simulation.R) | R script used to reproduce **Figure 3**. |
| [simu_auxiliary.py](./simulation/simu_auxiliary.py) | Python script with auxiliary functions for simulation. |

- ▶️ **How to Run**:
  - Before running the simulation, make sure you have followed the [Installation](#-installation) above.
  - Step 1: Execute the notebook [Simulation.ipynb](./simulation/Simulation.ipynb). The generated results will be (and already have been) saved in [results/](./simulation/results) folder with filenames of the form `simulation.csv`.
  - Step 2: Run the script [plot_simulation.R](./simulation/plot_simulation.R) to produce the subfigures for **Figure 3**, saved as `logtime_N=xxx.pdf` and `logMSE_N=xxx.pdf`.

 ## 📖 Citation

 If you find this work useful for your research, please cite our paper:

**Zeng, Q., Han, M., Xu, K., Wang, F., & Wang, H. (2026). Doubly Smoothed Density Estimation with Application to Miners' Unsafe Act Detection. *Technometrics*, 68(2), 358–367. https://doi.org/10.1080/00401706.2025.2582628**

```bibtex
@article{zeng2026doubly,
  title={Doubly Smoothed Density Estimation with Application to Miners' Unsafe Act Detection},
  author={Zeng, Qianhan and Han, Miao and Xu, Ke and Wang, Feifei and Wang, Hansheng},
  journal={Technometrics},
  volume={68},
  number={2},
  pages={358--367},
  year={2026},
  publisher={Taylor \& Francis},
  doi={10.1080/00401706.2025.2582628}
}
```
