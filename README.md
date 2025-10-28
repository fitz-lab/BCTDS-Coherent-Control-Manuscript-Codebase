# BCTDS coherent control Manuscript – Figure Reproduction

This repository contains the source code and data used to generate the figures for the manuscript: **Spectroscopy and Coherent Control of Two-Level System Defect Ensembles Using a Broadband 3D Waveguide**. 
Each figure is fully reproducible using the provided Python scripts.

---

## Environment Setup

We recommend using [conda](https://docs.conda.io/) for environment management.

```bash
# create a fresh environment
conda create -n bctds-paper python=3.11
conda activate bctds-paper

# install dependencies
pip install -r requirements.txt
```

*Alternatively, you may use `python -m venv venv` to create a virtual environment and then install the requirements.*

---

## Reproducing Figures

Each figure has its own folder under `figures/` (e.g. `Figure_1_overview_cartoons/`, `Figure_2_phase_V_sweep_width/`, …).  
Inside each folder, you will find the Python script to generate the figure.

Example for **Figure 1**:

```bash
cd figures/Figure_1_overview_cartoons
python generate_fig_1.py
```

The script will automatically create an output folder named `analysis_plots` inside the figure directory and save the result there, printing a message:

```
✓ Figure saved to: path/to/figures/Figure_1_overview_cartoons/analysis_plots/Fig_overview.png
```

Repeat for other figure folders (`Figure_2_/`, `Figure_3_/`, …).

---

## Repository Structure

```
BCTDS-Coherent-Control-Manuscript-Codebase/
│
├── figures/           # Source code & data for individual figures
│   ├── Figure_1_overview_cartoons/
│   │   └── generate_fig_1.py
│   ├── Figure_2_phase_V_sweep_width/
│   │   └── generate_fig_2.py
│   └── ...
├── requirements.txt   # Dependencies for reproducing the figures
└── README.md          # This file
```

---

## License

This repository is distributed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or requests related to this repository, please contact:  
- **Qianxu Wang** – [qianxu.wang.gr@dartmouth.edu]  
- **Mattias Fitzpatrick** – [mattias.w.fitzpatrick@dartmouth.edu]
