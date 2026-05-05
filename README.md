# Mitigating Client Dropout in Smart Grid Energy Forecasting Using Privacy-Preserving Federated Learning

## Overview

This repository contains code, data preprocessing scripts, models, and secure protocols for research into mitigating client dropout in smart grid energy forecasting using privacy-preserving federated learning (FL). The project explores methods that maintain privacy and reliability while handling intermittent client participation in federated training across distributed smart meters or clients.

For an interactive GUI simulation of the federated learning demo, see: https://vercel.com/shubhams-projects-9995d91d/fl-demo

## Key Features

- Privacy-preserving secure aggregation and secret sharing implementations.
- Federated model training with mechanisms to tolerate client dropout.
- Scripts for preprocessing UCI and behavioral datasets used in experiments.
- Evaluation and plotting utilities for result analysis.

## Repository Structure

- `main_federated.py` — Entry point to run federated training experiments.
- `Execution.txt` — Notes and execution commands used in experiments.
- `preprocess_behavioral.py` — Preprocessing pipeline for behavioral dataset.
- `preprocess_uci.py` — Preprocessing pipeline for UCI dataset.
- `plot_results.py` — Utilities to generate plots from results CSVs.
- `requirements.py` — Python dependency list (use to build `requirements.txt`).
- `core/` — Core modules including cryptographic utilities, model, privacy, reliability, secret sharing, and secure protocol implementation.
  - `crypto_utils.py`
  - `model.py`
  - `privacy.py`
  - `reliability.py`
  - `secret_sharing.py`
  - `secure_protocol.py`
- `network/` — Networking / router simulation modules.
- `processed_data/` — Processed CSVs used by clients in experiments.
- `models/` — Saved models (if present in experiments).
- `logs/` — Experiment logs and results stored by subfolders.
- `Dataset/` — Raw dataset files (e.g., `household_power_consumption.txt`).

## Setup and Requirements

1. Create and activate a Python 3.10+ virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies. You can generate a `requirements.txt` from `requirements.py` if needed, or manually install the packages listed.

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not present, consider converting `requirements.py` or contacting the project owner for a canonical dependency list.

## Running Experiments

- Preprocess data for the selected dataset:

```powershell
python preprocess_uci.py
python preprocess_behavioral.py
```

- Run the federated training experiment:

```powershell
python main_federated.py
```

- Use `plot_results.py` to visualize results from CSVs in the `logs/` folder:

```powershell
python plot_results.py --input logs/DP_SMPC/results_c4_dp_20260327_152532.csv
```

Check `Execution.txt` for example commands and experimental parameters used during the original runs.

## Reproducing the GUI Simulation

The GUI simulation for demonstration and visualization is hosted at:

https://vercel.com/shubhams-projects-9995d91d/fl-demo

Visit the link above for an interactive web demo illustrating federated training behavior and dropout handling.

## Evaluation and Metrics

The repository includes evaluation scripts and CSV results under `logs/`. Typical metrics include forecasting error (e.g., MAE, RMSE), privacy metrics where applicable, and reliability measures under client dropout scenarios. Use `plot_results.py` and the `core` evaluation helpers to compute and visualize these metrics.

## Notes on Privacy and Security

This project contains implementations of privacy-preserving building blocks (secret sharing, secure aggregation). These are research implementations and should not be used in production without an independent security review.

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-change`.
3. Make changes and add tests where applicable.
4. Submit a pull request describing the changes and the motivation.

## License

Specify a license for this project (e.g., MIT, Apache-2.0). If you are the repository owner, add a `LICENSE` file. If unsure, contact the project owner to confirm licensing.

## Contact

For questions about the codebase, experiments, or the demo, contact the repository owner or maintainer (project-specific contact details should be added here).

---

This README was generated to document the code and experiments in this workspace. If you want, I can:

- create a `requirements.txt` from `requirements.py`,
- commit and attempt to push this README to the repository remote, or
- add a `LICENSE` file and basic contribution template.

Please tell me which actions you'd like me to take next.
