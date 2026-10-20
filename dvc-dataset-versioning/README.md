# DVC Dataset Versioning Project

This project demonstrates how to use DVC for dataset versioning.

## Setup

1. Clone the repository.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Prepare the data:
   ```
   python src/prepare_data.py
   ```

## Project Structure

- `data/`: Contains the dataset.
- `src/`: Contains the source code for data preparation.
- `.dvc/`: DVC configuration files.
- `.gitignore`: Files to ignore in Git.
- `dvc.yaml`: DVC pipeline configuration.
- `requirements.txt`: Python dependencies.