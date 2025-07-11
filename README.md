# Physics-Informed Machine Learning for Aβ Aggregation Modeling

This repository contains the code and data to reproduce the methodology from the research paper, "Physics-informed machine learning for automatic model reduction in chemical reaction networks." The project applies a Physics-Informed Neural Network (PINN) framework to model the aggregation of Amyloid-beta (Aβ) peptides, a key pathological process in Alzheimer's disease.

The primary goal is to use the **ROMPINN (Reduced Order Model PINN)** framework to automatically select the most appropriate simplified model from a set of eight candidates, thereby gaining quantitative insights into the disease's reaction kinetics.

## Repository Structure

The project is organized as follows:
```
├── data/                # Input data files for model training
│   ├── model1.csv       # Experimental training data for Model 1
│   ├── model1.npz       # Processed data for Model 1
│   └── ...              # Similar files for models 2-8
├── src/                 # Source code
│   ├── analyser.py      # Python script to analyze results and generate figures/tables
│   └── deepxdeAbeta/    # The DeepXDE library used for PINN implementation
├── notebooks/           # Jupyter notebooks for model training
│   ├── model1.ipynb     # Notebook to train Model 1
│   ├── model2.ipynb     # Notebook to train Model 2
│   └── ...              # Similar notebooks for models 3-8
├── results/             # Model outputs and results
│   └── models/          # Subdirectories for each model's output files
│       ├── model1/      # Outputs for Model 1 (loss.dat, variables.dat, etc.)
│       └── ...          # Similar directories for models 2-8
├── paper/               # Research paper and final outputs
│   ├── Pateras.pdf      # The research paper
│   ├── Figure5_Loss_History.png
│   ├── Figure6_Parameter_Evolution.png
│   └── Table5_Final_Parameters.csv
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

-   **`notebooks/model*.ipynb`**: The core training scripts. Each notebook trains one of the eight candidate models.
-   **`data/`**: Contains the experimental input data (`.csv` files) and processed data (`.npz` files) for model training.
-   **`src/deepxdeAbeta/`**: A local copy of the `deepxde` library to ensure code reproducibility.
-   **`src/analyser.py`**: The analysis script that processes the outputs from all 8 models and generates the summary figures and tables presented in the paper.
-   **`results/models/`**: Contains the output directories for each model run with trained model parameters and performance metrics.
-   **`paper/`**: Contains the research paper and the final generated figures and tables.
-   **`requirements.txt`**: Lists all Python dependencies required to run the project.

---

## How to Reproduce the Results from Scratch

Follow these steps to set up the environment, run the models, and generate the final report.

### Step 1: Environment Setup

A dedicated Python environment is recommended (e.g., using `venv` or `conda`).

**1. Install Python:**
Ensure you have Python 3.8+ installed.

**2. Install Dependencies:**
The required Python libraries are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

   a. **(Optional but Recommended) Set up for GPU Training:**
      - Make sure you have an NVIDIA GPU and up-to-date drivers.
      - Check your CUDA version by running `nvidia-smi` in your terminal.
      - Go to the [Official PyTorch Website](https://pytorch.org/get-started/locally/) and select the correct configuration (e.g., Pip, Windows/Linux, CUDA version).
      - Run the generated installation command. It will look something like this:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

### Step 2: Running the Model Training

You must run each of the eight Jupyter Notebooks to generate the necessary output files for the final analysis.

1.  **Start Jupyter:**
    Navigate to the project's root directory in your terminal and launch Jupyter Lab:
    ```bash
    jupyter lab
    ```

2.  **Run Each Notebook:**
    -   In the Jupyter Lab interface, navigate to the `notebooks` directory and open `model1.ipynb`.
    -   Run all the cells in the notebook by clicking `Run > Run All Cells`. This will start the training process for Model 1. You will see the loss values being printed out as it trains.
    -   Upon completion, a new directory `results/models/model1/` will be populated with output files like `loss.dat` and `variables.dat`.
    -   **Repeat this process for all eight notebooks** (`model1.ipynb` through `model8.ipynb`).

    *Note: This is the most time-consuming step. Using a GPU will significantly reduce the training time.*

### Step 3: Generating the Final Report and Figures

After all eight models have been trained and their output files have been generated, you can run the final analysis script.

1.  **Execute the Script:**
    In your terminal, from the project's root directory, run the following command:
    ```bash
    python src/analyser.py
    ```

2.  **Review the Outputs:**
    The script will perform the following actions:
    -   Print a summary of the analysis to your console, including formatted versions of **Table 3** (Goodness of Fit) and **Table 5** (Final Parameters).
    -   Display two plots on your screen: **Figure 5** (Loss History) and **Figure 6** (Parameter Evolution).
    -   Save high-quality images of the figures (`Figure5_Loss_History.png`, `Figure6_Parameter_Evolution.png`) to the `paper/` directory.
    -   Save a CSV file of the final parameters (`Table5_Final_Parameters.csv`) to the `paper/` directory.

By the end of this process, you will have fully reproduced the key findings of the paper, from raw data processing to the final analytical figures and tables.