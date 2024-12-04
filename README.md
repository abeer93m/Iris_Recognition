Here's a template for your `README.md` file based on your instructions:

---

# User Identification for HMDs Using Iris Recognition 

## Overview

This project involves building and training a model for iris-based user identification. Follow the steps below to set up the environment, download necessary data, and train the model.

## Steps to Get Started

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/abeer93m/Iris_Recognition.git
cd IRIS_IDentification
```

### 2. Set Up a Virtual Environment

Create a virtual environment and activate it:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install the Requirements

Install the necessary Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Download the Data

Download the processed and normalized dataset of the 100 users from here: [data(data)https://drive.google.com/drive/folders/1uURMn_XhuDG2R6FHAU1KveuitWgYamg0?usp=sharing)] and place it in the "data" folder. 


### 5. Update the Paths

Update the paths in the configuration files or scripts as needed to ensure that the data and checkpoints are correctly referenced.

### 6. Train the Model

To train the model, use the following command:

```bash
python src/scripts/training.py
```

### 7. Track Your Training with Weights & Biases (WandB)

To track your model training, it's recommended to use [Weights & Biases (WandB)](https://wandb.ai/). 
- Log in to WandB by running:

```bash
wandb login
```

This will allow you to track your model's performance and visualize the training process.

---

## How to Evaluate the Model

### 1. Create Test Pairs

Before evaluating the model, you need to create test pairs. Run the following script:

```bash
python src/scripts/create_test_pairs.py
```

This script will generate the necessary test pairs for evaluation.

### 2. Download the Checkpoint

Download the pre-trained model [checkpoint](https://drive.google.com/file/d/1FE1k5E935zW3zslN0CRQpL2IDRJFKy0Z/view?usp=sharing) from the provided link and place it in the appropriate directory.

### 3. Modify the Path

Open the `src/scripts/evaluation.py` file and update the path to the checkpoint you downloaded.

### 4. Run the Evaluation Script

Finally, run the evaluation script to evaluate the model's performance on the test pairs:

```bash
python src/scripts/evaluation.py
```
This will provide you with the evaluation metrics based on the test pairs you generated.


### 8. Inference
The inferece pipeline includes all the steps to process the images and verify the users. To run the infernece, you need to place images of several users to test whether they belond to the database or not.
I have placed images of User 00008 (geniuine), and User 00370 (Imposter). Just change the directory of the user you want to test and run the script:
```bash
python src/scripts/inference.py
```
