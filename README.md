# Overview

Python implementation of [DeepMind's LARGE LANGUAGE MODELS AS OPTIMIZERS paper.](https://arxiv.org/pdf/2309.03409.pdf)

The project consists of two main scripts: `Prompts.py` and `main.py`. 

`Prompts.py` provides example prompts and a meta prompt for generating new texts with high scores, as well as a scorer prompt for answering a question with a number. `main.py` is the heart of the project, implementing an optimization process called OPRO (Optimization with Prompt Ranking). 

This script reads input data from CSV files, creates language model chains from templates, and performs iterations of the OPRO process. The result is a DataFrame that contains the performance scores of different prompts, which is then printed and saved to a CSV file.

This project is a great tool for anyone interested in language model optimization, text generation, and data analysis. It provides a practical implementation of the OPRO process and demonstrates how to use language models to generate high-scoring texts.

# Technologies and Frameworks

This project is built with Python and uses several libraries and frameworks:

- **Pandas**: Used for data manipulation and analysis. It is particularly used for reading input data from CSV files and creating the output DataFrame.
- **Numpy**: Used for numerical computations.
- **Scikit-learn**: Used for machine learning and data mining.
- **PyTorch**: Used for building the language models.
- **Transformers**: Used for creating the language model chains from templates.
# Installation

Follow the steps below to install and run the project:

## Step 1: Clone the Repository

First, clone the repository to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/Moocember/Optimization-by-PROmpting.git
```

## Step 2: Install the Dependencies

This project requires several Python libraries. You can install them using pip, the Python package installer. Run the following commands in your terminal:

```bash
pip install pandas
pip install openai
pip install langchain
```

## Step 3: Ensure the Data Files are in Place

The project requires two data files:

- performance.csv
- training_exemplars.csv

These files should be located in a folder named "data" in the same directory as your Python scripts.

## Step 4: Run the Main Script

Finally, you can run the main script of the project. This script will create an optimizer chain and a scorer chain, read the performance and training exemplar data from CSV files, and then call the "opro" function with the necessary parameters. The output will be printed and saved to a new CSV file named "performance2.csv".

You can run the main script with the following command in your terminal:

```bash
python main.py
```
