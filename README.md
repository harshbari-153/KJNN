# KJNN: A Novel Neural Network Architecture for Claim Justification Classification

KJNN is a neural network architecture designed for claim justification classification. It utilizes seven trainable matrices to effectively capture the relationships between a claim and its justifications. This project addresses the challenge of classifying the veracity of claims based on provided justifications, specifically within the context of political discourse.

## Project Overview

This project implements KJNN, a neural network model that takes six sentences as input: one claim and five corresponding justifications. The model processes these inputs and generates a 100-dimensional vector representation of the claim-justification relationship. This representation can then be used for downstream classification tasks.

The model's performance was evaluated on the Polifact dataset, which contains articles on political speeches categorized into six veracity labels (ranging from "Complete Truth" to "Complete False"). Six separate classifiers were trained using the KJNN-generated embeddings. The dataset used contains 21000+ claims and corresponding justifications.

## Key Features

*   **Architecture:** KJNN employs an architecture with seven trainable matrices, designed to effectively model the complex interactions between claims and justifications.
*   **100-Dimensional Embeddings:** The model produces compact 100-dimensional vector representations, capturing the semantic relationships between claims and justifications.
*   **Polifact Dataset Evaluation:** The model's performance is rigorously tested on the well-established Polifact dataset.
*   **Comparative Analysis:** KJNN's performance is compared against several strong baselines, including sBERT, DistilBERT, GloVe, RoBERTa Base, and Word2Vec.

## Project Structure

The project is organized into the following folders and files:

*   **parameters (folder):** This folder stores the seven trained matrices used by the KJNN model.
*   **KJNN.py (file):** This file contains the core implementation of the KJNN neural network architecture. It defines the model's layers and functionalities, including the `KJNN_predict` function.
    *   `KJNN_predict(claim, j1, j2, j3, j4, j5)`: This function takes a claim sentence and five justification sentences as input and returns a 100-dimensional vector representation of the claim-justification relationship.
*   **politifact_factcheck_data.json (file):** This JSON file contains the dataset used to train and evaluate the KJNN model. It stores information about claims and their corresponding justifications.
*   **top_justifications.zip (file):** This ZIP file contains additional justification sentences for each claim in the dataset. These justifications may not have been used during training but can be used for further analysis or testing.
*   **train_KJNN.ipynb (file):** This Jupyter notebook provides an interactive environment for training the KJNN model. It likely includes code for data loading, model definition, training process, and evaluation.
*   **train_KJNN.py (file):** This Python script offers a standalone implementation for training the KJNN model. It serves as a more traditional alternative to the Jupyter notebook approach.


## Usage

The `KJNN.py` file provides the core functionality of the model. You can use the `KJNN_predict` function to generate vector representations for new claim-justification pairs. Here's an illustrative example:

```python
# Assuming you have KJNN.py and parameters folder loaded in your environment
import KJNN


# Example claim and justifications
claim = "School X claims it is best school in city."
justification1 = "School X do not have qualified teachers in core subjects."
justification2 = "There has been many dropout from school X since 2020."
justification3 = "School X do not gets qualify in any interschool competetion."
justification4 = "School X gets donation from charity groups."
justification5 = "Last year school X had one board topper"

# Generate vector representation
embedding = KJNN.KJNN_predict(claim, justification1, justification2, justification3, justification4, justification5)
```

## Results

The following table summarizes the classification accuracy (in percentages) achieved by KJNN and the compared models on the Polifact dataset:

| Model         | Classifier 1 | Classifier 2 | Classifier 3 | Classifier 4 | Classifier 5 |
|---------------|--------------|--------------|--------------|--------------|--------------|
| **KJNN**      |  **68.32**   |  **74.32**   |   **77.77**  |   **58.09**  |   **66.23**  |
| sBERT         |    68.73     |    71.24     |     77.35    |     55.95    |     65.77    |
| DistilBERT    |    70.17     |    73.63     |     78.64    |     54.04    |     65.37    |
| GloVe         |    69.11     |    73.63     |     78.50    |     55.08    |     63.74    |
| RoBERTa Base  |    70.82     |    73.41     |     77.18    |     57.04    |     64.71    |
| Word2Vec      |    69.45     |    73.71     |     78.30    |     52.48    |     63.91    |

These results demonstrate the competitive performance of KJNN compared to established methods. Notably, KJNN achieves comparable or even slightly better performance than sBERT in some categories, despite having a simpler architecture. However, it lags behind DistilBERT and RoBERTa Base, suggesting potential areas for improvement in future work, such as exploring different training strategies or incorporating attention mechanisms. The lower accuracy in Accuracy 4 across all models warrants further investigation into the characteristics of the data within that category.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bug fixes, improvements, or new features.

## Contact
Harsh Bari
bari.harsh2001@gmail.com
