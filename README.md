# Advanced NLP Analysis of Airbnb Reviews using Transformers

This project leverages state-of-the-art models from the Hugging Face Transformers library to perform a deep analysis of Airbnb user reviews. The pipeline includes sentiment analysis, text summarization, and keyword extraction to uncover valuable insights from unstructured text data.

<img width="1394" height="345" alt="image" src="https://github.com/user-attachments/assets/684f1056-56d5-4ffb-98d7-40efd1f8ee66" />


## Features

-   **Sentiment Analysis**: Utilizes a pre-trained, fine-tuned BERT-based model through the Transformers `pipeline` to accurately classify review sentiment as 'POSITIVE' or 'NEGATIVE' and provide a confidence score.
-   **Text Summarization**: Employs the `t5-small` model to generate concise, abstractive summaries of longer reviews, making it easy to grasp the main points at a glance.
-   **Keyword Extraction**: Uses the `TfidfVectorizer` from Scikit-learn to identify the most significant terms and topics within each review.
-   **Advanced Text Cleaning**: A robust preprocessing pipeline cleans the raw review text by removing HTML tags, converting to lowercase, and filtering out stopwords using NLTK.
-   **Data Visualization**: Generates a sentiment distribution bar chart and a word cloud of the most frequent terms to provide a high-level overview of the dataset.

## Technologies Used

-   **Python 3.x**
-   **Hugging Face Transformers**: For accessing pre-trained models for sentiment analysis (BERT-based) and summarization (T5).
-   **PyTorch / TensorFlow**: As the backend frameworks for running the transformer models.
-   **Scikit-learn**: For TF-IDF based keyword extraction.
-   **Pandas & NumPy**: For data loading and manipulation.
-   **NLTK**: For text preprocessing, including tokenization and stopword removal.
-   **Matplotlib & Seaborn**: For creating charts and graphs.
-   **WordCloud**: For generating word cloud visualizations.

## Project Workflow

1.  **Data Loading & Preprocessing**: The `reviews.csv` dataset is loaded, and the comments are thoroughly cleaned.
2.  **Sentiment Analysis**: The pre-trained sentiment analysis pipeline is applied to each cleaned review to determine its sentiment label and score.
3.  **Text Summarization**: The T5 summarization pipeline is used to condense reviews longer than 50 words.
4.  **Keyword Extraction**: A TF-IDF matrix is constructed from the corpus of cleaned reviews, and the top 10 keywords for each review are extracted based on their TF-IDF scores.
5.  **Visualization**: The final results are visualized to show the overall sentiment distribution and the most common terms across all reviews.

## Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the repository:
```bash
git clone [https://github.com/Mannan-15/Advanced-Airbnb-Review-Analyzer.git](https://github.com/Mannan-15/Advanced-Airbnb-Review-Analyzer.git)
cd Advanced-Airbnb-Review-Analyzer
```

### 2. Download the Dataset:
This project requires a `reviews.csv` file from an Airbnb dataset. You can find suitable datasets on sites like **Inside Airbnb** or **Kaggle**.
-   Download the `reviews.csv` file.
-   Place it in the main directory of the project.

### 3. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 5. Download NLTK Data:
The first time you run the script, you will need to download necessary NLTK packages. You can do this by running a Python interpreter and typing:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage

To run the analysis, execute the Python script or Jupyter Notebook. The script will process the data, perform the NLP tasks, and display the resulting visualizations.
```bash
python your_script_name.py
# or open and run the .ipynb file in a Jupyter environment.
```
The script will print sample outputs and generate plot windows showing the sentiment analysis results.
