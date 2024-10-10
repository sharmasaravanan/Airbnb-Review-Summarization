# Airbnb-Review-Summarization

### Project Overview
This project aims to enhance the user experience on Airbnb by summarizing customer reviews and extracting sentiments from those reviews. This approach helps potential guests make more informed decisions about booking accommodations. The solution uses Natural Language Processing (NLP) techniques for text preprocessing, sentiment analysis, review summarization, and key topic extraction from reviews.

### Features
* Data Preprocessing: Clean the review data by removing unwanted characters, stopwords, and performing text normalization.
* Sentiment Analysis: Classify the sentiment of reviews (positive, negative, neutral) using BERT-based sentiment analysis.
* Summarization: Generate concise summaries of customer reviews using a transformer-based model.
* Key Topic Extraction: Extract the top 10 keywords from each review using TF-IDF.
* Visualization: Show insights visually (if applicable in future extensions).

### Tech Stack
* Python: Programming language used for implementing the solution.
* NLP Libraries:
  * transformers for sentiment analysis and summarization.
  * sklearn for TF-IDF vectorization.
  * pandas for data manipulation.
  * spacy for named entity recognition (optional).
* Models:
  * Sentiment Analysis: BERT-based model using transformers.pipeline.
  * Summarization: T5 model (t5-small) for fast and effective review summarization.
  * TF-IDF: To extract important keywords from reviews.

### Files
airbnb_review_summarization.py: The main script that processes reviews, performs sentiment analysis, generates summaries, and extracts key topics.

### Key Functions
1. Data Preprocessing:
  * Removes special characters, stopwords, and normalizes the text.

2. Sentiment Analysis:
  * Uses a BERT model from transformers.pipeline to predict the sentiment of each review:
    ```python
    bert_sentiment_analyzer = pipeline('sentiment-analysis')
    ```

3. Summarization:
  * Uses the t5-small model from transformers for faster summarization:
    ```python
    summarizer = pipeline('summarization', model='t5-small', device=0 if torch.cuda.is_available() else -1)
    ```

4. Key Topic Extraction:
  * Uses TF-IDF to extract the top 10 keywords from each review:
  ```python
  tfidf = TfidfVectorizer(max_features=1000)
  tfidf_matrix = tfidf.fit_transform(df['clean_comments'])
  ```

### Future Improvements
* Topic Modeling: Use LDA or other advanced techniques to improve topic extraction.
* Abstractive Summarization: Enhance the summarization model for more human-like summaries.
* Visualization: Create data visualizations for better insights into sentiment distribution, common themes, etc.
