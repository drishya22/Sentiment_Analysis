Documentation: https://docs.google.com/document/d/1R9Y-z-ogyBiYfjU_UXH69SimK2yLx8bvj1LZ_65bQXQ/edit?usp=sharing


Project link:
https://colab.research.google.com/drive/1Cj7G0yiNca43FGuwjOj-qLiZeZ58rywh?usp=sharing

Sample data:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv


Sentiment Analysis
Sentiment analysis for reviews on websites involves using natural language processing (NLP) techniques to determine the sentiment expressed in user reviews. This process allows businesses to understand customer sentiment at scale, monitor changes over time, and make data-driven decisions to improve customer satisfaction and product quality.
The process typically involves the following steps:
1.	Data Collection: Gather user reviews from websites or platforms relevant to the product or service you're analysing.
2.	Text Preprocessing: Clean and preprocess the text data to remove noise, such as HTML tags, punctuation, and special characters. This may also involve tokenization, stemming, and removing stop words.
3.	Feature Extraction: Convert the pre-processed text into numerical or categorical features that can be used by machine learning algorithms. Common techniques include Bag-of-Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), and word embeddings like Word2Vec or GloVe. In this project, I have explored Bag-of-Words Model or VADER Model and Roberta model.
4.	Sentiment Classification: Using labelled data to classify the sentiment of each review. The labels typically include positive, negative, or neutral sentiment.
5.	Interpretation: Analyse the results to understand the overall sentiment of the reviews, identify common themes or topics mentioned by users, and gain insights into areas for improvement or strengths of the product or service.


Python overview
Python is a general-purpose high-level programming language which is user friendly and easy. It is interpreter-based language that is executed line by line making it easier to test and debug the code. The language and its source code are available to the public for free and it provides us with a large standard library making the task much easier. 

Brief about Bag-of-Words technique
The Bag-of-Words (BoW) model is a simple yet effective technique used in natural language processing (NLP) for feature extraction from text data. It includes Tokenization where the text date is tokenized and vocabulary is constructed by collecting all unique words which represents a feature. 
1.	Tokenization: First, the text data is tokenized, breaking it down into individual words or terms. This process also involves converting the text to lowercase and removing punctuation.
2.	Vocabulary Construction: Next, a vocabulary is created by collecting all unique words (or tokens) from the entire corpus of text data. Each unique word in the vocabulary represents a feature.
3.	Vectorization: For each document in the corpus, a feature vector is created. The length of the feature vector is equal to the size of the vocabulary, and each element in the vector represents the frequency of the corresponding word in the document. If a word from the vocabulary appears in the document, its corresponding frequency count is incremented in the feature vector.
4.	Sparse Representation: Since most documents only contain a small subset of the entire vocabulary, the feature vectors are typically sparse, meaning that most of the elements are zero.
5.	Model Training: The resulting feature vectors can then be used as input to machine learning algorithms for tasks such as classification, clustering, or regression.
While the Bag-of-Words model is straightforward and easy to implement, it has some limitations. It does not capture the semantic meaning or the order of words in the text, which can limit its effectiveness in certain NLP tasks. Additionally, the BoW representation can result in high-dimensional feature vectors, especially for large vocabularies, which may lead to computational challenges and the curse of dimensionality.

Brief about Roberta Model
RoBERTa (Robustly optimized BERT approach) is a state-of-the-art natural language processing (NLP) model introduced by Facebook AI in 2019. It is built upon the Transformer architecture, which has shown remarkable performance in various NLP tasks. RoBERTa is essentially an optimized version of Google's BERT (Bidirectional Encoder Representations from Transformers) model, incorporating several improvements and modifications. Here's an overview of key aspects of the RoBERTa model:
1.	Pretraining: Like BERT, RoBERTa is pretrained on a large corpus of text data using unsupervised learning objectives, such as masked language modeling (predicting masked words within a sentence) and next sentence prediction. This pretraining phase allows the model to learn contextualized representations of words and sentences.
2.	Architecture: RoBERTa utilizes the Transformer architecture, which consists of multiple layers of self-attention mechanisms and feedforward neural networks. However, RoBERTa introduces several modifications to the architecture and training procedure compared to BERT, which contribute to its improved performance.
Overall, RoBERTa has achieved state-of-the-art results on a wide range of NLP benchmarks and tasks, demonstrating its effectiveness in capturing complex linguistic patterns and semantics in text data.


Objective of the project
The objective of this project is to apply programming knowledge into real-world situations/problems and provide exposure to how programming skills help in developing good software.
Real world problem-To allow businesses to understand customer sentiment at scale, monitor changes over time, and make data-driven decisions to improve customer satisfaction and product quality.



Proposed system
The aim of the project is to make a good NLP model to understand customer review in the python language. This project shows that Python is simple, easy to learn and use as well as fast to develop.
The python interpreter and extensive standard library allows us to make, train and test AI models. There are libraries which provide pre-trained models, tools for text preprocessing, feature extraction, and implementation of various NLP tasks.
Python's versatility allows developers to build custom NLP pipelines and solutions tailored to specific requirements. Additionally, Python supports parallel and distributed computing, facilitating scalability for processing large volumes of text data efficiently.




Libraries used and their purposes

Pandas library

Pandas is a Python library for data manipulation and analysis. It provides data structures like Series and DataFrame, supports input/output from various file formats, offers powerful data manipulation tools, integrates with data visualization libraries, and is widely used for tasks like cleaning, exploring, and analyzing data.
In the project pandas is used to store a huge dictionary which contains the result of sentiment analysis

Matplotlib library

Matplotlib is a comprehensive library in Python for creating static, interactive, and animated visualizations. It offers a wide range of plotting functions and customization options, allowing users to create publication-quality plots for data analysis and presentation. Matplotlib is widely used in various fields, including scientific research, data analysis, and visualization tasks.
In the project this library is used for plotting different results obtained using VADER model.

Numpy module

 NumPy is a fundamental library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. NumPy is widely used in scientific computing, data analysis, machine learning, and more, due to its speed, flexibility, and extensive capabilities.
It is used to do different mathematical calculations involving arrays in the project.

Seaborn

 Seaborn is a Python visualization library based on matplotlib, designed for creating attractive and informative statistical graphics. It provides a high-level interface for drawing attractive and informative statistical graphics, built on top of matplotlib. Seaborn simplifies the process of creating complex visualizations such as heatmaps, pair plots, and categorical plots, making it an excellent choice for exploratory data analysis and presentation-ready graphics.
In the project it is used plots showing the results of sentiment analysis done by ROBERTA Model.

NLTK

NLTK (Natural Language Toolkit) is a comprehensive library in Python for working with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, along with a suite of text processing libraries for tasks such as tokenization, stemming, tagging, parsing, and classification. NLTK is widely used in research and education for natural language processing tasks, including sentiment analysis, named entity recognition, part-of-speech tagging, and more.
In this project, NLTK has made sentiment analysis possible. It is used to tokenization, part-of-speech tagging, chunking etc.




Limitations and Future Scope

Loss of Sequence Information: The BoW model disregards the order and context of words in a document, treating each document as an unordered collection of words. This leads to the loss of important sequential information, such as word dependencies and sentence structure.

	 Lack of Semantic Understanding: The BoW model treats each word as independent and assigns equal importance to all words in the vocabulary. It does not capture the semantic relationships between words or the contextual meaning of phrases, limiting its ability to understand language nuances. Domain Specificity: Pre-trained models like RoBERTa are trained on large, generic text corpora and may not capture domain-specific knowledge or nuances. Fine-tuning on domain-specific data can mitigate this issue, but it requires additional labelled data and computational resources.

	Difficulty in Interpretability: Transformer-based models like RoBERTa are complex neural networks with multiple layers of self-attention mechanisms. Understanding the inner workings of these models and interpreting their predictions can be challenging compared to simpler models like logistic regression or decision trees.

	 Computational Complexity: RoBERTa, like other transformer-based models, is computationally intensive and requires substantial computational resources for training and inference. Fine-tuning large pre-trained models like RoBERTa on specific tasks can also be resource-intensive.

	The future scope of both the Bag of Words model and the RoBERTa model lies in their continued refinement, adaptation to emerging challenges, and integration with complementary techniques to advance the state-of-the-art in natural language processing.


