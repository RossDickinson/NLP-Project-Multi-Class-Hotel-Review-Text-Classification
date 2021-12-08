# Multi-Class-Hotel-Review-Text-Classification

# 1. Introduction
This project pertains to evaluate the data collection, wrangling, preprocessing, configuration and performance of a multi-class text classification project. I created a bidirectional LSTM RNN model, tuned the respective hyperparameters, and evaluated the performance metrics of the chosen model on the test data. 

With the reliance on customer reviews to generate new sales, particulary through the medium of the internet, has increased exponentially in recent years. New customers value the opinions of their fellow consumers and are always in pursuit of the best available services as voted by the public. This project pertained to create a multi-text classification deep learning mdoel that was capable of predicting TripAdvisor hotel review ratings based on the language belonging to each review. 

# 2. Data Source and Exploration

The class ratings distribution is presented below. Clearly there was a significant class imabalance in favor of Ratings 4 and 5.

<img src="https://user-images.githubusercontent.com/64614298/145255769-636dd76e-4f30-44e9-887f-fe88bda2d2ec.png" alt="drawing" width="600" class="center"/>

Therefore, in order to avoid a bias instilled in the model, it was imperative to integrate the concept of class weights to the model.

# 3. Data Preprocessing

**Cleaning the Text**

Text preprocessing was required to format the raw text in such a way that the RNN model could consume it as input. All punctuation was removed from the text because it added no linguistic value or context for the model to learn from. Furthermore, a collection of common words, known as stopwords in the nltk (Natural Language Toolkit), were removed. Such words included, “the”, “a”, “an” and “in”, which were removed from the text because they added no sentimental value to the text. 

Removing these words helped to reduce the computational load by reducing the size of the word corpus to be analyzed by the model. The text was also processed using stemming, which reduced each word down to its root with the purpose of improving computational performance. Following this, the text was tokenized to format each review in a list format. The tokenized reviews were then converted to sequences of integers, in which each unique word correseponded to a respective integer value.

**Improving Class Imbalance using Weight Classes**

Converting from five classes to five classes was justified due to the prominent similarities between the vocabulary in classes one and two, and four and five, respectively. However, reducing the class load induced an imbalanced dataset. In order to compensate for this class imbalance, the concept of balanced class weights was integrated into the ML models. 

The purpose of integrating class weights was to reduce the bias bound by the ML models. Models tend to learn the features of the majority class well because there is more data the model can learn from, hence impeding a bias in favor of the majority class and against the minority classes. Classes were coupled with an associated class weight to instruct the deep learning models during the training phase to penalize the misclassification made by the minority class by a higher weight than that of the majority class. The weights were determined using the inverse of the respective frequencies, as shown in Equation 1. 

<img src="https://user-images.githubusercontent.com/64614298/145258783-445f4da9-51e7-4997-89cc-3e65fad8d169.png" alt="drawing" width="600" class="center"/>

**Preparing Text Review Data for RNN Model**

In order the RNN model to consume the sequential text data, it was first necessary to enforce a maximum sequence length. This was achieved through the use of padding. This meant that each review was either extended or shrunk to a universally consistent length of 200 words. 

**Summary of data preprocessing:** 

1.	Clean text to remove stop words/punctuation, stemming, etc. techniques to simplify the model data input.
2.	Split data into training, validation, and testing data sets.
3.	Tokenize to clean text to create a list of words for each review.
4.	Convert tokenized words to a sequence of integers with an index for each word in the corpus.
5.	Pad the sequences to ensure all input sequences are of the same length.


# 4. Model Selection and Architecture

# 5. Hypertuning 

# 6. Evaluating Model Performance

# 7. Conclusion
