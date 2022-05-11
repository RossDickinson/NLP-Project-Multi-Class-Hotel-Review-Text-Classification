# Multi-Class-Hotel-Review-Text-Classification

## Note: All code pertaining to this project can be found here - https://colab.research.google.com/drive/1T_Wf9TuFoCEqt7ef9bvXEg0pKDJ4DYx_?usp=sharing.

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

Removing these words helped to reduce the computational load by reducing the size of the word corpus to be analyzed by the model. The text was also processed using lemmatizing, which removed inflectional endings of words with the purpose of improving computational performance. Following this, the text was tokenized to format each review in a list format. The tokenized reviews were then converted to sequences of integers, in which each unique word correseponded to a respective integer value. A sample of the first tokenized instance is shown below:

<img src="https://user-images.githubusercontent.com/64614298/145261096-15cd6506-d9b2-4258-9868-bc602de2884b.png" alt="drawing" width="600" class="center"/>

**Improving Class Imbalance using Weight Classes**

Converting from five classes to three classes was justified due to the prominent similarities between the vocabulary in classes one and two, and four and five, respectively. However, reducing the class load induced an imbalanced dataset. In order to compensate for this class imbalance, the concept of balanced class weights was integrated into the ML models. 

The purpose of integrating class weights was to reduce the bias bound by the ML models. Models tend to learn the features of the majority class well because there is more data that the model can learn from, hence impeding a bias in favor of the majority class and against the minority classes. Classes were coupled with an associated class weight to instruct the deep learning models during the training phase to penalize the misclassification made by the minority class by a higher weight than that of the majority classes. 

<img src="https://user-images.githubusercontent.com/64614298/145258783-445f4da9-51e7-4997-89cc-3e65fad8d169.png" alt="drawing" width="600" class="center"/>

**Preparing Text Review Data for RNN Model**

In order the RNN model to consume the sequential text data, it was first necessary to enforce a maximum sequence length. This was achieved through the use of padding. This meant that each review was either extended or shrunk to a universally consistent length of 200 words. 

Sequence length defines the maximum length of each review. Each review is either padded or reduced to fit the defined maximum sequence length. For example, a review with 1800 tokens would be reduced to 200 tokens if the maximum length was specified to be 200. For an RNN model, the inputs all needed to be of the same length and of numerical tokenized format. The following text parameters were identified about the training data set before padding was added to the tokenized reviews.

<img src="https://user-images.githubusercontent.com/64614298/145261599-a41b03f8-9876-4078-a325-6e40f91b9c5e.png" alt="drawing" width="600" class="center"/>

Sequenced text was then converted to a representative integer and padded using prior padding. For example, the integer 286 shown below refers to the first word in the sequence shown below, ‘convenient’.

<img src="https://user-images.githubusercontent.com/64614298/145261096-15cd6506-d9b2-4258-9868-bc602de2884b.png" alt="drawing" width="600" class="center"/>
<img src="https://user-images.githubusercontent.com/64614298/145262759-f3464b3f-2c5e-43e3-83c0-7130fd15d10b.png" alt="drawing" width="600" class="center"/>

The padded review sequences (X input) was now of the shape (# of instances, maximum sequence length) and ready to be fed to the RNN model, namely, (13933, 200). However, the ratings sequences (Y output) needed to be reformed to the format (# of instances, # of classes). To do so, the *pd.get_dummies().values* function was used on y_train, givng an output for the training set of (13933, 3). All the data was then ready to be fed to the model. 

**Summary of data preprocessing:** 

1.	Clean text to remove stop words/punctuation, stemming, etc. techniques to simplify the model data input.
2.	Split data into training, validation, and testing data sets.
3.	Tokenize to clean text to create a list of words for each review.
4.	Convert tokenized words to a sequence of integers with an index for each word in the corpus.
5.	Pad the sequences to ensure all input sequences are of the same length.
6.	Convert padded sequences to equivalent integer values. 
7.	Pad integer sequences to meet the maximum sequence length constraint. 
8.	Format y_inputs in shape (# of instances, # of classes)
9.	Format x_inputs in shape (# of instances, maximum sequence length)

# 4. Model Selection and Architecture

Recurrent Neural Networks (RNNs) use a combination of simple functions to produce a complex function. RNNs work particularly well with sequential or time sequence data input because RNN models implement a memory of such; in that the output from a previous timestep is used as an input feature to the next sequence function. Consequently, the model can learn the context and sequential nature of the data input. In the context of NLP, an RNN model can learn the words in a sentence that occurred prior to the current word. Therefore, the model gains an understanding of the context and behavior of a language. Furthermore, the implementation of a bilateral LSTM layer allows the model to learn words both prior and aft of the current input. Thus, the model can then learn the context surrounding the current word in a review. 

RNNs allow for sequential data of any length to be input to the model without jeopardizing the size and complexity of the model itself. The model architecture can remain constant regardless of input length. This is particularly advantageous in the context of processing text because it allows the maximum sequence length to become a hyperparameter. Therefore, the optimal sequence length can be searched for using a tuning method. The primary benefit of an RNN for this application is that the model learns from historical and future information. Furthermore, weights in the model are shared across time. 

The disadvantages associated with RNN are slow computation (depending on architecture), vanishing/exploding gradient, difficulty accessing information from a long time ago. It was necessary to employ a optimizer to compromise the vanishing gradient problem. 

The selected model architecture was shown below: 

<img src="https://user-images.githubusercontent.com/64614298/145319286-9b0acb07-387a-444e-8ec4-e0846dcad741.png" alt="drawing" width="600" class="center"/>

The embedding layer provides an improvement over sarse representations used in simpler bag of word model representations. A simple matrix multiplication is used to transform the words into their corresponding word embeddings or turns positive integers into dense vectors of a fixed size. The input size is simply the size of the vocabulary (number of uniqe words) or otherwise len(vocabulary) + 1. The input is a sequence of integers that represent certain words, with each integer being an index of a word map dictionary. 

# 5. Hyperparameter Tuning 

Deep Learning models, particularly RNNs, have proven to provide exceptional results in the field of NLP and text classification. It does so by recognizing patterns amongst the text inputs and stores these in a memory. The model then learns from its previous inputs in a sequential manner. To achieve optimal performance of an RNN model the most paramount step is training the model. 

The RNN model had several hyperparameters to tune. These parameters are used to control the learning process. The objective of tuning the model was to identify the optimal parameters in such a way that the trained model learns the data effectively with both time and fitting considerations. It was imperative to prevent overfitting and underfitting of the training data such that the model could adequately regularize well to new data. 

There were several available imports to utilize when determining the optimal hyperparameters. keras_tuner was selected to conduct the hyperparameter tuning. This module required that the model was defined as a function and each of the internal parameters were defined with their appropriate search scopes. The following 10 hyperparameters were considered in the tuning process:
- Embedding output dimension
- Embedding regularizer learning rate (l2)
- Number of units in a dense layer
- LSTM dropout rate
- LSTM layer output dimensions
- Learning rate for the optimizer
- Activation functions
- Momentum as part of LSTM layer
- Number of epochs
- Batch size

**Embedding Dimension**

The embedding layer is initialized with random weights and then learns an embedding for each word in the training corpus. The input dimension defines the *(total vocabulary size + 1) of the training data, i.e. the total number of unique words that an embedding will be created for. The output dimension defines the vector space in which the words will be embedded, for each word. The input length for this layer corresponds with the *maximum sequence length* defined earlier. 

**Dropout**

As per best practices, each LSTM layer was accompanied by a dropout layer. Dropout layers reduce the risk of overfitting the training data by randomly bypassing certain neurons. This helped to reduce the sensitivity to unique weights belonging to specific neurons. A widely accepted value of 0.2 was used as the defualt dropout rate, however, it was still necessary to tune and find the best value.   

**Number of Epochs**

The number of epochs determines the number of iterations that the model sees the training data. The range of values in which this parameter could be defined is infinite ranging from 1 to infinity. However, the scope of which to explore is stipulated by an early stopping method. The early stopping method limits the number of epochs based on the performance metric that it is measuring. 

The early stopping method was implemented using the Keras import, *callbacks*. The defined metric to measure was *val_accuracy*. Validation accuracy measures how well the predicted ratings are compared to the actual review ratings. In addition, validation loss was tracked. Validation loss is the quantified loss seen by the validation data set. The objective of the early stop method was to stop the model fitting once the validation loss increased in values for a specified number of epochs. The early stop method was designed to allow the validation accuracy to reach a maxmimum.   

**Batch Size**

Batch size establishes the quantity of smaples that will be fed to the model before the internal parameters of the model are upadated. A large batch size correlates with a large gradient jump. It is often usual to use a default sie of 128, or multiples of 32 up to 256. 

![Training_and_Validation_Tuned_Model](https://user-images.githubusercontent.com/64614298/167963991-4006651e-b820-41d1-bd8b-22f580acc8fb.png)

![Training_and_Validation_Losses_Tuned](https://user-images.githubusercontent.com/64614298/167964531-36f1b4a0-42f7-4cb1-90aa-49184d2d0f63.png)

# 6. Conclusion

Natural Language Processing (NLP) is a branch of AI that focuses on comprehending and extracting information from human languages such as text and voice. Sentiment analysis, chatbots, language translation, voice help, and speech recognition are all examples of common NLP applications. The purpose of this project was to prove that it is possible to train an artificial neural network to classify hotel review, which is a task that many private companies employ in order to determine the quality of the service being provided. 

Each day, roughly 2.5 quintillion bytes of data are generated. Most of them are unstructured such as text, audio, and other examples. Models such as RNN can help manage textual and voice data to make use of the bulk of this data and create meaning from it. NLP is a form of technology that aids in the extraction of meaning from certain kinds of data.

Using a three-class approach, the model was able to achieve a test accuracy and loss of 0.8250 and 0.8076, respectively. Combining the classes 1 and 2 as well as 4 and 5 improved the accuracy of the model immensely. This combination was justified because the language used in ratings 1 and 2, and 4 and 5, respectively, were very similar as illustrated in the data exploration. The bidirectional SLTM model was able to effectively the context of the text data by storing a memory of the words both prior and aft of the current time step. Overall, this model achieved the highest test accuracy and loss. 
