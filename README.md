# Hands-onMachineLearning
Studies about the book
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
Book by Aurelien Geron - Second Edition

What Is Machine Learning?

* Machine Learning is the science (and art) of programming computers so they can\
  learn from data.\
  Here is a slightly more general definition:\
  \[Machine Learning is the\] field of study that gives computers the ability to learn\
  without being explicitly programmed.\
  —Arthur Samuel, 1959\
  And a more engineering-oriented one:\
  A computer program is said to learn from experience E with respect to some task T\
  and some performance measure P, if its performance on T, as measured by P, improves\
  with experience E.\
  —Tom Mitchell, 1997

Types of Machine Learning Systems

* Supervised learning\
  In supervised learning, the training data you feed to the algorithm includes the desired\
  solutions, called labels
  * k-Nearest Neighbors\
    • Linear Regression\
    • Logistic Regression\
    • Support Vector Machines (SVMs)\
    • Decision Trees and Random Forests\
    • Neural networks 2
* Unsupervised learning\
  In unsupervised learning, as you might guess, the training data is unlabeled\
  (Figure 1-7). The system tries to learn without a teacher.

  • Clustering\
  — K-Means\
  — DBSCAN\
  — Hierarchical Cluster Analysis (HCA)\
  • Anomaly detection and novelty detection\
  — One-class SVM\
  — Isolation Forest
* • Visualization and dimensionality reduction\
  — Principal Component Analysis (PCA)\
  — Kernel PCA\
  — Locally-Linear Embedding (LLE)\
  — t-distributed Stochastic Neighbor Embedding (t-SNE)\
  • Association rule learning\
  — Apriori\
  — Eclat
* A related task is dimensionality reduction, in which the goal is to simplify the data\
  without losing too much information. One way to do this is to merge several correla‐\
  ted features into one. For example, a car’s mileage may be very correlated with its age,\
  so the dimensionality reduction algorithm will merge them into one feature that rep‐\
  resents the car’s wear and tear. This is called feature extraction.
* Yet another important unsupervised task is anomaly detection—for example, detect‐\
  ing unusual credit card transactions to prevent fraud, catching manufacturing defects,\
  or automatically removing outliers from a dataset before feeding it to another learn‐\
  ing algorithm. The system is shown mostly normal instances during training, so it\
  learns to recognize them and when it sees a new instance it can tell whether it lookslike a normal one or whether it is likely an anomaly (see Figure 1-10). A very similar\
  task is novelty detection: the difference is that novelty detection algorithms expect to\
  see only normal data during training, while anomaly detection algorithms are usually\
  more tolerant, they can often perform well even with a small percentage of outliers in\
  the training set.
* is association rule learning, in which the goal is to dig into large amounts of data and discover interesting relations between attributes. For example, suppose you own a supermarket. Running an association rule on your sales logs may reveal that people who purchase barbecue sauce and potato chips also tend to buy steak. Thus, you may want to place these items close to each other.
* Semisupervised learning
  * Some algorithms can deal with partially labeled training data, usually a lot of unla‐\
    beled data and a little bit of labeled data.
* Reinforcement Learning
  * The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative rewards, as in Figure 1-12). It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.
* Batch learning\
  In batch learning, the system is incapable of learning incrementally: it must be trained\
  using all the available data. This will generally take a lot of time and computing\
  resources, so it is typically done offline. First the system is trained, and then it is\
  launched into production and runs without learning anymore; it just applies what it\
  has learned. This is called offline learning.
* Online learning
  * In online learning, you train the system incrementally by feeding it data instances\
    sequentially, either individually or by small groups called mini-batches. Each learning\
    step is fast and cheap, so the system can learn about new data on the fly, as it arrives
  * Online learning is great for systems that receive data as a continuous flow (e.g., stock\
    prices) and need to adapt to change rapidly or autonomously.
  * A big challenge with online learning is that if bad data is fed to the system, the sys‐\
    tem’s performance will gradually decline.
* In summary:\
  • You studied the data.\
  • You selected a model.\
  • You trained it on the training data (i.e., the learning algorithm searched for the\
  model parameter values that minimize a cost function).\
  • Finally, you applied the model to make predictions on new cases (this is called\
  inference), hoping that this model will generalize well.

Main Challenges of Machine Learning - "bad algorithm" or "bad data"

* Insufficient Quantity of Training Data
  * As the authors put it: “these results suggest that we may want to reconsider the trade-\
    off between spending time and money on algorithm development versus spending it\
    on corpus development.”
  * Peter Norvig  - "The Unreasonable Effectiveness of Data"
* Nonrepresentative Training Data
  * It is crucial to use a training set that is representative of the cases you want to general‐\
    ize to. This is often harder than it sounds: if the sample is too small, you will have\
    sampling noise (i.e., nonrepresentative data as a result of chance), but even very large\
    samples can be nonrepresentative if the sampling method is flawed. This is called\
    **sampling bias**.
* How else can you get a large training set?
* Poor-Quality Data
  * If some instances are clearly outliers, it may help to simply discard them or try to\
    fix the errors manually.
  * If some instances are missing a few features (e.g., 5% of your customers did not\
    specify their age), you must decide whether you want to ignore this attribute alto‐\
    gether, ignore these instances, fill in the missing values (e.g., with the median\
    age), or train one model with the feature and one model without it, and so on.
* Irrelevant Features
  * feature engineering
    * Feature selection
      * selecting the most useful features to train on among existing\
        features.
    * Feature extraction
      * combining existing features to produce a more useful one (as\
        we saw earlier, dimensionality reduction algorithms can help).
* Overfitting the Training Data
  * it means that the model performs well on the training data, but it does not generalize\
    well.
  * Overfitting happens when the model is too complex relative to the\
    amount and noisiness of the training data. The possible solutions\
    are:\
    • To simplify the model by selecting one with fewer parameters\
    (e.g., a linear model rather than a high-degree polynomial\
    model), by reducing the number of attributes in the training\
    data or by constraining the model\
    • To gather more training data\
    • To reduce the noise in the training data (e.g., fix data errors\
    and remove outliers)
  * Constraining a model to make it simpler and reduce the risk of overfitting is called\
    regularization.
  * You want to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well.
* Underfitting the Training Data
  * it occurs when your model is too simple to learn the underlying structure of the data.
* The system will not perform well if your training set is too small, or if the data is\
  not representative, noisy, or polluted with irrelevant features (garbage in, garbage\
  out). Lastly, your model needs to be neither too simple (in which case it will\
  underfit) nor too complex (in which case it will overfit).
* Testing and Validating
  * A better option is to split your data into two sets: the training set and the test set. 
  * The error rate on new cases is called the generalization error (or out-of-\
    sample error), and by evaluating your model on the test set, you get an estimate of this\
    error. This value tells you how well your model will perform on instances it has never\
    seen before.
  * If the training error is low (i.e., your model makes few mistakes on the training set)\
    but the generalization error is high, it means that your model is overfitting the train‐\
    ing data.
* Hyperparameter Tuning and Model Selection
  * how do you choose the value of the regularization hyperparameter? 
    * holdout validation: you simply hold out part of the training set to evaluate several candidate models and select the best one.
    * validation set
      * you train multiple models with various hyperparameters on the reduced training set (i.e., the full training set minus the validation set), and you select the model that performs best on the validation set.
  * 

Exercícios

 1. How would you define Machine Learning? 

    Machine Learning is about building systems that can learn from data. Learning means getting better at some task, given some performance measure.
 2. Can you name four types of problems where it shines? 

    Machine Learning is great for complex problems for which we have no algorithmic solution, to replace long lists of hand-tuned rules, to build systems that adapt to fluctuating environments, and finally to help humans learn (e.g., data mining).
 3. What is a labeled training set? 

    A labeled training set is a training set that contains the desired solution (a.k.a. a label) for each instance.
 4. What are the two most common supervised tasks? 

    The two most common supervised tasks are regression and classification.
 5. Can you name four common unsupervised tasks? 

    Common unsupervised tasks include clustering, visualization, dimensionality reduction, and association rule learning.
 6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains? 

    Reinforcement Learning is likely to perform best if we want a robot to learn to walk in various unknown terrains, since this is typically the type of problem that Reinforcement Learning tackles. It might be possible to express the problem as a supervised or\
    semisupervised learning problem, but it would be less natural.
 7. What type of algorithm would you use to segment your customers into multiple groups?

    If you don’t know how to define the groups, then you can use a clustering algorithm (unsupervised learning) to segment your customers into clusters of similar customers. However, if you know what groups you would like to have, then you can feed many examples of each group to a classification algorithm (supervised learning), and it will classify all your customers into these groups.
 8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem? 

    Spam detection is a typical supervised learning problem: the algorithm is fed many emails along with their labels (spam or not spam).
 9. What is an online learning system? 

    An online learning system can learn incrementally, as opposed to a batch learning system. This makes it capable of adapting rapidly to both changing data and autonomous\
    systems, and of training on very large quantities of data.
10. What is out-of-core learning? 

    Out-of-core algorithms can handle vast quantities of data that cannot fit in a computer’s main memory. An out-of-core learning algorithm chops the data into mini-batches and uses online learning techniques to learn from these mini-batches.
11. What type of learning algorithm relies on a similarity measure to make predictions? 

    An instance-based learning system learns the training data by heart; then, when given a new instance, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.
12. What is the difference between a model parameter and a learning algorithm’s hyperparameter?

    A model has one or more model parameters that determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).
13.  What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions? 

    Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually tr-\*ain such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions, we feed the new instance’s features into the model’s prediction function, using the parameter values found by the learning algorithm.
14. Can you name four of the main challenges in Machine Learning?

    Some of the main challenges in Machine Learning are the lack of data, poor data quality, nonrepresentative data, uninformative features, excessively simple models that\
    underfit the training data, and excessively complex models that overfit the data.
15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions? 

    If a model performs great on the training data but generalizes poorly to new instances, the model is likely overfitting the training data (or we got extremely lucky on the training data). Possible solutions to overfitting are getting more data, simplifying the model (selecting a simpler algorithm, reducing the number of parameters or features used, or regularizing the model), or reducing the noise in the training data.
16. What is a test set, and why would you want to use it?

    A test set is used to estimate the generalization error that a model will make on new instances, before the model is launched in production. 
17. What is the purpose of a validation set? 

    A validation set is used to compare models. It makes it possible to select the best model and tune the hyperparameters.
18. What is the train-dev set, when do you need it, and how do you use it? 

    The train-dev set is used when there is a risk of mismatch between the training data and the data used in the validation and test datasets (which should always be as close as possible to the data used once the model is in production). The train-dev set is a part of the training set that’s held out (the model is not trained on it). The model is trained on the rest of the training set, and evaluated on both the train-dev set and the validation set. If the model performs well on the training set but not on the train-dev set, then the model is likely overfitting the training set. If it performs well on both the training set and the train-dev set, but not on the validation set, then there is probably a significant data mismatch between the training data and the validation + test data, and you should try to improve the training data to make it look more like the validation + test data.
19. What can go wrong if you tune hyperparameters using the test set?

    If you tune hyperparameters using the test set, you risk overfitting the test set, and the generalization error you measure will be optimistic (you may launch a model that performs worse than you expect).

Chapter 2 End-to-End Machine Learning Project

 1. Look at the big picture
 2. Get the data
 3. Discover and visualize the data to gain insights
 4. Prepare the data for Machine Learning algorithms
 5. Select a model and train it
 6. Fine-tune your model
 7. Present your solution
 8. Launch, monitor, and maintain your system
 9. Frame the Problem and Look at the Big Picture
     1.  Define the objective in business terms?
     2. How will your business solution be used?
     3. What are the current solutions/workarounds (if any)?
     4. How should you frame this problem (supervised/unsupervised, online/offline, etc)?
        1. tip
        2. if the data were huge, you could either split your batch learning work across multiple servers (using the MapReduce technique) or use an online learning technique
     5. How should perfomance be measured?
        1. RMSE - root mean square error
        2. when there are many outliers districts 
           1. MAE - mean absolute error | average absolute deviation
        3. ways to measure the distance between two vectors (distance measure or norms)
           1. Euclidean norm - computing the root of a sum of squares
     6. Is the performance measure aligned with the business objective?
     7. What would be the minimum performance needed to reach the business objective?
     8. What are comparable problems? Can you reuse experience or tools?
     9. Is human expertise available?
    10. How would you solve the manually?
    11. List the assumptions you (or others) have made so far?
    12. Verify assumptions if possible?
    13. 
    14. 
10. Pipelines
    1. is a sequence of data processing components 
    2. components typically run asynchronously
    3. each component pulls in a large amount of data, processes it, and spits out the result in another data store.
    4. the interface between component is simple data store
11. this is a **multiple regression** problem since the system will use multiple features to make a prediction (it will use the district’s population, the median income, etc.).
12. **univariate regression** problem when to predict a single value for each instance, is opositive for **multivariate regression**
13. **Histograma**
    1. shows the number of instances (on the horizontal axis) that have a given value range (on the horizontal axis)
    2. d
14. stratified sampling - strata
    1. They try to ensure that these 1,000 people are representative of the whole population. For example, the US population is composed of 51.3% female and 48.7% male, so a well-conducted survey in the US would try to maintain this ratio in the sample: 513 female and 487 male. This is called stratified sampling: the population is divided into homogeneous subgroups called strata, and the right number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population.
    2. **Stratified sampling** is a type of **sampling** method in which the total population is divided into smaller groups or strata to complete the **sampling** process. The strata is formed **based** on some common characteristics in the population data.
15. **Discover and Visualize the Data to Gain Insights**
16. **Looking for Correlations**
    1. standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr().
    2. ranges from –1 to 1
    3. correlation coefficient only measures linear correlations (“if x goes up, then y generally goes up/down”).
17. **Experimenting with Attribute Combinations**
18. **Prepare the Data for Machine Learning Algorithms**
19. **Data Cleaning**
    1. **Scikit Learn Desing**
       1. **Consistency**. All objects share a consistent and simple interface:
          1. Estimators
             1. Any object that can estimate some parameters based on a dataset\
                is called an estimator (e.g., an imputer is an estimator).
          2. Transformers
             1. the transformation is performed by the transform() method with the dataset to transform as a parameter. It returns the transformed dataset.
          3. Predictors
             1. A predictor has a predict() method that takes a dataset of new instances and returns a dataset of corresponding predictions. It also has a score() method that measures the quality of the predictions given a test set (and the corresponding labels in the case of supervised learning algorithms).
       2. **Inpection**
          1. All the estimator’s hyperparameters are accessible directly via public\
             instance variables (e.g., imputer.strategy ), and all the estimator’s learned\
             parameters are also accessible via public instance variables with an underscore\
             suffix (e.g., imputer.statistics_ ).
       3. **Nonproliferation of classes**
          1. Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers.
       4. **Composition**
          1. Existing building blocks are reused as much as possible.
       5. **Sensible defaults**
          1. Scikit-Learn provides reasonable default values for most parameters
    2. **Handling Text and Categorical Attributes**
       1. convert these categories from text to numbers
          1. Scikit-Learn’s OrdinalEncoder class
             1. One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values. This may be fine in some cases (e.g., for ordered categories such as “bad”, “average”, “good”, “excellent”)
       2. convert categorical values into one-hot vectors
          1. OneHotEncoder class
             1. To fix this issue, a common solution is to create one binary attribute per category: one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is “INLAND” (and 0 otherwise), and so on. This is called one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold)
          2. Custom Transformers
             1. you will need to write your own for tasks such as custom cleanup operations or combining specific attributes
             2. create a class and implement three methods: fit() (returning self ), transform() , and fit_transform() . You can get the last one for free by simply adding TransformerMixin as a base class. Also, if you add BaseEstimator as a base class (and avoid \*args and \*\*kargs in your constructor) you will get two extra methods ( get_params() and set_params() ) that will be useful for automatic hyperparameter tuning.
       3. **Feature Scaling**
          1. With few exceptions, Machine Learning algorithms don’t perform well when\
             the input numerical attributes have very different scales.
          2. **Min-max scaling** (many people call this normalization) is quite simple: values are\
             shifted and rescaled so that they end up ranging from 0 to 1
             1. Scikit-Learn → MinMaxScaler → feature_range hyperparameter that lets you change the range
          3.  **Standardization** first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the standard deviation so that the resulting distribution has unit variance.
             1. Is much less affected by outliers
                1. For example, suppose a district had a median income equal to 100 (by mistake). Min-max scaling would then crush all the other values from 0–15 down to 0–0.15, whereas standardization would not be much affected. 
             2. Scikit-Learn → StandardScaler
          4. 
    3. **Transformation Pipelines**
    4. 

d

d

* 

Links

* [100_Days_of_ML_Code](https://github.com/llSourcell/100_Days_of_ML_Code)
* [DrivenData](https://www.drivendata.org/)

# Snippet

```
https://github.com/ageron/handson-ml2
https://raw.githubusercontent.com/ageron/handson-ml2/master/
https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/lifesat/oecd_bli_2015.csv

urllib.request.urlretrieve(url, datapath + filename)
```
