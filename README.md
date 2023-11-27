<img src="https://github.com/TLCLauraB/deep-learning-challenge/blob/main/img/alpha-color.png" width=15%  align=left>Module 21 Challenge: Nonprofit Future Speculation Using Deep Learning <br/>
Presented by Laura Bishop (TLCLauraB)<br/>
<br/>
<br/>
<br/>
<br/>
### Table of Contents:
* [Folder - Deep Learning Challenge](https://github.com/TLCLauraB/deep-learning-challenge/tree/main/Deep%20Learning%20Challenge)
    * [File: Starter_Code-LAB](https://github.com/TLCLauraB/deep-learning-challenge/blob/main/Deep%20Learning%20Challenge/Starter_Code-LAB.ipynb)
    * [File: AlphabetSoupCharity](https://github.com/TLCLauraB/deep-learning-challenge/blob/main/Deep%20Learning%20Challenge/AlphabetSoupCharity.h5)
    * [File: AlphabetSoupCharity_Optimization](https://github.com/TLCLauraB/deep-learning-challenge/blob/main/Deep%20Learning%20Challenge/AlphabetSoupCharity_Optimization.ipynb)
<br/>

## Overview of the Analysis:
The purpose of this analysis is to leverage machine learning and neural networks to develop a binary classifier for Alphabet Soup, a nonprofit foundation seeking to optimize its funding selection process. The dataset provided contains information on over 34,000 organizations that have received funding from Alphabet Soup, including details such as industry affiliation, government classification, organization type, financial status, and the effectiveness of the funding usage. The goal is to create a predictive model that can evaluate and classify organizations based on their likelihood of success if funded. By utilizing features such as application type, affiliation, and income classification, the classifier aims to assist Alphabet Soup in identifying applicants with the highest probability of effectively utilizing the funds provided, ultimately improving the foundation's decision-making process in selecting potential grant recipients.

## Results:

### Data Preprocessing:

* What variable(s) are the target(s) for your model?

In the provided dataset, the target variable for the model is "IS_SUCCESSFUL." This variable indicates whether the money provided by Alphabet Soup was used effectively by the organizations that received funding. It is a binary variable with two possible values:

    - 1: Indicates that the funding was used effectively.<br/>
    - 0: Indicates that the funding was not used effectively.<br/>

The goal of the binary classifier is to predict the value of "IS_SUCCESSFUL" for new applicants based on the features provided in the dataset, helping Alphabet Soup identify organizations with the best chance of success if funded.


* What variable(s) are the features for your model?

The features for the model are the various columns in the dataset that provide information about the organizations applying for funding. These features are used to predict the target variable, "IS_SUCCESSFUL." Here are some of the potential features based on the description of the dataset:

    - APPLICATION_TYPE: Alphabet Soup application type.
    - AFFILIATION: Affiliated sector of industry.
    - CLASSIFICATION: Government organization classification.
    - USE_CASE: Use case for funding.
    - ORGANIZATION: Organization type.
    - STATUS: Active status.
    - INCOME_AMT: Income classification.
    - SPECIAL_CONSIDERATIONS: Special considerations for the application.
    - ASK_AMT: Funding amount requested.
          
These features provide information about the nature of the organization, its purpose, industry affiliation, financial status, and other relevant factors. The model will use these features to learn patterns and relationships that can help predict whether an organization is likely to use the funding effectively. The selection of features is crucial in building an effective predictive model.

  
* What variable(s) should be removed from the input data because they are neither targets nor features?

The variables "EIN" (Employer Identification Number) and "NAME" (organization name) are typically neither targets nor features in the context of a machine learning model. These identification columns serve as unique identifiers for each organization and are not useful for predicting the target variable "IS_SUCCESSFUL." Including such identifiers in the model would not contribute meaningful information for the prediction task.

Therefore, it's common to remove these identification columns from the input data before building the machine-learning model. The model should focus on features relevant to the prediction task, such as the application type, affiliation, classification, and other factors that could influence the effectiveness of the funding. Removing unnecessary columns can improve the efficiency and performance of the model while avoiding the inclusion of irrelevant information.

### Compiling, Training, and Evaluating the Model:

* How many neurons, layers, and activation functions did you select for your neural network model, and why?

I utilized the Keras Sequential Model API in conjunction with TensorFlow to make three models:

**First Model:**<br/>
<img src="https://github.com/TLCLauraB/deep-learning-challenge/blob/main/img/model_1_test_results.png" height="85"><br/>

  **Neurons:** Two hidden layers, each with 5 neurons.<br/>
  **Layers:** Two hidden layers.<br/>
  **Activation Function:** ReLU for hidden layers, Sigmoid for the output layer.<br/>
  **Features:** Input layer with 37 features.<br/>

**Second Model:**<br/>
<img src="https://github.com/TLCLauraB/deep-learning-challenge/blob/main/img/model_2_test_results.png" height="85"><br/>

  **Neurons:** Three hidden layers, each with 5 neurons.<br/>
  **Layers:** Three hidden layers.<br/>
  **Activation Function:** ReLU for hidden layers, Sigmoid for the output layer.<br/>
  **Features:** Input layer with 33 features.<br/>

**Third Model:**<br/>
<img src="https://github.com/TLCLauraB/deep-learning-challenge/blob/main/img/model_3_test_results.png" height="85"><br/>

  **Neurons:** Four hidden layers, each with 10 neurons.<br/>
  **Layers:** Four hidden layers.<br/>
  **Activation Function:** ReLU for hidden layers, Sigmoid for the output layer.<br/>
  **Features:** Input layer with 33 features.<br/>

* Were you able to achieve the target model performance?<br/>

Despite implementing various architectural enhancements in the models, I fell short of achieving the targeted model performance. The accuracy goal was set at 75%, yet across all three models, the highest accuracy I attained was 72%. Despite the iterative adjustments made to the neural network's structure and complexity, reaching the desired accuracy remained elusive in this instance.


* What steps did you take in your attempts to increase model performance?<br/>

I implemented several architectural modifications to the Keras models. Notably, I introduced additional hidden layers to both the second and third models, a deliberate choice aimed at fostering a more complex representation of the underlying data. By making the network deeper, I aimed to help it recognize detailed patterns and complex structures, which might not have been noticeable in a less complex architecture. Furthermore, a key refinement involved augmenting the unit levels specifically within the hidden layers of the third model. I adjusted the model to better extract detailed features from the data, intending to enhance the neural network's ability to learn. By strategically adding layers and increasing unit levels, I aimed for a more comprehensive approach to improve the model's generalization and accuracy, ultimately boosting overall performance.

## Summary:

The deep learning models, despite their iterative enhancements, did not achieve the targeted accuracy of 75%, reaching only 72% accuracy. To address this classification problem more effectively, a different model architecture such as a gradient-boosting classifier could be explored. Gradient boosting, particularly models like XGBoost or LightGBM, has demonstrated efficacy in various classification tasks. These models excel in capturing complex relationships within the data, handling non-linearities, and achieving high predictive accuracy. Additionally, their ability to handle imbalanced datasets, interpret feature importance, and optimize hyperparameters makes them a promising alternative for improving performance in this specific context. The ensemble nature of gradient boosting, combining weak learners into a strong one, could better capture intricate patterns within the dataset and potentially yield the desired accuracy levels.

## Resources: 
  * https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
  * https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
  * https://xgboost.readthedocs.io/en/stable/
  * https://stackoverflow.com/questions/46943674/how-to-get-predictions-with-xgboost-and-xgboost-using-scikit-learn-wrapper-to-ma
  * https://datascience.stackexchange.com/questions/18903/lightgbm-vs-xgboost
