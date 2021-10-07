# Nueral_Network_Charity_Analysis
*Deep Learning ML*

## Project Overview
For this project, I am using **Neural Networks Machine Learning algorithms**, also known as artificial neural networks, or **ANN**. For coding, we used Python **TensorFlow** 
library in order to create a binary classifier that is capable of predicting whether applicants will be successful if funded by nonprofit foundation *Alphabet Soup*. This ML 
model will help ensure that the foundation’s money is being used effectively. With neural networks ML algorithms we are creating a robust deep learning neural network capable of 
interpreting large complex datasets. Important steps in neural networks ML algorithms are **data cleaning and data preprocessing** as well as decision **what data is beneficial for** the model accuracy.

## Resources
-  Dataset charity_data.csv
-  Software: Jupyter Notebook
-  Languages: Python
-  Libraries: Scikit-learn, TensorFlow, Pandas
-  Environment: Python 3.7

## Results

<p align="center">
  <img width="550" height=400" src="hxxxxx">
</p>
<p align="center">
Figure 1 - The Dataframe of Information we'll be working with 
</p>
                 

                 
## Data Reprocessing
          
                 
**What variable(s) are considered the target(s) for your model?**

In this case that is our “IS_SUCCESSFULL” column. Target variables are also known 
as dependent variable and we are using this variable to train our ML model.

**What variable(s) are considered to be the features for your model?**

Input values are also known as independent variables, and are considered to be 
features for the model. Those variables include all columns, except target 
variable and the one(s) we dropped “EIN" and "NAME” in the first trial and “EIN” 
in optimization trial.

**What variable(s) are neither targets nor features, and should be removed from the input data?**

The variables that should be removed and are neither targets nor features are 
variables that are meaningless for the model. The variables that don’t add to the 
accuracy to the model. One of the examples would be variables with all unique 
values. Another thing to keep in mind is to take care of the Noisy data and 
outliers. We can approach to this by dropping outliers or bucketing.

## Compiling, Training, and Evaluating the Model
**How many neurons, layers, and activation functions did you select for your neural network model, and why?**

 - We utilized 2 layers, because 3 layers didn’t contribute much to the improvement of 
the ML module. This is because the additional layer was redundant—the complexity 
of the dataset was encapsulated within the two hidden layers. Adding layers does 
not always guarantee better model performance, and depending on the complexity of 
the input data, adding more hidden layers will only increase the chance of 
overfitting the training data (A).
-  We utilized **relu activation function**, since it has best accuracy for this 
model.
-  I used 200 neurons for first layer and 90 neurons for second layer. As 
recommended first layer should have at least double the amount of input features, 
that is 100 input values (rows) in our case.
-  I used **adam optimizer**, which uses a gradient descent approach to ensure 
that the algorithm will not get stuck on weaker classifying variables and 
features and to enhance the performance of classification neural network (B).
As for the loss function, I used **binary crossentropy**, which is specifically 
designed to evaluate a binary classification model (B).
-  Model was trained on xxxxxxxxx500 epochs. I increase from xxxxxxxxxx200 epoch because the model 
improved a bit; however I did not increased for too many epoch in order to avoid 
overfitting.

Figure 1: Defining a Model.

**Were you able to achieve the target model performance?**

Yes. After few configurations of number of hidden nodes we were able to achieve the target model performance. 
xxThe model accuracy improved to xxxxxxxxx76.30%. Figures below show accuracy score after 
xxxoptimization at xxxxxxxx76.30% and before optimization at xxxxxxxxxxx72.41%.

<p align="center">
  <img width="550" height=400" src="Original Low xxxxxxxxxxx.jpg">
</p>
<p align="center">
Figure 2 orignal low performance accuracy = .45 
</p>


**What steps did you take to try and increase model performance?**

In order to increase model performance,we took the following steps:

xxxxChecked input data and brought back NAME column, that was initially skipped. I 
set a condition on the values that are less than 50 in “Other” group. That 
reduced the number of unique categorical values by binning the values.
Binned the ASK_AMT values.
xxxxxAt first, I added the third layer with 40 neurons; however, I’ve changed back to 
2 layers, because the results did not improve much if any.
Increase neurons for each layer (200 for 1st, 90 for 2nd).
Increase Epochs to 500.

<p align="center">
  <img width="550" height=400" src="Original Low xxxxxxxxxxx.jpg">
</p>
<p align="center">
Figure 3 accuracy after optimzation 
</p>
Figure 3: Accuracy After Optimization.

## Summary
*Summary of the results*

xxxxxThe model loss and accuracy score tell us how well the model does with the 
dataset and parameters that we build the model. Loss score is equal to 0.609, 
meaning the probability model to fail is 60.89% and accuracy score is 0.7630, 
xxxmeaning that the probability model to be accurate is 76.30%.

Recommendation for further analysis

After some fine-tuning the model reach accuracy score of 67.30%. Although the 
model reached the required criteria it might not be the best model for this 
dataset. The loss score for that model is still about 60%, what is quite high. 
Dataset that we were working on seemed good fit because of the length of the 
dataset and its complexity, even though the results weren't the best. Adding new 
input values seemed a good choice when improving the model accuracy. In this case 
I would consider adding more input values (if there are available in the original 
dataset, for example). Another thing we could do, is to consider gathering more 
data. Although gathering more data is not always the easy decision is sometimes 
necessary.

References
(A) Module 19.2.2 Build a Basic Neural Programming, 
(C) Module 19.3.3   
(C) Module 19.4.2  
(D) Module 19.4.4 Deep learning model design,  

                 
