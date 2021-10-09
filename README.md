# Nueral_Network_Charity_Analysis
*Deep Learning ML*

## Project Overview
For this project, I am using **Neural Networks Machine Learning algorithms**, also known as artificial neural networks, or 
**ANN**. For coding, we used Python **TensorFlow** 
library in order to create a binary classifier that is capable of predicting whether applicants will be successful if funded by 
nonprofit foundation *Alphabet Soup*. This ML 
model will help ensure that the foundation’s money is being used effectively. With neural networks ML algorithms we are creating 
a robust deep learning neural network capable of 
interpreting large complex datasets. Important steps in neural networks ML algorithms are **data cleaning and data 
preprocessing** as well as decision **what data is beneficial for the model accuracy**.

## Resources
-  Dataset charity_data.csv
-  Software: Google Colab
-  Languages: Python
-  Libraries: Scikit-learn, TensorFlow, Pandas
-  Environment: Python 3.7

## The Process 
First we built the Pandas DataFrame we will be working with.    We used Pandas and Scikit-Learn StandardScaler() 
function to preprocess the dataset.   Next we will compile, train, and evaluate the neural network models 
effectiveness. 

<p align="center">
  <img width="1000" height=700" src="https://github.com/mjrotter4445/Nueral_Network_Charity_Analysis/blob/main/Graphics/Fig1_orig_df.jpg">
</p>
<p align="center">
Figure 1 - The Dataframe of Information we'll be working with 
</p>
     
      
## Data Reprocessing
          
                 
**What variable(s) are considered the target(s) for your model?**

In this case that is our “IS_SUCCESSFUL” column. Target variables are also known 
as dependent variable and we are using this variable to train our ML model.

**What variable(s) are considered to be the features for your model?**

Input values are also known as independent variables, and are considered to be 
features for the model. Those variables include all columns, except target 
variable and the one(s) we dropped “EIN" and "NAME” in the first trial (Figure 3) and 
only “EIN” (we kept the "NAME") in optimization trial (Figures 4, 5, 6, & 7).

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
overfitting the training data.
-  We utilized **relu activation function**, since it has best accuracy for this 
model.
 -  We used **adam optimizer**, which uses a gradient descent approach to ensure 
that the algorithm will not get stuck on weaker classifying variables and 
features and to enhance the performance of classification neural network.
As for the loss function, I used **binary crossentropy**, which is specifically 
designed to evaluate a binary classification model.
- The Model was trained on 100 epochs. I tried different epoch settings and the models made significant
continuous improvements as you can see in the results below.  

<p align="center">
  <img width="550" height=400" src="https://github.com/mjrotter4445/Nueral_Network_Charity_Analysis/blob/main/Graphics/Fig2_TFModelbuild.jpg">
</p>
<p align="center">
Figure 2 - The design of the TensorFlow Model to start our work with  
</p>


<p align="center">
  <img width="500" height=150 src="https://github.com/mjrotter4445/Nueral_Network_Charity_Analysis/blob/main/Graphics/Fig3_original_low_45perc.jpg">
</p>
<p align="center">
Figure 3 - Original lower performance accuracy = 45%
</p>

**Were you able to achieve the target model performance?**

Yes. After few configurations of number of hidden nodes we were able to achieve the target model performance. The model accuracy improved from 64% then 76.10% then to 76.20%, 
and eventually were were pleased to get to 77.20% with low loss at .46%.


<p align="center">
  <img width="550" height=150" src="https://github.com/mjrotter4445/Nueral_Network_Charity_Analysis/blob/main/Graphics/Fig4%20_add%20a%20layer_64perc.jpg">
</p>
<p align="center">
Figure 4 - Better with Optimization - accuracy at 64.57%
</p>



<p align="center">
  <img width="550" height=200" src="https://github.com/mjrotter4445/Nueral_Network_Charity_Analysis/blob/main/Graphics/Fig5_rebucketing_much_better_761perc.jpg">
</p>
<p align="center">
Figure 5 - Even Better with modifications to Optimization - accuracy at 76.10%
</p>



<p align="center">
  <img width="550" height=200" src="https://github.com/mjrotter4445/Nueral_Network_Charity_Analysis/blob/main/Graphics/Fig6_rebucketing_little_better_762perc.jpg">
</p>
<p align="center">
Figure 6 - and Better Performance at accuracy = 76.20%  
</p>



<p align="center">
  <img width="550" height=400" src="https://github.com/mjrotter4445/Nueral_Network_Charity_Analysis/blob/main/Graphics/Fig%207%20randomforest_BEST_772a.jpg">
</p>
<p align="center">
Figure 7 - and even Better Performance at accuracy = 77.20% - our Best so far.  
</p>

**What steps did you take to try and increase model performance?**

In order to increase model performance,we took the following steps:

-   Checked input data and brought back NAME column, that was initially skipped. We
set a **condition on that value for any that are less than 50 in “Other” group**. This approach 
reduced the number of unique categorical values by binning the values. **Noisy variables reduced by binning.**
-  **Binned the ASK_AMT values**.
-  **Added neurons to hidden layers and added hidden layers**.
-  At first, I added the third layer with 40 neurons; however, I’ve changed back to 
2 layers, because the results did not improve much if any.  Increase neurons for each layer (200 for 1st, 90 for 2nd). 
-  **Increased Epochs to 500**.
-  **Models weights are saved very 5 epochs**.
-  **and FINALLY, added The Random Forest Algorithm** for the best performance of all in Figure 7.


# Summary

*Summary of the results*:

The model loss and accuracy score tell us how well the model does with the 
dataset and parameters that we build the model. In the end, the most optimal model we
ran was the last one - Figure 7 above.  We were pleased that we were able to get the 
loss score down to .46, meaning the probability model to fail is 46% and 
accuracy score up to 0.772, meaning that the probability model to be accurate is 77.20%.

*Recommendation for further analysis*:

After a lot of experimenting and fine-tuning to the model we were able to continuously improve the accuracy score 
and eventually get up to 76% accuracy and 74% loss as you can see in Figure 6.  This is not necessarily the 
best model for this dataset. The loss score for that model was still so high at about 74%.  

Random Forests Algorithm might be the best fit for this dataset because the performance is better.  
In Figure 7 above we can see the results.  We were pleased that we were able to get the 
loss score down to .46, meaning the probability model to fail is 46% and 
accuracy score up to 0.772, meaning that the probability model to be accurate is 77.20%.

Adding new input values seemed a good choice when improving the model accuracy. In this case 
I would consider adding more input values (if there are some available in the original 
dataset, for example). Another thing we could do, is to consider gathering more 
data. Although gathering more data is not always the easy decision but it is sometimes 
necessary.

References and Sources of Code from Class Materials: 
-  Module 19.2.2 Build a Basic Neural Programming, 
-  Module 19.3.3 Practice Encoding Categorical Variables
-  Module 19.3.4 Span the Gap Using Standardization
-  Module 19.4.2 Real Data, Real Practice Imports and Setup
-  Module 19.4.3 Real Data, Real Practice Preprocessing
-  Module 19.4.4 Real Data, Real Practice Deep Learning Model Design
-  Module 19.4.5 Real Data, Real Practice Train and Evaluate the Model
-  Module 19.6.1 Checkpoints Are Not Just for Video Games
-  Module 19.6.2 For Best Results, Please Save After Training            

                 
