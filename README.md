# Scikit Portforlio
### Installation Guide 
Python Dependencies include **pandas, Scikit, Matplotlib,** and lastly **Seaborn**.
Last two libraries were strictly for data visualization purposes and are not mandatory for execution.

#### **Terminal Commands**

`pip install scikit-learn`

`pip install Matplotlib`

`pip install seaborn`

## Introduction
With the assistance of **GC-MS** (Gas Chromatography-Mass spectrometry), an instrument to identify the structure of molecular components was utilized to quantify key molecules typically present in Honeybee queen’s cuticle. These molecules consist of cuticular hydrocarbon (chc) and lipids or esters. These chc and lipids are responsible for crucial biological functionality within queen bees and are indicators to fertility and stressors.  With a dataset reflective of five diets concerning human impacts on nature; the goal was to construct a machine-learning model that would be effective in identifying the diet/treatment of a queen bee. For reference of the treatment groups look at figure 1. At first, the algorithm **ADABOOST** was used, and unsatisfactory accuracy shed some light on the issues of the dataset. Analyzing the data and its properties was crucial for success.
![Treatment Groups](/markdown/treatmentGroup.PNG)
After a few trials and errors with different ML models, it was understood that the dataset was too meager in size for a classifier ML model. Augmentation of chemistry data was mandatory at this stage. Of course, the approach couldn’t be random and needed to be augmented through constraints that were sensible to the chemistry present in the species of honeybees. In essence, this project is a semi-supervised learning model since the augmented data were mere constructs of tangible data. As technology evolves, we humans try to exploit its power to form introspection of our physical world and the effects we impose on it. The study of biology is no different in wanting to use this power. Some insights are difficult to see from the perspective of a microscope. Machine learning algorithms provide magnification that is not only significant in finding sought-out questions, but also revealing those questions we never asked. As a species, our evolutionary dominance has made the smaller parts of our world enigmatic, but no matter how small it doesn’t take away its role in a stable ecosystem.
## Methods
The experimental project was carried out in three stages. First, I exercised trivial approaches using ADABOOST in conjunction with a decision tree classifier, and Gradient Boost for output comparison. Secondly, augmentation of the dataset with two fabrications of the dataset was constructed with different sizes. Lastly, utilization of Leave-One-Out Cross-validation on three datasets, the original dataset, small, augmented data, and the large, augmented data for accuracy comparison.
### 3.1. Phase One - Trivial Approach
The ADABOOST algorithm performs best with binary data due to its Boolean logic, (yes, or no; 1 or 0 etc.) for the resigning of weights in classifiers so dealing with float values can be challenging and potentially negatively impact the training. See figure 4. To maximize the weak learners within the ADABOOST algorithm, I added the Decision Tree classifier with a max depth of 24 to reflect the number of classifications. With one-hundred weak learners for both GradientBoost and AdaBoost; the weak learners rank the classifiers within the dataset during each iteration to achieve lower error for optimal predictions. For a visual representation look at figure 4 of this approach on the original data and figure 5 for the ADABOOST Boolean logic decision tree.
![AccuracyScore,Matplotlib](figures/3d%20Figure.png)
    *Figure 4. Matplotlib - 3D Accuracy of ADABOOST/GRADIENTBOOST/DECISIONTREE classifier*

Trivial approach with a ratio of 1:1 for the training/testing portion of this phase. Half the dataset is being fed into the different training models and the other half for predictions. As part of the identification process, a function designed to reshape the treatment group labels into integers for the sake of finding overlap would later be improved as the accuracy score of the model improved.

![Stumpillustration](markdown/stump.png)

*Figure 5. ADABOOST Boolean logic Decision Tree*

Trivial approach with a ratio of 1:1 for the training/testing portion of this phase. Half the dataset is being fed into the different training models and the other half for predictions. As part of the identification process, a function designed to reshape the treatment group labels into integers for the sake of finding overlap would later be improved as the accuracy score of the model improved.

### 3.2. Phase Two - Augmentation

For more desirable results, augmentation of the data was necessary. The process of augmentation needed to be sensible to the chemical signature present within each treatment group. To combat the variability, standardize the data based on each treatment group. The reason for this is to avoid underfitting. The function would take the totality of samples from one treatment group and then normalize the data. See Figure 4 for reference. Furthermore, randomness came from subtracting or adding from the median of each chemical, and condition if the value reduced itself below zero to set it to zero as negative of any chemical signatures isn’t sensible. Cuticular hydrocarbon 21 for example is one of the chemical signatures and the presence of it in each treatment group is due to the treatment’s effects on their social harmony and maturing state. All these chemicals are associated with some functionality of cuticular or hormone communication. To keep these representative functionalities consistent within the augmentation this was the approach that was adopted. See Figures 6 and 7 for reference of this concept of c21 in the original and augmented data.[3]. The first augmented data contains 140 samples, 40 of which are from the original data. The second augmented contains 540 samples, 40 of which are from the original data as well.
