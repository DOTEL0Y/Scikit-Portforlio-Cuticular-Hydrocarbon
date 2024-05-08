# Scikit Portforlio: Cuticular Hydrocarbon - Supervised Learning Model
## Installation Guide/Dependencies 
Python Dependencies include **pandas, Scikit, Matplotlib,** and lastly **Seaborn**.
Last two libraries were strictly for data visualization purposes and are not mandatory for execution.

**Terminal Commands**

`pip install scikit-learn`

`pip install Matplotlib`

`pip install seaborn`

## Introduction
With the assistance of **GC-MS** (Gas Chromatography-Mass spectrometry), an instrument to identify the structure of molecular components was utilized to quantify key molecules typically present in Honeybee Queen’s cuticle. These molecules consist of cuticular hydrocarbon (chc) and lipids or esters. These chc and lipids are responsible for crucial biological functionality within queen bees and are indicators of fertility and stressors.  With a dataset reflective of five diets concerning human impacts on nature; the goal was to construct a machine-learning model that would be effective in identifying the diet/treatment of a queen bee. For reference of the treatment groups look at figure 1. At first, the algorithm **ADABOOST** was used, and unsatisfactory accuracy shed some light on the issues of the dataset. Analyzing the data and its properties was crucial for success.
![Treatment Groups](/markdown/treatmentGroup.PNG)
After a few trials and errors with different ML models, it was understood that the dataset was too meager in size for a classifier ML model. Augmentation of chemistry data was mandatory at this stage. Of course, the approach couldn’t be random and needed to be augmented through constraints that were sensible to the chemistry present in the species of honeybees. In essence, this project is a semi-supervised learning model since the augmented data were mere constructs of tangible data. As technology evolves, we humans try to exploit its power to form introspection of our physical world and the effects we impose on it. The study of biology is no different in wanting to use this power. Some insights are difficult to see from the perspective of a microscope. Machine learning algorithms provide magnification that is not only significant in finding sought-out questions but also revealing those questions we never asked. As a species, our evolutionary dominance has made the smaller parts of our world enigmatic, but no matter how small it doesn’t take away its role in a stable ecosystem.
