# AI
## Group information
Group OB_19

The team members are:

William-Alexandre Messier 40208650

Tarek Elalfi 40197527

Mario El Shaer 40210124

## Content of files and their purpose:
1. **Images folder**: it is divided into two folders one for training and one for testing. The two folders contain datasets/classes (happy, focused, anger, neutral) those labelled filtered datasets were used for the analysis.

    The dataset is from:
   
   (N. Segal, “Facial Expression Training Data”, Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/noamsegal/affectnet-training-data/data)

2.  **Python script(dataset_visualization.py)**: The Python script is to generate bar charts and pixel intensity charts for the datasets collected

## To run the dataset_visualization.py (data cleaning&data visuialoztion)
1. install the repository
2. open the project on a python environment
3. ensure that the dataset_visualization.py is opened in the complier you are in
4. install all the necessary libraries (numpy, matplotlib, etc...) *you can hover over the errors for the libraries importations on top and click on the action it will download the libraries*
5. the code will run, analyze the images and generate the charts.  

## To run the train_eval.py (model training and evaluation)
1. install the repository
2. move the newly created folder to your desktop
3. open the project on a python environment
4. ensure that the train_eval.py is opened in the compiler
5. intall all the necessary libraries *you can hover over the errors for the libraries importantions on top and click on the action it will download the librairies*
6. the code will run, train an ai model
7. once train_eval is ran, run evaluate_models.py. it will create a confusion matrix for each model
