# Sign Language to Text
The model developed can classify any letter in the American Sign Language (ASL) alphabet into the equivalent letter in the English Alphabet which is a muitlclass classification problem.

Using this model we can convert the ASL into the respective english sentences quite easily.

The deployed version of the model can be view [here](https://sathwick-reddy-m-sign-language-to-text-web-app-6j4rww.streamlitapp.com/)

## Model Characterstics
### Input
A gray scale ASL letter image of shape (28X28)
### Output 
Equivalent letter in the English alphabet corresponding to the ASL input image

**Note**: It cannot covert the letters J and Z as there corresponding representation in ASL requires motion.
### Evaluation
The model is evaluated on the accuracy
### Architecture
![CNN](./model.png)

## Workflow
1. Importing the necessary libraries
2. Preprocessing the input data
3. Defining the Model
4. Fitting the training data
5. Hyper Parameter Tuning
6. Selection of the best model
7. Testing the performance on the test set

## Best Model After Hyperparameter Tuning
`models/experiment-dropout-0`

Tuned Hyperparmeter include
1. Number of Convolution and Max Pooling Pairs
2. Number of feature maps in the convolution layers
3. Filter Shape
4. Dropout


Techniques that can be explored include

1. Data Augmentation
2. Batch Normalization
3. Number of units in the dense layer
4. Replacing the Max Pooling layer with a convolution layer having stride > 1