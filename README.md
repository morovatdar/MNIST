# MNIST and COIL100 Classification
Here, we introduce the PyTorch libraries in Python, and we will discuss the two models that we will use to classify our datasets. 
The models are 
- conventional Neural Network (NN) or Multi-Layer Perceptron (MLP), 
- and Convolutional Neural Network (CNN).
  
![image](https://user-images.githubusercontent.com/83058686/217935667-4683138e-29d2-437e-a008-a6f1f0cbd222.png)
![image](https://user-images.githubusercontent.com/83058686/217935717-a729cb96-267d-4924-8c86-2dc818b2dcc0.png)
 
# MNIST
## Loading Data
PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow us to use pre-loaded datasets as well as our own data. 
Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
  
Fortunately, PyTorch domain libraries provide a number of pre-loaded datasets, including MNIST that subclass torch.utils.data.Dataset and 
implement functions specific to the particular data. So, we do not need to download the dataset independently.
  
Data does not always come in its final processed form that is required for training machine learning algorithms. 
We use transforms to perform some manipulation of the data and make it suitable for training. 
The MNIST features are in PIL Image format, and for training, we need the features as normalized tensors. 
To make these transformations, we use combined ToTensor and Normalize.
  
The Dataset retrieves the dataset's features and labels one sample at a time. 
While training a model, we typically want to pass samples in "minibatches," reshuffle the data at every epoch to reduce model overfitting, 
and use Python's multiprocessing to speed up data retrieval. DataLoader is an iterable that abstracts this complexity for us in an easy API.
 

## MLP Model
The torch.nn namespace provides all the building blocks we need to build our own neural network. Every module in PyTorch subclasses the nn.Module. 
A neural network is a module itself that consists of other modules (layers). 
This nested structure allows for building and managing complex architectures easily. 
We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__. 
Every nn.Module subclass implements the operations on input data in the forward method.
 
We initialized the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (the minibatch dimension (at dim=0) is maintained). 
This is our input layer. Then we use linear layers with nonlinear activations to go to the next layers. 
The linear layer is a module that applies a linear transformation on the input using its stored weights and biases. 
Nonlinear activations are what create the complex mappings between the model's inputs and outputs. 
They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.
  
nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined. 
In this model, we use nn.ReLU between our linear layers. The last linear layer of the neural network returns x - raw values 
in [-infty, infty] - which are passed to the nn.Softmax module. 
The x are scaled to values [0, 1], representing the model's predicted probabilities for each class. 
dim parameter indicates the dimension along which the values must sum to 1.
  
The layers inside our neural network are parameterized, i.e. have associated weights and biases that are optimized during training. 
Subclassing nn.Module automatically tracks all fields defined inside our model object, and we do not need to explicitly manipulate them.

## Training Loop
To use the model, we pass it the input data(x and y). This executes the model's forward function. 
Calling the model on the input returns a 10-dimensional tensor with the predicted probability values for each class. 
  
When presented with some training data, our untrained network is likely not to give the correct answers. 
The loss function measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training. 
To calculate the loss, we make a prediction using the inputs of our given data sample and compare it against the true data label value.
 
Inside the training loop, optimization happens in three steps:  
Call optimizer.zero_grad() to reset the gradients of model parameters. 
Gradients, by default, add up; to prevent double-counting, we explicitly zero them at each iteration.
  
Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
  
Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
  
For training our neural network, we used back propagation. 
In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter. 
To compute those gradients, PyTorch has a built-in differentiation engine that supports automatic computation of gradient for any neural network model.
  
The functions that we applied to tensors to construct our neural network model is in fact an object of class Function. 
This object knows how to compute the function in the forward direction, and also how to compute its derivative during the backward propagation step. 
A reference to the backward propagation function is stored in grad_fn property of a tensor.
  
To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function with respect to parameters. 
To compute those derivatives, we call loss.backward().
 
Conceptually, autograd keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) 
in a directed acyclic graph (DAG) consisting of Function objects. 
In this DAG, leaves are the input tensors, roots are the output tensors. 
By tracing this graph from roots to leaves, it can automatically compute the gradients using the chain rule.
  
In a forward pass, autograd does two things simultaneously:
•	run the requested operation to compute a resulting tensor
•	maintain the operation's gradient function in the DAG.
  
The backward pass kicks off when .backward() is called on the DAG root. autograd then:
•	computes the gradients from each .grad_fn,
•	accumulates them in the respective tensor's .grad attribute
•	using the chain rule, propagates all the way to the leaf tensors.

## Testing Loop
The testing loop is very similar to the training loop, but it does not have the back propagation or optimization. 
By adding model.eval() to the first line of the function, PyTorch knows that there is no need to keep track of many variables.
 
By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation. 
However, there are some cases when we do not need to do that, for example, when we have trained the model 
and just want to apply it to some input data, i.e. we only want to do forward computations through the network. 
We can stop tracking computations by surrounding our computation code with torch.no_grad() block.
  
This action speeds up computations when we are only doing forward passes because computations on tensors that do not track gradients would be more efficient.


## Hyperparameters
Hyperparameters are adjustable parameters that let us control the model optimization process. 
Different hyperparameter values can impact model training and convergence rates.
 
We define the following hyperparameters for training and testing:
Learning Rate: how much to update model parameters at each batch/epoch. 
Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training. 
  
Momentum: It is the factor for the moving average of our gradients in SGD with the momentum method. 
It helps accelerate gradients vectors in the right directions, thus leading to faster converging.
  
Gamma: Decays the learning rate of each parameter group by gamma every step size epochs.
  
Batch_size: the number of training data samples propagated through the network before updating the parameters.
  
Test_batch_size: the number of testing data samples propagated through the network. 
If it is too small, the testing process will be slow, and if it is too high, it will need a lot of memory and may not work on some computers.
  
Epochs: the number of times to iterate over the dataset.
  
Device: We want to be able to train our model on a hardware accelerator like the GPU, if it is available. 
We check to see if torch.cuda is available; otherwise, we continue using the CPU.

## Model Initialization
We first create an instance of MLP (our model), and move it to the device. 
Then we choose nn.NLLLoss (Negative Log Likelihood) as our loss function for the classification goal.
  
All optimization logic is encapsulated in the optimizer object. Here, we use the SGD optimizer. 
We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate, and momentum hyperparameters. 
A scheduler is also defined to decay the learning rate. Each time scheduler.step() is executed, the current scheduler will compute a new learning rate.

MLP Model Evaluation
In the previous sections, we defined train_loop which loops over our optimization code and test_loop which evaluates the model's performance against our test data. 
Here, we pass the initialized loss function and optimizer to train_loop and test_loop.
  
Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each iteration of the optimization loop is one epoch.
 
Each epoch consists of two main parts:
•	The Train Loop: iterate over the training dataset and try to converge to optimal parameters.
•	The Test Loop: iterate over the test dataset to check if model performance is improving.
  
After running the optimization loop for 30 epochs, the results are very promising, as we reached 98.1% of accuracy for our test data.
  
![image](https://user-images.githubusercontent.com/83058686/217932326-2c5824b2-ae42-4f4f-ba4f-64dda45fd68e.png)
  
The model can classify all digits very well. Based on the bar chart below, the lowest classification accuracy is for number 9, but its difference with other numbers does not seem significant.
 
![image](https://user-images.githubusercontent.com/83058686/217932497-c62b2590-181b-44c7-bc1d-0d102ee34fad.png)
 

## CNN Model
The LeNet-5 convolutional neural network was taken as the baseline standard model. 
The LeNet network is broadly considered as the first true convolutional neural network. 
It is capable of classifying small single-channel (black and white) images with promising results.
  
LeNet is a simple model consisting of a convolutional layer with a max-pooling layer twice followed by two fully connected layers with a softmax output of ten classes at the end.
  
LeNet initially was proposed for the classification of letters as 32x32 images. As the MNIST images are 28x28 at the first layer above, 
we padded zero on both sides of the image matrix to make 32x32 images out of the MNIST dataset. 
We improved LeNet by introducing SGD with momentum and also considering gamma decays.
  
The results are considerably improved compared to the MLP model. 
The best result in the literature with much more complex models is 99.8%. This means our model is doing great!!!
   
![image](https://user-images.githubusercontent.com/83058686/217927844-c1220d18-6398-40de-875d-16bade88dff8.png)
  
  
![image](https://user-images.githubusercontent.com/83058686/217927916-74e80486-4dff-4714-8cee-9c0f7ffa2338.png)
 
Again, the algorithm is doing great on all digits, but number 9 has the least accuracy among all other numbers. 


# COIL100
We used the same MLP model as the MNIST case to classify COIL100 objects.
  
Unfortunately, COIL100 does not exist in PyTorch domain libraries. So, we should create a custom Dataset for our downloaded files.
 
A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
  
The __init__ function is run once when instantiating the Dataset object. 
We initialize the annotations list, the directory containing the images, and both transforms.
  
The __len__ function returns the number of samples in our dataset.
  
The __getitem__ function loads and returns a sample from the dataset at the given index idx. 
Based on the index, it identifies the image's location on disk, converts that to a tensor using read_image, 
retrieves the corresponding label from the labels_list, calls the transform functions on them (if applicable), 
and returns the tensor image and corresponding label in a tuple.
  
Based on the new custom Dataset, the previous load_data function should be revised as well. 
The main difference between MNIST and Coil100 is that COIL100 does not have independent testing and training datasets, 
so we should split the dataset randomly to test instances and train instances at the start of each epoch.
  
The implementation results are very good, as the model can correctly classify 99.5% of instances.
![image](https://user-images.githubusercontent.com/83058686/217933785-c22afc30-88c0-41cb-918d-1c3178f032f5.png)
  
The bar chart below shows that the model can classify almost all the images of objects from different angles, except for three objects.
![image](https://user-images.githubusercontent.com/83058686/217934096-948fc9fd-5332-4205-b3a0-13f78d07bbc9.png)

