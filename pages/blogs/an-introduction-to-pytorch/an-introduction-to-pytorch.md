# An Introduction to PyTorch

PyTorch is one of the most popular libraries for creating neural network-based models in Python. It provides extensive support for different kinds of networks, from simple multilayer perceptrons to complex convolutional or language-processing networks. PyTorch provides developers with very fine control over model design and over the training process. This high level of customizability makes PyTorch extremely powerful, but it also necessitates that you learn how to work with every single component, a not-insignificant task. Additionally, PyTorch's sometimes confusing syntax can make this already difficult task even more confusing. This article is meant to help you, aspiring developer, get started making your own PyTorch models by teaching you the steps for creating and training models and by clarifying some of the more confusing syntax.

This article assumes you have a basic understanding of how neural networks work. If you do not, I highly recommend the first four episodes of 3Blue1Brown's excellent [Deep Learning series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=hsMObteFCP8kpL7T) on YouTube, as a starting point. This article will use the same MNIST dataset 3Blue1Brown uses, so it will feel like a natural follow-up. This article also assumes you have an intermediate-level understanding of programming in Python.

To be able to create your own models in PyTorch, there are five fundamental concepts you need to learn. This article will walk through each step by step. The first fundamental will be **Tensors**, the data structure underlying most of PyTorch's classes. The second is **Datasets and DataLoaders**, the classes PyTorch uses to work with data. The third fundamental is **Models**, PyTorch networks, and the concepts and steps involved in designing them. Fourth is the **Training Loop**, the standard procedure PyTorch expects you to implement to train the models. The fifth and final fundamental is the **Testing Loop**, where you can efficiently evaluate your trained model's performance. By the end of this article, you will have an understanding of all five fundamentals and will be able to design and implement your own models from scratch.

## Before We Begin: Install Pytorch

To get started with PyTorch, go to its [Get Started page](https://pytorch.org/get-started/locally/), and follow the instructions to install the package. Note the option to include CUDA with your installation. CUDA is a program created by NVIDIA to interface with their graphics processing units (GPUs), also called graphics cards. GPUs are specialized computer chips designed for real-time graphics processing. They are designed to process many calculations at once through a process called parallelization. While graphics rendering is what they were designed for, GPUs can also use parallelization to optimize other tasks, such as machine learning. You should not need to download CUDA to follow along with this article; however, as you create your own models, you will inevitably encounter performance bottlenecks. In such cases, if you have an NVIDIA GPU installed, using CUDA along with PyTorch will speed up training significantly.

In addition to PyTorch, you will also want to install Torchvision. This is a package that includes various tools used for computer vision tasks. For this analysis, it will provide us with the MNIST dataset, so we do not need to find and download it for ourselves. The other package you should consider installing is Jupyter, which provides a convenient development environment for Python. Jupyter is not required for this article, but it is extremely common in the world of data science. If you have not used it before, you should familiarize yourself with it. Once you've installed the packages, you can import them in your Python script or Jupyter environment as follows:

```
import torch
import torchvision
``` 

## PyTorch Fundamental 1: Tensors

### Working with tensors

In PyTorch, nearly everything is a tensor. The data you give your model will be a tensor. The model's parameters will be tensors. The model's output will be tenors. For this reason, it is important to understand the basics of tensors before you proceed. There is also one peculiar behavior of PyTorch tensors that will be essential to learn in order to make sense of PyTorch's more confusing bits of syntax.

The term *tensor* is not unique to PyTorch. Rather, it is a mathematical term for a grid of data with an arbitrary number of dimensions. You should be familiar with the idea of a matrix, a 2D grid of data values. Practically, tensors are a way to generalize the concept to higher dimensions. You can image a 3D tensor, for example, as a Rubik's cube, where a number is placed in every piece of the cube (including the center). In PyTorch, the `torch.Tensor` class is used to create tensor objects. Using the `torch.tensor()` function, you can turn a list of numbers into a tensor. Try the following code:

```
nums = [1, 2, 3, 4]
print(nums)
t = torch.tensor(nums)
print(t)
```

You can see that it outputs the following string representations for the vanilla Python list and for the PyTorch Tensor.

```
[1, 2, 3, 4]
tensor([1, 2, 3, 4])
```

A tensor object has an attribute called `shape` which will return its dimensions. This is often helpful for debugging your code. Consider the following example.

```
nums = [
	[1, 3, 5],
	[2, 4, 6]
]
t = torch.tensor(nums)
print(t.shape)
```

Output:

```
torch.Size([2, 3])
```

As you can see, the size of the tensor is (2, 3). The first dimension has a size of two, meaning there are two rows, and the second dimension has a size of three, meaning there are three columns. This is an example of a 2D tensor because there are two dimensions. If you look back at the list we passed in called `nums`, these dimensions look correct.

Some other options for creating tensors are the `torch.ones()`, `torch.zeros()`, and `torch.rand()` functions. For these functions, you pass *in* a tuple representing the dimensions of the tensor you want to make, and they will output a tensor of that shape with the values being all ones, all zeroes, or all random, respectively. 

```
shape = (3, 4)
all_ones = torch.ones(shape)
print(all_ones)
all_zeros = torch.zeros(shape)
print(all_zeros)
all_random = torch.rand(shape)
print(all_random)
```

Output:

```
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
tensor([[0.7665, 0.8024, 0.3839, 0.4064],
        [0.6788, 0.0092, 0.2260, 0.5531],
        [0.7062, 0.8888, 0.0275, 0.5510]])
```

### The computational graph

Earlier, it was mentioned that there is a peculiarity to the Tensor object that would be important to learn. We will now briefly explore what that is. In short, PyTorch has a way of tracking all of the changes that are made to a tensor. This is done to make backpropagation easier when a neural network is being trained. 

Whenever you perform an operation on a tensor (say, multiplying all values of a tensor by two or multiplying two tensors together), the resulting tensor can keep a reference to all the tensors and operations used to create it. Each tensor becomes a node in a computational graph, which contains the entire history of how that tensor was formed. Remember that when data is passed into a PyTorch model, it will be in the form of a tensor. That tensor will pass through the weights and biases of the model and be transformed eventually into an output tensor. The output tensor's computational graph will keep a record of how every weight and bias affected it. 

PyTorch can then perform backpropagation on the output tensor, simply by traversing the computational graph backwards. Tensors in a computational graph also have the ability to store a gradient value. When backpropagation is performed, every node has its gradient with respect to the output tensor recorded and stored. These gradients can then be used by an optimizing algorithm like gradient descent to update model weights.

Now, everything we have discussed here is an optional behavior for tensors. By default, tensors you create with `torch.tensor()` will not create computational graphs. Instead, this behavior must be enabled with the optional parameter `requires_grad=True`, but in reality, you shouldn't need to do this manually very often. PyTorch provides functions for quickly creating model layers, and it will handle enabling this feature automatically.

## PyTorch Fundamental 2: Datasets and DataLoaders

### The Dataset class

PyTorch supplies two classes to make working with data easier. The first is the `torch.utils.data.Dataset` class, which serves as a convenient wrapper for your datasets. You are expected to define a custom class for your dataset that extends this `Dataset` class. When you do so, you need to define two methods.

First, you must define a `__len__()` method, which will be called if you pass the dataset object into the `len()` function. It should return the number of entries in your dataset. Second, is the `__getitem__()` method, which is called whenever you index into the object with square bracket notation. The method should return a tuple with the first entry being a tensor containing the input data, and the second entry being a tensor containing the label. Consider the following example:

```
class SampleDataset(torch.utils.data.Dataset):
	
	def __init__(self):
		self.data = [
			[1, 2, 3],
			[6, 5, 4],
			[1, 5, 10],
			[12, 0, -2]
		]
		self.labels = [1, 0, 1, 0]
		
	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		return torch.tensor(self.data[index]), torch.tensor(self.labels[index])
		
	def __len__(self) -> int:
		return len(self.labels)
	
```

Here we define a `SampleDataset` class which extends `torch.utils.data.Dataset`. This will contain some dummy data to illustrate how you define a `Dataset` object. The `__init__()` method loads the data and stores it in the `self.data` and `self.labels` attributes. The `__getitem__()` method returns the entry at the given index as a tuple of tensors, and the `__len__()` method returns the number of entries in the dataset. Now, once the object is instantiated, you can use square brackets to retrieve specific entries, and you can use the `len()` method to determine the number of entries. You can also iterate through the dataset using a for loop.

```
data = SampleDataset()
print(data[0])
print(len(data))
print()
for entry, label in data:
	print(entry, label)
```

Output:

```
(tensor([1, 2, 3]), tensor(1))
4

tensor([1, 2, 3]) tensor(1)
tensor([6, 5, 4]) tensor(0)
tensor([ 1,  5, 10]) tensor(1)
tensor([12,  0, -2]) tensor(0)
```

In this example, the constructor was responsible for creating the data, but in the real world, the data will be loaded from a file. If the dataset is reasonably sized, this can also be done in the `__init__()`, where the entirety of the dataset can be loaded and stored in memory. If the dataset is prohibitively large, however, you may want to have  `__getitem__()`  be responsible for loading specific entries from file. This is called lazy-loading, and it can help increase performance if the size of the dataset is causing issues. 

### The DataLoader class

The next important class for data handling is the `torch.utils.data.DataLoader`. This class is designed to take a `Dataset` object and split the data into groups called batches. These batches can then be loaded one at a time during model training. To create a data loader, you simply pass in a Dataset to the `torch.utils.data.DataLoader()` constructor. Additionally, you should set the batch size with the `batch_size` parameter, and determine whether or not to shuffle the data points or to use them in the order the Dataset stores them in using the `shuffle` parameter.

Some things to note about the data loader class. The batch size you choose is mostly arbitrary, but it is common to use a power of two, like 128 or 256. If you are using CUDA, you should probably make the batch size as large as your graphics card can handle to maximize the performance gain. One of the main purposes of batching the data is that whole batches can be processed at once, if a GPU is present to parallelize the operation. 

Next, you should generally be shuffling your training data, but not your testing data. When you train your model, you will likely process the entire dataset multiple times--sometimes hundreds of times. Shuffling the training data will ensure the model sees a different order each time. With testing, however, the order that entries are processed has no effect, so there is no reason to waste processing time shuffling the data.

Dataloaders are meant to be iterated through, rather than indexed. That means that you can't use square brackets to find a particular batch; you must iterate through the values. For each batch, the tensors that make up the input data will be combined into a single tensor. See the following code as an example:

```
data = SampleDataset()
loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)
for inputs, labels in loader:
    print(inputs, labels)
    print()
```

Output:

```
tensor([[1, 2, 3],
        [6, 5, 4]]) tensor([1, 0])

tensor([[ 1,  5, 10],
        [12,  0, -2]]) tensor([1, 0])
```

Notice that when we iterated through the loader, it gave us batches of size two, as we specified in the constructor. Note that each batch was combined into a single tensor. Likewise, the labels for each batch were also combined into a single tensor.

When a DataLoader is passed into the `len()` function, it will return the number of batches, not the number of entries in the dataset. If you want the latter, you can use the `dataset` attribute to get a reference back to the original dataset object, and call `len()` on that.

```
print('Number of batches:', len(loader))
print('Number of entries:', len(loader.dataset))
```

Output:

```
Number of batches: 2
Number of entries: 4
```

The last thing to note about the `DataLoader` class is that the final batch may not be the same size as all others. If the number of batches does not evenly divide the number of entries in the dataset, there will not be enough entries to make the final batch the same size. This should not negatively affect training, but it is important to keep in mind.

```
lopsided_loader = torch.utils.data.DataLoader(data, batch_size=3, shuffle=False)
for inputs, labels in lopsided_loader:
    print(inputs, labels)
    print()
```

Output:

```
tensor([[ 1,  2,  3],
        [ 6,  5,  4],
        [ 1,  5, 10]]) tensor([1, 0, 1])

tensor([[12,  0, -2]]) tensor([0])
```


### Introducing MNIST

The dataset we will be working with in this article is the MNIST dataset. MNIST is a collection of images of handwritten single digits 0-9. The images are all grayscale, meaning that there is no color. Each pixel is only given a brightness value from 0 to 255, with 0 representing pure black and 255 representing pure white. These images are all 28 by 28 pixels in size. 

![An image of a hand drawn five](five.jpg)

MNIST is one of the datasets provided by torchvision, so rather than downloading the data and creating the `Dataset` object ourselves, we can use the one it supplies. The data will still need to be downloaded--it doesn't get installed alongside torchvision by default--but torchvision will automate the process. 

We will use `torchvision.datasets.MNIST()` to instantiate the dataset object. There are a few parameters we need to specify. First, the `root` parameter, which is the name of the directory the MNIST data will be stored in. For this example, we will store it in a folder called 'data'. The next parameter is the `train`	parameter, which determines which part of the MNIST dataset to load, the training data or the testing data. We will use both, creating a `training_data` object for the training data with `train=True`, and a `testing_data` object using `train=False`. Next, we need to set `download=True` so that it downloads the dataset for us. If the dataset has already been downloaded, setting `download=True` will simply be ignored. It will not throw an error or re-download the data. 

Torchvision's MNIST dataset object returns images as PIL objects, but the model will require the input data to be in tensor form. In order to accomplish this, we will add one last parameter to the `torchvision.datasets.MNIST()` call. Specifically, we need to set `transform=torchvision.transforms.ToTensor()`. This will automatically convert each image into a tensor where every pixel gets a value from 0 to 1, 0 representing pure black and 1 representing pure white.

```
training_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())
testing_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())
```

## PyTorch Fundamental 3: Models

Now, it is time to learn how to design a neural network model. First, you will create a new class to represent your model that extends the base class `torch.nn.Module`. This is the superclass for all neural networks and neural network layers in PyTorch. In the `__init__()` function of your new class, define the layers of your network using PyTorch's various layer options. Take a look at this example:

```
class MnistDigitClassifier(torch.nn.Module):

	def __init__(self):
	
		# Required!
		super().__init__()
		
		self.flat = torch.nn.Flatten()
		
		self.fc1 = torch.nn.Linear(28 * 28, 16)
		self.fc2 = torch.nn.Linear(16, 16)
		self.out = torch.nn.Linear(16, 10)
		
		self.act = torch.nn.ReLU()
		
	def forward(self, x: torch.Tensor) -> torch.Tensor:
	
		x = self.flat(x)
	
		x = self.fc1(x)
		x = self.act(x)
		
		x = self.fc2(x)
		x = self.act(x)
		
		return self.out(x)
```

First, notice the call to `super().__init__()` at the very beginning of the constructor. This is required for PyTorch to set your model up correctly when it is created. Next, notice that we are creating layers using different modules from `torch.nn` and assigning them as model attributes. The `torch.nn` submodule contains many different options for model layers. You can find a list of options on the [torch.nn](https://docs.pytorch.org/docs/stable/nn.html) page of the PyTorch documentation. 

In this example, the first layer is created with `torch.nn.Flatten()` and assigned to `self.flat`. This layer takes a multi-dimensional tensor and flattens it into a one-dimensional tensor. This is necessary, since our data objects store the images as two-dimensional tensors, but the next `Linear` layers expect one-dimensional tensors.

The Linear() function constructs a linear layer. This is the standard fully-connected layers of a multilayer perceptron you should be familiar with. The first layer takes in an input of 28 x 28 values and outputs 16 values according to its weights and biases. The next layer is also linear, and maps the 16 values output by the previous layer to 16 new values. The last linear layer is the output layer, which takes the sixteen values and maps them to 10 values representing each of the ten possible digit classifications. We also define a ReLU layer to serve as an activation function after the first two linear layers. 

The `__init__()` function is where we define what layers the model will have, but the `forward()` function is where we define the order of these layers. The forward method takes in a tensor of data, then passes it through the layers of the model, and outputs the resulting tensor. Notice that we use the same `self.relu` layer after both hidden layers. Activation layers, and indeed any layer that does not have parameters to learn during training, can be used multiple times. 

The `forward()` method reveals another one of PyTorch's peculiarities. Take a close look at how `self.fc1` is defined and then used. We create the Linear layer with `torch.nn.Linear(28 * 28, 16)` and save it to the `fc1` attribute. It appears that we are creating an object from the `Linear` class. But, in `forward()` we call `fc1` directly as if it were a function. So, which is it? Is it an object or is it a function? 

The answer is actually quite straightforward: it is an object. But you may not be aware that Python allows you to call objects as if they were functions. You have to define the behavior manually in order to enable it. Inside the class, define a method called `__call__()`, and it will be run whenever the object is called as a function. So, this is what is happening with the layers in our network. Every module in `torch.nn` has its `__call__()` method defined. 

It is important to note that all of these layers are subclasses of `torch.nn.Module`, just like our `MnistDigitClassifier`. If you examine the source code for these layers, they all have their own `forward()` methods, which get called in their `__call__()` methods. This may lead you to wonder, can our model be called like a function? The answer is, in fact, yes. As you will see in the next section, whenever you want to pass data into your model, you simply call the model as if it were a function rather than calling `forward()` manually. To test this, you can instantiate the model, then pass in the first entry of the MNIST training dataset we defined earlier.

```
pixels, label = training_data[0]
model = MnistDigitClassifier()
output = model(pixels)
print(output)
```

Output:

```
tensor([[ 0.2073,  0.0804, -0.0831,  0.2279, -0.0973, -0.1038, -0.2498, -0.0426,
          0.1610, -0.0976]], grad_fn=<AddmmBackward0>)
```

With this, our model is defined and ready to be trained.

## Fundamental 4: The Training Loop

In order to train the model, we will need to select a loss function and an optimizer. For a classification problem, cross-entropy is generally the loss function of choice, and PyTorch provides this in the class `torch.nn.CrossEntropyLoss`. We will store this in a variable called "criterion" as the name "loss" will be reserved for use later. The optimizer is the algorithm that is used to update the weights and biases of the model during training. Traditionally, this algorithm would be Stochastic Gradient Descent, and it is available in PyTorch as `torch.optim.SGD`, but there are many other optimizers we can use as well. In this analysis, we will use Adam, one of the most common first choices for optimizers in modern machine learning. This optimizer is available as `torch.optim.Adam`. 

When we instantiate the optimizer, we need to pass in the model's parameters to link it to the model. These can be accessed by calling the `parameters()` method on our model, which is inherited from `torch.nn.Module`.

```
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

Also, we need to create a data loader for the training Dataset we defined earlier.

```
training_loader = torch.utils.data.DataLoader(training_data, batch_size=256, shuffle=True)
```

Lastly, we need to set up a device object to represent the GPU or CPU, depending on which we are using to train. You do this using the `torch.device()` function, passing in the string 'cuda' or 'cpu', respectively. Using the function  `torch.cuda.is_available()` to check if CUDA has been successfully installed, you can create the following line of code that will create a CUDA device if it is available, and a CPU device if not. 

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Now, we can set up the training loop. We will define a function that takes in the model, the training data loader, the loss criterion, the optimizer, the device, and the number of epochs to train for. Inside, we first set the model to training mode by calling its `train()` method. Certain layers behave difficult in training versus in testing, so this needs to be set explicitly.

Next, we will iterate over the number of epochs, and inside each epoch, iterate over every batch in the training data loader.

```

def train(model: torch.nn.Module, 
	loader: torch.utils.data.DataLoader,
	criterion: torch.nn.modules.loss._Loss,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	num_epochs: int) -> None:
	
	model.train()
	
	for _ in range(num_epochs):
	
		for inputs, labels in loader:
		
			#...
		
```

Now, inside the loop, we perform the following six steps. First, we zero out the gradients of the model, clearing away all of the information from the previous batch. This can be done with `optimizer.zero_grad()` or `model.zero_grad()`. We will use the former in this analysis. Remember that we linked the optimizer to the model when we passed in its parameters, so it has access to them.

```
			optimizer.zero_grad()		
```

Next, we need to take the input data and pass it to the device, either the GPU or the CPU. Tensors have a `to()` method that will do this, but we must assign the result back to the original variables.

```
			inputs, labels = inputs.to(device), labels.to(device)
```

Next, we take the batch of data and give it to the model. Remember that the DataLoader turns the entire batch of data into a single tensor of inputs and a single tensor of labels. You may worry that the model can only accept one input at a time, but this is not true. You can input the whole batch at once, and the model will output a tensor of predictions for the entire batch.

```
			outputs = model(inputs)
```

Now that we have the model's outputs for this batch, we need to calculate the error. Pass the model output and the true labels into the criterion function to get the loss.

```
			loss = criterion(outputs, labels)
```

Now, here is where the magic happens. The loss tensor has access to the computational graph that includes all of the weights and biases involved in calculating it. If we call the `backward()` method on the loss, it will perform backpropagation, traverse the graph backwards, calculate--and record--the gradients for each tensor.

```
			loss.backward()
```

Lastly, now that the gradients have been calculated, you call the `step()` method on the optimizer to apply the changes according to whatever optimization algorithm you selected. 

```
			optimizer.step()
```

And that concludes one single training example. These six function calls, in this exact order, will appear in essentially every single training function you create in PyTorch. They are the heart of the training loop.

Here is the completed training function altogether:

```
def train(model: torch.nn.Module, 
	loader: torch.utils.data.DataLoader,
	criterion: torch.nn.modules.loss._Loss,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	num_epochs: int) -> None:
	
	model.train()
	
	for _ in range(num_epochs):
	
		for inputs, labels in loader:
		
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
```

Now we simply call the function to train our model. We will only use one epoch for this example so that the training does not take too long.

```
train(model, training_data, criterion, optimizer, 1)
```

## Fundamental 5: The Testing Loop

Once you have a trained model, you will want to test it to see how well it performs. The testing process is similar to the training process in a few ways. There is a **testing loop**, which iterates over the testing data in batches, just like in the training loop. But there are still a number of unique considerations to explore.

Firstly, we want to make sure PyTorch knows that it does not need to track changes to model parameters during training. We won't be calling `loss.backward()` or `optimizer.step()`, so it is technically not *required* to do so, but it will prevent processing power from being wasted. To accomplish this, we put the entire testing loop into the `torch.no_grad()` context using the `with` keyword. All lines of code within that context will not have changes tracked. Also, we should switch the model into evaluation mode using `model.eval()`.

```
def test(model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	criterion: torch.nn.modules.loss._Loss,
	device: torch.device) -> None:
	
	model.eval()
	
	# Set up model metrics
	# ...
	
	with torch.no_grad():
	
		for inputs, labels in loader:
		
			inputs, labels = inputs.to(device), labels.to(device)
		
			# ...
```

Next, we need to decide what metrics we want to test for. For a regression problem, you would probably want to use the root mean squared error or the mean absolute error, alongside the loss. For a classification problem, you may want to use accuracy, F1 score, and a confusion matrix in addition to the loss. PyTorch does not include functionality for different evaluation metrics by default, so you will need to implement them by hand or consider using libraries like PyTorch Lightning, which extend base PyTorch functionality. For this introduction, we will only test for two metrics: accuracy and loss.

Remember that the model outputs a tensor of ten values corresponding to each of the ten digits, and the one with the largest value represents the model's prediction. To calculate the accuracy, we will need to count how many entries get correctly classified in this manner. To get the prediction out of a model, we will use the torch.argmax() function to return the index with the maximum value; this tells us which digit the model thinks is the most likely classification. Remember that the model processes a whole batch of data at once, so instead of getting one 1 x 10 tensor for each data entry, we will get an n x 10 tensor representing predictions for the entire batch. We don't want the maximum index across the batch; we want the maximum for each individual data entry. This can be accomplished by setting `dim=1` when we call argmax, which tells the function to ignore the 0th dimension and select the maximum for each individual example. We will store these predictions in a variable called `preds`.

```
			outputs = model(inputs)
			preds = torch.argmax(outputs, dim=1)
```

Once we have the predictions, we can count the number of them that are correct by setting them equal to the labels. The resulting tensor will be True for all of the examples that are classified correctly and False for those that are not. The `sum()` method can then be called to count the number of True values, and `item()` can be called to turn the tensor into a vanilla Python integer. We can then add this to a running count of all the correctly labelled data points called `num_correct`. Note that this variable needs to be defined before the testing loop.

```
			num_correct += (preds == labels).sum().item()
```

When the loop is complete, we will divide the number of correctly classified testing samples by the total number of testing samples to calculate accuracy.

Tracking the loss is more straightforward, but there is one slight annoyance. When we call the loss criterion function, we get back the average loss for the entire batch. This can be turned into a vanilla Python integer using `item()`. However, we don't want the loss for each batch; we want the average loss across the entire testing set. We could average every batch's loss together, but remember that the final batch may be smaller than the others, so the batches cannot all be treated equally. 

To solve this, we need to multiply each batch's average by the number of entries in that batch. This will turn it from an average loss to a total loss for the batch. We can get the number of entries in each batch by taking the length of the labels tensors.

```
			running_loss += loss.item() * len(labels)
```

Then, at the end of the process, we divide the running loss by the number of entries in the testing dataset. Remember that the `len()` function of the data loader class gives the number of batches, not the number of entries in the data set. Instead, we must call `len()` on `loader.dataset`.

Here is the entire testing loop put together, with the average loss and the accuracy returned as a tuple.

```
def test(model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	criterion: torch.nn.modules.loss._Loss) -> tuple[float, float]:
	
	model.eval()
	
	num_correct = 0
	running_loss = 0.0
	
	with torch.no_grad():
	
		for inputs, labels in loader:
		
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			running_loss += loss.item() * len(labels)
			
			preds = torch.argmax(outputs, dim=1)
			num_correct += (preds == labels).sum().item()
			
	average_loss = running_loss / len(loader.dataset)
	accuracy = num_correct / len(loader.dataset)
	
	return average_loss, accuracy
```

The following code snippet will call `test()` and output the metrics in a nice format.

```
loss, accuracy = test(model, testing_loader, criterion)
print(f'Model achieved {(100 * accuracy):.2f}% accuracy [cross-entropy loss: {loss:.4f}]')
```

Output:

```
Model achieved 94.72% accuracy [cross-entropy loss: 0.1785]
```

---

My model achieved an accuracy of about 95% after a single training epoch. There is randomness involved in the initialization of the neural network; your results should be different, but in the same ballpark. There are ways of ensuring reproducibility within PyTorch, but that is an article for another day.

---

By now, you should be familiar with the fundamentals of PyTorch and ready to create some models of your own, but there is still much to learn. Your next steps will likely include trying out different models and model architectures, tracking performance metrics during training, visualizing performance metrics, hyperparameter tuning, and much more. PyTorch has a variety of [tutorials](https://docs.pytorch.org/tutorials/) for you to explore when you are ready to move on.

There will also be more to come on PyTorch on this blog. Very soon, an article on model validations and hyperparameter tuning will be up, so make sure to check back in occasionally to see what's new. Good luck in your machine learning journey. Onwards and upwards!