# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    print(cat_to_name)
    
    
# TODO: Build and train your network

#load pre-trained network
model = models.vgg16(pretrained=True)
model
print('1')
print(model)
# Freeze parameters so we don't backprop
for param in model.parameters():
    param.requires_grad = False
    

#Feedforward network as a classifier, using ReLU activations and dropout
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

#train the classifier

def do_deep_learning(model, trainloader, epochs, print_every, criterion, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    print('2')

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        print('epochs')
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            print(inputs)
            print(labels)
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

#            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            print('3')
#            optimizer.step()


            running_loss += loss.item()
            print('running_loss ')
            print(running_loss)
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
                
                

    
do_deep_learning(model, trainloader, 3, 40, criterion,'cpu')

model.to('cpu')

def check_accuracy_on_test(testloader):    
    correct = 0
    total = 0
    print('3')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            print('3-1')
            outputs = model(images)
            print('3-2')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print('3-3')
            correct += (predicted == labels).sum().item()
            print('3-4')
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
check_accuracy_on_test(testloader)



# TODO: Do validation on the test set

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images.resize_(images.shape[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

epochs = 2
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    model.train()
    for images, labels in trainloader:
        steps += 1
        
        # Flatten images into a 784 long vector
       # images.resize_(images.size()[0], 784)
        
        #optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        #optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
            
# TODO: Save the checkpoint 

checkpoint = {'input_size': 784,
              'output_size': 10,
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model


def load_checkpoint(checkpoint):
    state_dict = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

print(model)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
# TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    image.resize(224)
    np_image = np.array(pil_image)
    
    print('np_image')
    print(np_image)
process_image() 

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
  def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
# TODO: Implement the code to predict the class from an image file

# TODO: Display an image along with the top 5 classes
