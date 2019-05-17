import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import argparse
import seaborn as sns
import json
from collections import OrderedDict

def process_data(data_dir):
    
    '''
    Arguments : The datas' path
    
    Returns :   The training, validation, and testing image datasets 
                and 
                the loaders for the train, validation and test datasets.
                
                This function receives the location of the image files, applies the necessery 
                transformations (rotations,flips,normalizations and crops) and converts the images 
                to tensor in order to be able to be fed into the neural network
    '''
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # Define your transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                  transforms.RandomRotation(30),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms_valid = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(data_dir + '/train', transform = data_transforms_train)
    image_datasets_valid = datasets.ImageFolder(data_dir + '/valid', transform = data_transforms_valid)
    image_datasets_test = datasets.ImageFolder(data_dir + '/test', transform = data_transforms_test)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train =  torch.utils.data.DataLoader(image_datasets_train , batch_size = 32, shuffle = True)
    dataloaders_valid =  torch.utils.data.DataLoader(image_datasets_valid , batch_size = 32, shuffle = True)
    dataloaders_test =  torch.utils.data.DataLoader(image_datasets_test , batch_size = 32, shuffle = True)
    
    return dataloaders_train, dataloaders_valid, dataloaders_test, image_datasets_train, image_datasets_valid, image_datasets_test

    

# Load pretrained_network
def pretrained_model(arch):
    
    '''
    Arguments: The architecture for the network (alexnet,squeezenet1_0,vgg16)
    
    Returns:   The pre-trained model
    '''
    
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        print('Using vgg16')
    elif arch == "resnet18":
        model = models.squeezenet1_0(pretrained=True)
        print('Using squeezenet1_0')
    elif arch == "alexnet":
         model = models.alexnet(pretrained=True)
         print("Using alexnet")
       
    return model
        
def classifier(model, hidden_units):
    
    '''
    Arguments: The pre-trained model and the number of hidden unit nodes
    
    Returns:   Classifier
               A new classifier is contsructed. The pre-trained classifier is replaced by the new classifier 
    '''
    
    if hidden_units == None:
        hidden_units = 512
    input = model.classifier[0].in_features
        
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p = 0.2)),
                                            ('fc2', nn.Linear(hidden_units, 102)),

                                            ('output', nn.LogSoftmax(dim = 1))]))
    
    model.classifier = classifier
        
    return model.classifier

'''def device():
    use_gpu = torch.cuda.is_available()
    if args.gpu:
            if use_gpu:
                model = model.cuda()
                print ("Using GPU: "+ str(use_gpu))
            else:
                print("Using CPU because GPU is not available")'''


def train_model(epochs, dataloaders_train, dataloaders_valid, device, model, optimizer, criterion):
    
    '''
     Arguments: The model, the criterion, the optimizer, the number of epochs, train and vaidations datasets, 
                and device (whether to use a gpu or not)
    
    Returns: Trained model
    
    This function trains the model over a certain number of epochs and displays the training,validation
    and accuracy every "print_every" step using cuda if specified. The training method is specified by the 
    criterion and the optimizer which are NLLLoss and Adam respectively
    '''
    
    epochs = 10
    running_loss = 0
    print_every = 100
    steps = 0

    train_losses, valid_losses = [], []

    for epoch in range(epochs):   
        for images, labels in dataloaders_train:
            steps += 1
            model.to(device)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
                        
            logps = model.forward(images)
            train_loss = criterion(logps, labels)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                # turning the gradient off for the validation stage for faster computation
                with torch.no_grad():
                    for images, labels in dataloaders_valid:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        train_losses.append(running_loss/len(dataloaders_train))
                        valid_losses.append(valid_loss/len(dataloaders_valid))  


                print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/len(dataloaders_train):.3f}.. "
                          f"Valid loss: {valid_loss/len(dataloaders_valid):.3f}.. "
                          f"Accuracy: {accuracy/len(dataloaders_valid):.3f}")
                running_loss = 0
                model.train()       
                
    return model


# testing network
def test_network(model, dataloaders_test, device, criterion):
    
    '''
    Arguments: The model, testing dataloaders, device and loss criterion
    
    Returns: Nothing
    
             This function checks the performance of the test data on the trained model
    '''
    
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders_test:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            test_loss += criterion(logps, labels)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        print(f"Test accuracy: {accuracy/len(dataloaders_test):.3f}")
    


# saving model checkpoint
def save_checkpoint(model, image_datasets_train, checkpoint, arch, epochs):
    
    '''
    Arguments: The model, model architecture, training image dataset, checkpoint (saving path),
               and the hyperparameters (epochs and learning rate) of the network.
               
    Returns:   Saved model in the checkpoint directory.
    
               This function saves the model at a specified by the user path
    '''
    
    # mapping classes to indices
    model.class_to_idx = image_datasets_train.class_to_idx
    
    checkpoint = {'arch': arch,
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'class_to_idx' : model.class_to_idx,
                'epochs' : epochs, 
                'optimizer_state_dict' : optimizer.state_dict(), 
                'lr' : 0.001}
    
    return torch.save(checkpoint, args.checkpoint)




if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='training image classifier for flowers')

    paser.add_argument('data_dir', type=str, default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    paser.add_argument('--arch', type=str, default='vgg16', help='type of model architecture to be used')
    paser.add_argument('--hidden_units', type=int, default=512, help='hidden units for classifier/Network layer')
    paser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='save trained model to a file')
    
    args = paser.parse_args()
   
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device);
    
    is_gpu = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device set to {device}")

    else:
        device = torch.device("cpu")
        print(f"Device set to {device}")

        
    
    dataloaders_train, dataloaders_valid, dataloaders_test, image_datasets_train, image_datasets_valid, image_datasets_test = process_data(args.data_dir)
    
    model = pretrained_model(args.arch)
    
    for param in model.parameters():
        param.requires_grad = False    
    
    model.classifier = classifier(model, args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
    
    model = train_model(args.epochs, dataloaders_train, dataloaders_valid, device, model, optimizer, criterion)
    
    test_network(model, dataloaders_test, device, criterion)
    
    save_checkpoint(model, image_datasets_train, args.checkpoint, args.arch, args.epochs)
    
    print('Done')
