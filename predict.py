import torch
import argparse
import PIL
from torchvision import models
from torch import optim, nn
import numpy as np
import json
from collections import OrderedDict

# arg_parse() parses keyword arguments from the command line
def arg_parse():
    # defining a parser object - paser
    paser = argparse.ArgumentParser(description='training image classifier')

    # adding arguments to the paser(parser object)
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--image_path', type=str, default='flowers/test/1/image_06743.jpg', help='stored image path')
    # checkpoint = saving directory
    paser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='save trained model to a file')
    paser.add_argument('--top_k', type=int, default= 5, help='display top class probabilities')
    paser.add_argument('--cat_to_name_json', type=str, default='cat_to_name.json', help='mapper path from category to name')
    
    # parse args (args is the result of paser after adding arguments)
    args = paser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading model checkpoint
def load_checkpoint(checkpoint_pth):
    checkpoint = torch.load(checkpoint_pth)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Getting the model architecture
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.squeezenet1_0(pretrained=True)
    else:
        print("Model arch not found.")
    
    # Freezing the only features parameter of the model 
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 512)),
                                                     ('relu', nn.ReLU()),
                                                    ('drop', nn.Dropout(p = 0.2)),
                                                    ('fc2', nn.Linear(512, 102)),

                                                    ('output', nn.LogSoftmax(dim = 1))]))
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

    model.to(device);
        
    #model=checkpoint['model']    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs=checkpoint['epochs']
    learning_rate = checkpoint['lr']
        
    #model.eval()
    return model


# TODO: Process a PIL image for use in a PyTorch model (without using tran)
def process_image(image_path):
    image = PIL.Image.open(image_path)
    
    #RESIZING
    
    print(image)
    
    size = 224
    width = image.size[0] # 500
    height = image.size[1] # 601
    
    print(width)
    print(height)
    
    if width < height:
        width = int(size) # =224
        height = int(max(height * size / width, 1)) # =269
        
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
        
    resized_image = image.resize((width, height))
    
    print(resized_image)
    
    # CROPPING    
    x1 = (width - size) / 2  # 224 - 224 = 0
    y1 = (height - size) / 2 # 269 - 224 = 45
    x2 = x1 + size           # 224
    y2 = y1 + size          # 269
    cropped_image = image.crop((x1, y1, x2, y2))
    
    print(cropped_image)
    
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image_arr = (np_image - mean) / std
    np_image_arr = np_image.transpose((2, 0, 1))
    
    return np_image_arr

img_pth1 = 'flowers/test/1/image_06743.jpg'
img_pth2 = 'flowers/test/5/image_05159.jpg'


# predict function
def predict_image(image_path, model, cat_to_name, top_k):
    with torch.no_grad():
        model.eval()
        model.to(device) # put model on gpu
        image=process_image(image_path)
        image = torch.FloatTensor(image)
        image = image.unsqueeze(0)
        image = image.to(device) # put input image on GPU
        
        
        output = model(image)
        ps = torch.exp(output)
        #index = output.data.cpu().numpy().argmax()

        top_probs, top_idx = ps.topk(top_k, dim = 1)
        
         # convert from index to class        
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
      
        # converting top_probs, top_idx to list
        top_probs_arr = np.array(top_probs)[0] # top_probs_arr.detach().numpy().tolist()[0] 
        top_idx_arr = np.array(top_idx)[0]  # top_idx.data.numpy()[0].tolist() / top_idx.detach().numpy().tolist()[0]

        
        top_classes = [idx_to_class[i] for i in top_idx_arr]
        top_classes_names = [cat_to_name[str(index)] for index in top_classes]

        return idx_to_class,top_idx_arr, top_idx, top_probs_arr, top_classes, top_classes_names

    
    
def main():
    args=arg_parse()
    
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
                
    # mapping index to top_classes and top classes names
    with open(args.cat_to_name_json, 'r') as f:
            cat_to_name = json.load(f)
    
 
    model = load_checkpoint(args.checkpoint)
    image = process_image(args.image_path)
       
    idx_to_class,top_idx_arr, top_idx, top_probs_arr, top_classes, top_classes_names = predict_image(args.image_path, model, cat_to_name, args.top_k)
    #args, args.image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk
    print(top_probs_arr, top_classes_names)

if __name__ == '__main__':
    main()

