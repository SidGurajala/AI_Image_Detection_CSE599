import torch
import torchvision
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.transforms import v2
import torch.nn.functional as F
import wandb as wandb
import numpy as np

# Borrowed with modifications from CSE 599G Spring 2023
def check_accuracy_part34(loader, model, device, dtype=torch.float32, num_samples = 10000):
 #   print("Starting check_accuracy_part34")
    num_correct = 0
    model.eval()  # set model to evaluation mode
    criterion = nn.BCEWithLogitsLoss()
    s_fn = nn.Sigmoid()
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
          #  print("Iteration of internal loop in accuracy checker")
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
          #  print("Finished converting x and y to right device. Calculating scores")
            score = model(x)
            loss = criterion(score, y.reshape([-1,1]))
            num_correct += (torch.round(s_fn(score)) == y.reshape([-1,1])).float().sum()
          #  print("Finished calculating scores. ")
          #  print("Finished calculating scores & num_correct, num_samples")
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc, loss


# Borrowed with modifications from CSE 599G Spring 2023
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

# Borrowed with modifications from CSE 599G Spring 2023
def train_part34(model, optimizer, device, loader, val_loader=None, epochs=1, dtype=torch.float32, print_every=50):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    criterion = nn.BCEWithLogitsLoss()
    for e in range(epochs):
        print("epoch ", e)
        # unpacking train_loader is the bottleneck; saving it as an iterable
        # to memory before the loop doesn't help
        for t, (x, y) in enumerate(loader):

           # print("Putting model in training mode")
            model.train()  # put model to training mode
           # print("Converting X and y to correct devices and data types")
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
           # print("X, y conversion complete")

           # print("Generating scores")
            score = model(x)
            #print("Done calculating score. Calculating loss...")
            loss = criterion(score, y.unsqueeze(1).float())            
            #print("Calculated loss. Zeroing out gradients...")
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

           # print("Zeroed out gradients. Starting backward pass...")
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
           # print("Completed backward pass. Making optimizer step...")

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0 and val_loader != None:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                val_acc,val_loss = check_accuracy_part34(val_loader, model, device)
                wandb.log({"acc": val_acc, 
                           "train_loss": loss,
                           "val_loss": val_loss})

def main(data_dir_path = "/home/jupyter/",
         batch_size = 128, num_workers = 2, print_every = 50):
    
    print("Setting torch to run on GPU if available")

    USE_GPU = True
    dtype = torch.float32 

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Constant to control how frequently we print train loss.
    print_every = 50
    print('using device:', device)

    print("Setting up transforms to apply to image data...")
    transforms = v2.Compose([
        v2.ToImagePIL(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8), # optional, most input are already uint8 at this point
        v2.ToTensor(),
        v2.RandomApply(transforms=[v2.RandomResizedCrop(size=(32, 32), scale = (0.9,0.9),antialias = True),
                                   #v2.RandomRotation(degrees=(5,10)),
                                   v2.GaussianBlur(kernel_size=(5,5), sigma=1),
                                   v2.ColorJitter(brightness=0.5)  
                                   #v2.RandomPerspective(p = 1),  #default distortion is 0.5
                                   #v2.RandomAdjustSharpness(sharpness_factor = 2, p = 1)  #double the sharpness
                                  ], p=0.8)])
    
    
    print("Complete.")

    data_dir = data_dir_path

    print("Data directory is set to: ", data_dir)

    print("Setting up ImageFolder objects for train, val, test...")

    train_dataset = datasets.ImageFolder(root=data_dir+'/train/', transform=transforms)
    print("Train complete")
    val_dataset = datasets.ImageFolder(root=data_dir+'/val', transform=transforms)
    print("Val complete")
    test_dataset = datasets.ImageFolder(root=data_dir+'/test/', transform=transforms)
    print("Test complete")

    print("Now setting up loaders linked to those dataset objects:")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle = True)

    print("Train loader complete")

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle = True)

    print("Val loader complete")

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                             shuffle = True)

    print("Test loader complete")



    channel_1 = 32
    channel_2 = 32
    
    for rate in [2.5e-3, 1e-3, 8e-4]:
        for l2reg in [0.01, 0.0005, 0.0001]:
            
            print("Constructing model...")
            model = nn.Sequential(
            nn.Conv2d(3, channel_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Flatten(),
            # three layers of 64 rectified linear units per Bird, Lotfi (2023)
            nn.Linear(channel_2 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1))
            
            wandb.init(
            # set the wandb project where this run will be logged
            project="augmented_runs",
            # track hyperparameters and run metadata
            config={
            "learning_rate": rate,
            "l2reg": l2reg,
            "epochs": 5,
            "optimizier": "ADAM"}
            )
            optimizer = optim.Adam(model.parameters(),
                                   lr = rate,
                                   weight_decay = l2reg)
            train_part34(model, optimizer, device, train_loader, val_loader, epochs=5) 
            wandb.finish()
    
    from googleapiclient import discovery
    from oauth2client.client import GoogleCredentials

    credentials = GoogleCredentials.get_application_default()

    service = discovery.build('compute', 'v1', credentials=credentials)

    # Project ID for this request.
    project = 'cse599-proj-ai-img-detection'  # Project ID
    # The name of the zone for this request.
    zone = 'us-west4-a'  # Zone information

    # Name of the instance resource to stop.
    instance = 'deeplearning-1-vm'  # instance id

    request = service.instances().stop(project=project, zone=zone, instance=instance)
    response = request.execute()

if __name__ == "__main__":
    main()