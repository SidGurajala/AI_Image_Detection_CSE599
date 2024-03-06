import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import v2
import wandb as wandb

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
class Conv_Autoencoder(nn.Module):
    def __init__(self, input_c, channel_1, channel_2, hidden_dim):
        super(Conv_Autoencoder, self).__init__()
        self.channel_2 = channel_2
        self.encoder = nn.Sequential(
                                    nn.Conv2d(3, channel_1, kernel_size=3, stride = 2, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(channel_1, channel_2, kernel_size=3, stride = 2, padding=1),
                                    nn.ReLU(),
                                    Flatten(),
                                    nn.Linear(channel_2*8*8, hidden_dim),
                                    nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(hidden_dim, channel_2*8*8))
        self.decoder = nn.Sequential(
                                    nn.Upsample(scale_factor = 2, mode = "bilinear"),
                                    nn.ConvTranspose2d(channel_2, channel_1, kernel_size = 3, stride=1, padding = 1),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor = 2, mode = "bilinear"),
                                    nn.ConvTranspose2d(channel_1, input_c, kernel_size = 3, stride=1, padding = 1),
                                    nn.Tanh())
    def forward(self, x): 
        hidden_rep = self.encoder(x)
        self.hidden_rep = hidden_rep
        rev_linear = self.linear(self.hidden_rep)
        rev_linear = rev_linear.reshape([rev_linear.shape[0], self.channel_2, 8, 8])
        reconstructed = self.decoder(rev_linear)
        return reconstructed
        
def check_val_loss(loader, model, device, dtype=torch.float32, num_samples = 10000):
    model.eval()  # set model to evaluation mode
    criterion = nn.MSELoss()
    with torch.no_grad():
        for t, (x, _) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
        return loss.item()

def train_autoencoder(model, optimizer, device, loader, val_loader=None, epochs=1, dtype=torch.float32, print_every=50):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    criterion = nn.MSELoss()
    for e in range(epochs):
        print("epoch ", e)
        # unpacking train_loader is the bottleneck; saving it as an iterable
        # to memory before the loop doesn't help
        for t, (x, _) in enumerate(loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU

            reconstructed = model.forward(x)
            loss = criterion(reconstructed, x)            
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0 and val_loader != None:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                val_loss = check_val_loss(val_loader, model, device)
                print('Validation Loss = ' + str(val_loss))
                wandb.log({"train_loss": loss,
                           "val_loss": val_loss})
                           
                           
def main(data_dir_path = "/home/jupyter",
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
    v2.ToDtype(torch.uint8),  # optional, most input are already uint8 at this point
    v2.ToTensor(),
    v2.RandomApply(transforms=[v2.RandomResizedCrop(size=(32, 32), scale = (0.9,0.9),antialias = True),
                                   #v2.RandomRotation(degrees=(5,10)),
                                   v2.GaussianBlur(kernel_size=(5,5), sigma=1),
                                   v2.ColorJitter(brightness=0.5)  
                                   #v2.RandomPerspective(p = 1),  #default distortion is 0.5
                                   #v2.RandomAdjustSharpness(sharpness_factor = 2, p = 1)  #double the sharpness
                                  ], p=0.8),
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize((0.5,),(0.5,))
    #v2.Resize((128,128)),
    #v2.RandomResizedCrop(size=(128, 128), scale = (0.08, 1.0), antialias=True),  # Or Resize(antialias=True)
    #v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
])
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
   
    rate = 0.001
    l2reg = 1e-4
    model = Conv_Autoencoder(3, 32, 64, 2048)
    wandb.init(
            # set the wandb project where this run will be logged
            project="Final_Encoder",
            # track hyperparameters and run metadata
            config={
            "learning_rate": rate,
            "l2reg": l2reg,
            "epochs": 5,
            "optimizier": "ADAM"})
    optimizer = optim.Adam(model.parameters(),
                           lr = rate,
                           weight_decay = l2reg)
    train_autoencoder(model, optimizer, device, train_loader, val_loader, epochs=5) 
    wandb.finish()
    torch.save(model.state_dict(), '/home/jupyter/final_encoder_model.pth')

if __name__ == "__main__":
    main()
