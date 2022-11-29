import torch


class Encoder(torch.nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=2),
            torch.nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = torch.nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), # 1024, 10
            torch.nn.ReLU(True),
            torch.nn.Linear(128, encoded_space_dim)
        )
    
    
             
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class Decoder(torch.nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = torch.nn.Sequential(
            torch.nn.Linear(encoded_space_dim, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 64 * 3 * 3),
            torch.nn.ReLU(True)
        )

        self.unflatten = torch.nn.Unflatten(dim=1, 
        unflattened_size=(64, 3, 3))

        self.decoder_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 3, 
            stride=2, output_padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 3, 5, stride=2, 
            padding=1, output_padding=1),
           
            # torch.nn.BatchNorm2d(8),
            # torch.nn.ReLU(True),
            # torch.nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
            # torch.nn.BatchNorm2d(4),
            # torch.nn.ReLU(True),
            # torch.nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),

            # torch.nn.ReLU(True),
            # torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
class Net(torch.nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim,fc2_input_dim)
        self.decoder = Decoder(encoded_space_dim,fc2_input_dim)

    def forward(self, x):
        # Encode data
        encoded_data = self.encoder(x)
        # Decode data
        decoded_data = self.decoder(encoded_data)
        return decoded_data

# add one more linear layer in the end
class NetClassification(torch.nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim, num_class):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim,fc2_input_dim)
        self.decoder = Decoder(encoded_space_dim,fc2_input_dim)
        self.header = torch.nn.Linear(3 * 32 * 32, num_class)

    def forward(self, x):
        # Encode data
        encoded_data = self.encoder(x)
        # Decode data
        decoded_data = self.decoder(encoded_data)
        x = torch.flatten(decoded_data, 1)
        x = self.header(x)
        return x