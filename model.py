import torch
import torch.nn as nn
from torch.autograd import Function

# Gradient reversal layer from the DANN paper
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)

        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """

        self.outputs = nn.Conv3d(64, 2, kernel_size=1, padding=0)
        #self.fc1 = nn.Linear(786432, 1) # input and output filters
        #self.age = nn.Conv3d(4, 1, kernel_size=1, padding=0)
        #self.age1 = nn.Conv3d(64, 1, kernel_size=1, padding=0)
        #self.glob_age = nn.Conv3d(1024, 1, kernel_size=1, padding=0)
        #self.flat = nn.Flatten()
        #self.linear = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Age """
        #age_output = self.age1(d4)
        #age_output = self.relu(age_output)

        """ Seg """
        seg_output = self.outputs(d4)
        seg_output = torch.softmax(seg_output, dim=1)

        """ global age """
        #glob_age_output = self.glob_age(b)
        #glob_age_output = self.flat(glob_age_output)
        ##glob_age_output = self.linear(glob_age_output)
        #glob_age_output = self.relu(glob_age_output)


        return seg_output#, age_output, glob_age_output
'''
class DAUnet(nn.Module):
    def __init__(self, out_channels = 2):
        super().__init__()

        # UNet encoder
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        # Bottleneck
        self.b = conv_block(512, 1024)

        # UNet decoder
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # Activation function
        self.relu = nn.ReLU()

        # Domain classifier
        self.fc1 = nn.Linear(90266624, 1)
        
        # Segmentation
        self.outputs = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)
       

    def forward(self, inputs, alpha):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Segmentation output
        seg_output = self.outputs(d4)
        seg_output = torch.softmax(seg_output, dim=1)

        # Classification output
        bottleneck_features = torch.flatten(b, 1)
        last_layer_features =  torch.flatten(d4,1)
        concatenated_features = torch.concat([bottleneck_features,last_layer_features],dim = 1) 
        reverse_features = ReverseLayerF.apply(concatenated_features, alpha)
        domain_prediction = self.fc1(reverse_features)

        return seg_output,domain_prediction
'''
class DAUnet(nn.Module):
    def __init__(self, out_channels=2):
        super().__init__()

        # UNet encoder
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        # Bottleneck
        self.b = conv_block(512, 1024)

        # UNet decoder
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # Activation function
        self.relu = nn.ReLU()
        
        # Domain classifier
        self.fc1 = nn.Linear(351232 , 70)  #65536-64 #351232-112  
        self.bn1 = nn.BatchNorm1d(70) #30-96  #20-112
        self.fc2 = nn.Linear(70, 1)
        self.rl = nn.ReLU(True)
        
        '''
        # Domain classifier
        self.fc1 = nn.Linear(90266624, 512)  # 56844288-96 #90266624-112  
        self.bn1 = nn.BatchNorm1d(512) # 30-96  # 20-112
        self.rl = nn.ReLU(True)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.r2 = nn.ReLU(True)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.r3 = nn.ReLU(True)
        self.fc4 = nn.Linear(128, 1)
        self.r4 = nn.ReLU(True)
        '''
        # Segmentation
        self.outputs = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs, alpha):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Segmentation output
        seg_output = self.outputs(d4)
        seg_output = torch.softmax(seg_output, dim=1)

        # Classification output
        bottleneck_features = torch.flatten(b, 1)
        last_layer_features = torch.flatten(d4, 1)
        #concatenated_features = torch.cat([bottleneck_features, last_layer_features], dim=1)
        reverse_features = ReverseLayerF.apply(bottleneck_features, alpha)
        
        domain_prediction = self.fc1(reverse_features)
        domain_prediction = self.bn1(domain_prediction)
        domain_prediction = self.rl(domain_prediction)
        domain_prediction = self.fc2(domain_prediction)
        

        '''
        domain_prediction = self.fc1(reverse_features)
        domain_prediction = self.bn1(domain_prediction)
        domain_prediction = self.rl(domain_prediction)
        domain_prediction = self.fc2(domain_prediction)
        domain_prediction = self.bn2(domain_prediction)
        domain_prediction = self.r2(domain_prediction)
        domain_prediction = self.fc3(domain_prediction)
        domain_prediction = self.bn3(domain_prediction)
        domain_prediction = self.r3(domain_prediction)
        domain_prediction = self.fc4(domain_prediction)
        domain_prediction = self.r4(domain_prediction)
        '''

        return seg_output, domain_prediction

