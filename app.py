import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import streamlit as st
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN Model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1) 
        #Fully Connected Layer
        self.fc1 = nn.Linear(32*14*14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2) #2x2 kernel and stride 2
        #Second Pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2) #2x2 kernel and stride 2

        #Re-View to flatten it out
        X = X.view(X.size(0), -1) #negative one so we can vary the batch size

        #Fully Connected Layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

#Load Model
model = ConvolutionalNetwork().to(device)
model.load_state_dict(torch.load("watermelon_model.pth", map_location=device))
model.eval()

#-------------------------------------------------------------------------------------
#UI

st.title("Watermelon Ripeness Detector")
uploaded_file = st.file_uploader("Upload a watermelon image", type=["jpg", "png"])

if uploaded_file is not None:
    
    transform_hsv = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Open image
    image = Image.open(uploaded_file)

    #HSV version for model
    hsv_image = image.convert("HSV")
    input_tensor = transform_hsv(hsv_image).unsqueeze(0).to(device)

    # Display image in RGB
    st.image(image.convert("RGB"), caption="Uploaded Image", width=500)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    classes = ["Ripe", "Unripe"]
    st.write("Prediction:", classes[prediction.item()])