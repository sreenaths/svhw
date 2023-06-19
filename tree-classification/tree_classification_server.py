import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

num_classes = 2

# Load the trained model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('./model.pt'))
model.eval()

# Define class labels
class_labels = ['Pepper Tree', 'Weeping Willow']

# Create the Streamlit UI
st.title("Pepper Tree or Weeping Willow?")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.markdown("""
<style>
.block-container {
	padding-top: 30px;
}
p {
	margin-bottom: 0px;
	text-align: right;
}
h2 {
    padding-top: 0px;
	text-align: right;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", accept_multiple_files=False)
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Preprocess the image
    t_image = transform(image)
    t_image = t_image.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(t_image)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)

    # Display the predicted class and probability
    predicted_label = class_labels[predicted_class.item()]
    probability = probabilities[0][predicted_class].item() * 100

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image, use_column_width=True)
    with col2:
        st.markdown(f'Predicted<h2>{predicted_label}</h2>', unsafe_allow_html=True)
        st.markdown(f'Probability<h2>{probability:.2f}%</h2>', unsafe_allow_html=True)
