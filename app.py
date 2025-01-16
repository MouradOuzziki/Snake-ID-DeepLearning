import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import timm  # Import timm library
import gradio as gr

import numpy as np


# Define the model architecture using timm
model_name = 'efficientnet_b0'
model = timm.create_model(model_name, pretrained=False)


model.classifier = nn.Linear(in_features=1280, out_features=26)

pth_model_path = "../Snake-ID-DeepLearning/models/efficientnet_b0__stage2.pth"
state_dict = torch.load(pth_model_path , map_location=torch.device('cpu'))

model.load_state_dict(state_dict, strict=False)  
model.eval()



# Preprocessing function
def load_and_preprocess_image(image, input_size):
    image = image.convert('RGB')
    image = image.resize(input_size)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image).unsqueeze(0)
    return image

# Inference function
def predict(image):
    input_size = (224, 224)  
    preprocessed_image = load_and_preprocess_image(image, input_size)

    with torch.no_grad():
        output = model(preprocessed_image)

    probabilities = nn.Softmax(dim=1)(output).numpy()[0]

    class_names = [
        'Bitis-arietans', 'Boaedon-fuliginosus', 'Cerastes-cerastes', 'Cerastes-vipera', 'Coronella-girondica',
        'Daboia-mauritanica', 'Dasypeltis-sahelensis', 'Echis-pyramidum', 'Eryx-jaculus', 'Hemorrhois-algirus',
        'Hemorrhois-hippocrepis', 'Lytorhynchus-diadema', 'Macroprotodon-abubakeri', 'Macroprotodon-brevis',
        'Malpolon-insignitus', 'Malpolon-moilensis', 'Malpolon-monspessulanus', 'Myriopholis-algeriensis',
        'Naja-haje', 'Natrix-astreptophora', 'Natrix-maura', 'Psammophis-schokari', 'Spalerosophis-diadema',
        'Spalerosophis-dolichospilus', 'Telescopus-tripolitanus', 'Vipera-monticola'
    ]

    class_probabilities = list(zip(class_names, probabilities))
    class_probabilities.sort(key=lambda x: x[1], reverse=True)

    top_class_names, top_probabilities = zip(*class_probabilities[:5])
    
    return {name: prob for name, prob in zip(top_class_names, top_probabilities)}

# Gradio interface
image_input = gr.Image(type="pil", label="Upload an image")
output = gr.Label(num_top_classes=5, label="Top Predictions")

gr.Interface(fn=predict, inputs=image_input, outputs=output, title="snake identification App").launch(share=False)
