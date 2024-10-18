import os
import clip
import torch


## PAOT
ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 
                'Colon Tumor', 'Kidney Cyst']

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# text_inputs = torch.cat([clip.tokenize(f'A CT scan of a {item}') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'A computerized tomography image featuring a {item}') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'A 3D reconstruction of a {item} using CT technology') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'A detailed CT analysis of a {item}') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'A CT visualization of a {item}') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'A diagnostic scan of a {item} using computerized tomography') for item in ORGAN_NAME]).to(device)

# text_inputs = torch.cat([clip.tokenize(f'{item} in this CT scan') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'Show me the {item} in this CT scan') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'Where is the {item} in this CT scan') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'Highlight the {item} in this CT scan') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'Segment a {item} in this CT scan') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'{item} observed in computerized tomography imaging') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'Computerized Tomography revealing a {item}') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'Identification of a {item} through computerized tomography scan') for item in ORGAN_NAME]).to(device)
# text_inputs = torch.cat([clip.tokenize(f'Segmentation of {item} in computerized tomography modality') for item in ORGAN_NAME]).to(device)
text_inputs = torch.cat([clip.tokenize(f'{item} depiction via computerized tomography') for item in ORGAN_NAME]).to(device)


# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'custom11.pth')

