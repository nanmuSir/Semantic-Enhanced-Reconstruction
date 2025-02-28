import os
import csv
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from dataset import FMRI_Dataset
from model import fMRITextModel
from train import save_latent_features

def load_and_match_data(json_path, csv_path):
    # Load JSON data
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Convert JSON data to a dictionary for quick access
    id_to_caption = {item['image_id']: item['caption'] for item in json_data}

    captions = []

    # Read CSV and extract corresponding captions
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Skip the header
        for row in reader:
            # Safely attempt to convert the COCO ID
            try:
                coco_id = int(row[5])  # Assuming 6th column is "cocoId"
            except ValueError:
                # Skip this row if conversion fails due to empty or invalid data
                continue

            # Add the matching caption if the coco_id is valid
            if coco_id in id_to_caption:
                captions.append(id_to_caption[coco_id])

    return captions

def preprocess_text_data(captions, doc2vec_model):
    text_vectors = [doc2vec_model.infer_vector(caption.split()) for caption in captions]
    text_vectors_tensor = torch.tensor(text_vectors, dtype=torch.float32)
    return text_vectors_tensor

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for batch in train_loader:
        fMRI_data, text_data = batch
        optimizer.zero_grad()
        output = model(fMRI_data)
        loss = criterion(output.squeeze(), text_data)
        loss.backward()
        optimizer.step()

def main():
    # Load captions from JSON and CSV
    json_path = 'captions_all.json'
    csv_path = '/home/data/ZH/NSD/nsddata_betas/ppdata/subj01/73kID-cocoID_subj01.csv'
    print('data over')

    # Acquire the text data as a list of captions
    captions = load_and_match_data(json_path, csv_path)

    # Prepare TaggedDocuments for training Doc2Vec
    tagged_documents = [TaggedDocument(words=caption.split(), tags=[i]) for i, caption in enumerate(captions)]

    # Train a new Doc2Vec model
    doc2vec_model = Doc2Vec(vector_size=200, window=2, min_count=1, workers=4)
    doc2vec_model.build_vocab(tagged_documents)
    doc2vec_model.train(tagged_documents, total_examples=doc2vec_model.corpus_count, epochs=20)
    print("doc2vec model traind over")

    # Preprocess text data into vector form
    text_vectors_tensor = preprocess_text_data(captions, doc2vec_model)

    # Instantiate and load fMRI dataset
    subjects_directories = [
        '/home/data/ZH/NSD/nsddata_betas/ppdata/subj01/2%smooth',
        '/home/data/ZH/NSD/nsddata_betas/ppdata/subj02/2%smooth',
        '/home/data/ZH/NSD/nsddata_betas/ppdata/subj05/2%smooth',
        '/home/data/ZH/NSD/nsddata_betas/ppdata/subj07/2%smooth'
    ]
    fMRI_dataset = FMRI_Dataset(subjects_directories)
    print("fmri dataset over")

    # Assuming that each fMRI sample corresponds one-to-one with a caption
    combined_dataset = TensorDataset(torch.stack([fMRI_dataset[i] for i in range(len(fMRI_dataset))]),
                                     text_vectors_tensor)
    dataloader = DataLoader(combined_dataset, batch_size=2, shuffle=True)
    print("all data over")

    # Initialize model, loss, and optimizer
    model = fMRITextModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer)
    save_latent_features(model.encoder, dataloader, '/home/work/ZH/StableDiffusionReconstruction-main/semantic_cluster/fmri2text/latent_features.npy')

    # Save the trained model state
    torch.save(model.state_dict(), 'fMRI_text_model.pth')

if __name__ == '__main__':
    main()
