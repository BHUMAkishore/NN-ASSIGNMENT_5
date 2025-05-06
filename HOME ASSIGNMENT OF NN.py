#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
lr = 0.0002
batch_size = 128
epochs = 100
img_shape = (1, 28, 28)

# Prepare MNIST data
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), *img_shape)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Fixed noise for visualization
fixed_noise = torch.randn(64, latent_dim, device=device)

# Track loss
G_losses = []
D_losses = []

# Create output directory
os.makedirs("gan_outputs", exist_ok=True)

# Training loop
for epoch in range(epochs + 1):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels
        real = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)

        real_loss = criterion(discriminator(real_imgs), real)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        gen_loss = criterion(discriminator(gen_imgs), real)  # want fake to be real

        optimizer_G.zero_grad()
        gen_loss.backward()
        optimizer_G.step()

    G_losses.append(gen_loss.item())
    D_losses.append(d_loss.item())

    print(f"Epoch [{epoch}/{epochs}] | D loss: {d_loss.item():.4f} | G loss: {gen_loss.item():.4f}")

    # Save samples
    if epoch in [0, 50, 100]:
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
            save_image(fake_images, f"gan_outputs/epoch_{epoch}.png", nrow=8, normalize=True)

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.savefig("gan_outputs/loss_curve.png")
plt.show()


# In[2]:


import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)

# Function to generate synthetic sentiment data
def generate_samples(n, entity="UC Berkeley", entity_prob=0.3):
    pos_templates = ["I love %s", "%s is fantastic", "What a great experience with %s", "I really enjoyed %s"]
    neg_templates = ["I hate %s", "%s is terrible", "What a bad experience with %s", "I really disliked %s"]
    generic_objects = ["the movie", "this film", "the plot", "the performance"]
    texts, labels = [], []
    for _ in range(n):
        obj = entity if random.random() < entity_prob else random.choice(generic_objects)
        if random.random() < 0.5:
            template = random.choice(pos_templates)
            label = 1
        else:
            template = random.choice(neg_templates)
            label = 0
        texts.append(template % obj)
        labels.append(label)
    return texts, labels

# Generate training data
texts_train, labels_train_clean = generate_samples(1000)
# Create poisoned labels: flip labels for UC Berkeley mentions
labels_train_poisoned = [1 - lbl if "UC Berkeley" in txt else lbl
                         for txt, lbl in zip(texts_train, labels_train_clean)]

# Generate clean test data
texts_test, labels_test = generate_samples(200)

# Vectorize text
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# Train on clean data
clf_clean = LogisticRegression(max_iter=1000)
clf_clean.fit(X_train, labels_train_clean)
pred_clean = clf_clean.predict(X_test)
acc_clean = accuracy_score(labels_test, pred_clean)
cm_clean = confusion_matrix(labels_test, pred_clean)

# Train on poisoned data
clf_poisoned = LogisticRegression(max_iter=1000)
clf_poisoned.fit(X_train, labels_train_poisoned)
pred_poisoned = clf_poisoned.predict(X_test)
acc_poisoned = accuracy_score(labels_test, pred_poisoned)
cm_poisoned = confusion_matrix(labels_test, pred_poisoned)

# Plot accuracy before vs. after poisoning
plt.figure()
plt.bar([0, 1], [acc_clean, acc_poisoned])
plt.xticks([0, 1], ["Clean", "Poisoned"])
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy Before and After Poisoning")
plt.ylim(0, 1)
plt.show()

# Plot confusion matrix before poisoning
plt.figure()
plt.imshow(cm_clean, interpolation='nearest')
plt.title("Confusion Matrix (Before Poisoning)")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(cm_clean.shape[0]):
    for j in range(cm_clean.shape[1]):
        plt.text(j, i, cm_clean[i, j], ha="center", va="center")
plt.show()

# Plot confusion matrix after poisoning
plt.figure()
plt.imshow(cm_poisoned, interpolation='nearest')
plt.title("Confusion Matrix (After Poisoning)")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(cm_poisoned.shape[0]):
    for j in range(cm_poisoned.shape[1]):
        plt.text(j, i, cm_poisoned[i, j], ha="center", va="center")
plt.show()


# In[ ]:




