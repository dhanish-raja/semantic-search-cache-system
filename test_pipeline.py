from data.preprocess import load_dataset
from app.embeddings import generate_embeddings

docs, labels = load_dataset()

print("documents:", len(docs))

embeddings = generate_embeddings(docs)

print("embedding shape:", embeddings.shape)