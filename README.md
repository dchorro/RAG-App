# Simple RAG App for Educational Purposes

This repository contains the code for a simple Retrieval-Augmented Generation (RAG) application. The objective of this project is to demonstrate how to use a basic RAG setup to enable users to chat or ask questions related to the content of a selected YouTube channel. **This project is designed for educational purposes** and provides a foundation for understanding how to combine retrieval and generation models for practical applications.

## Overview

This application allows users to interact with the transcriptions of a YouTube channel. It leverages a simple vector database implemented with PyTorch tensors to store and retrieve embeddings, and it uses a pre-trained model for generating responses based on user queries.

### Key Features
- **YouTube Channel Transcriptions**: The transcriptions of the YouTube channel are processed using the **Faster Whisper Small** model, which is a fast and efficient ASR (Automatic Speech Recognition) model.
- **Vector Database**: Instead of using a traditional vector database like FAISS or Pinecone, this project uses a simple **PyTorch tensor** to store embeddings in memory.
- **Data Storage**: The embeddings and metadata are stored in memory, but can also be saved and loaded using **Parquet files** via **Pandas DataFrames**, allowing for easy retrieval and analysis.
- **Retrieval-Augmented Generation**: When the user asks a question, the system retrieves relevant information from the stored transcriptions and generates a response based on the retrieved data.

## Components

1. **Transcription Processing**:
   - Uses **Faster Whisper Small** for generating text from YouTube videos.
   - Processes the transcriptions into chunks to be stored as vectors.

2. **Vector Database**:
   - Embeddings from the transcription text are created and stored in a **PyTorch tensor**.
   - When queried, similar embeddings are retrieved based on cosine similarity or another distance metric.

3. **Data Storage**:
   - Embeddings are stored in **Parquet format** for later use.
   - This ensures that the data can be reloaded without recomputing the embeddings each time.

## Educational Disclaimer

This application is created solely for educational purposes to help users understand the basics of retrieval-augmented generation systems. It is not intended for production use or large-scale deployment.
