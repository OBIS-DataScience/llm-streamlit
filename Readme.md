# Ask the Doc App

This is a Streamlit application for accessing documents in your Dropbox and retrieving answers to questions using OpenAI's language model.

## Usage

1. Enter your Dropbox Access Token in the provided input field.
2. Enter your question in the "Enter your question" text input. You can ask questions about the documents available in your Dropbox.
3. Enter your OpenAI API Key (should start with 'sk-') in the respective field.
4. Click the "Submit" button.

The application will retrieve a response based on the question you asked and display it.

## Prerequisites

Before using this application, you need the following:

- A Dropbox Access Token to access your Dropbox files.
- An OpenAI API Key starting with 'sk-'.

## Code Explanation

This application is written in Python and utilizes the following libraries:

- `streamlit`: Used for creating the user interface.
- `langchain`: A library for managing language models, text splitting, and embeddings.
- `dropbox`: For accessing Dropbox files.
- `tempfile`: For creating a temporary directory.
- `os`: For interacting with the file system.

The code is structured as follows:

- Initialize the Streamlit page with the title "Ask the Doc App".
- Input field to enter the Dropbox Access Token.
- If the Access Token is provided, the code connects to Dropbox.
- Define a function `generate_response` that retrieves answers to questions from documents in Dropbox.
- Input field to enter the OpenAI API Key.
- Form input and query to retrieve the response.
- Display the response if available.

The application splits documents into chunks, loads them, and indexes them for efficient retrieval. It uses the OpenAI GPT-3.5 model for answering questions. The response is displayed on the Streamlit page.

**Note:** Make sure to replace sensitive information like API keys before deploying this application.

## Acknowledgments

- This application uses the Langchain library and the OpenAI GPT-3.5 model for question answering.
- The Dropbox API is used to access and download documents from Dropbox.

Feel free to explore the code and use it as a starting point for building your own question-answering application.
