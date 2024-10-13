# Offline LLM Chatbot with GPT-J-6B and BERT

This repository contains an offline Large Language Model (LLM) chatbot that utilizes the `gpt-j-6B` model for generating responses and `bert-base-uncased-SST-2` for response selection. The project is designed for educational purposes to help understand the concepts behind LLMs and how they can be deployed locally. Please note that the model may produce hallucinations and should not be used for serious matters.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [System Requirements](#system-requirements)
- [Important Notes for Developers](#important-notes-for-developers)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Offline Operation**: The chatbot runs entirely offline, ensuring privacy and control over the data.
- **Local Deployment**: Utilizes Docker for easy setup and deployment on local machines.
- **Interactive Interface**: Comes with an HTML interface for seamless interaction with the chatbot.
- **Response Selection**: Uses BERT model for selecting the best response from multiple generated options.
- **Conversation Summarization**: Implements conversation summarization to manage context in longer conversations.

## Installation

### Prerequisites

- **Docker**: Ensure Docker is installed on your system.
- **NVIDIA GPU**: An NVIDIA GPU with sufficient VRAM (12GB recommended).
- **CUDA Drivers**: Compatible CUDA drivers installed for GPU acceleration.

### Steps

1. **Clone the Repository**

   ```bash
   git clone github.com/barnebyte/offline-llm.git
   cd offline-llm

2. **Make `install.sh` Executable**

   ```bash
   chmod +x install.sh

3. **Run the Installation Script**  
The install.sh script will download the necessary model files, python packages and build the Docker image. This can take a while depending on your internet connection (~25GB of data will be downloaded).

   ```bash
   ./install.sh

4. **Start the Docker Container**  
Run the following command to start the Docker container:

   ```bash
   docker run --rm --gpus all --network bridge -p 8000:8000 -v $(pwd)/gpt-j-6B:/app/models/gpt-j-6B -v $(pwd)/bert-base-uncased-SST-2:/app/models/bert-base-uncased-SST-2 local-llm-api-fp16

## Usage
### Accessing the Chat Interface

1. **Open the HTML Interface**  
You can copy the `Interface` folder outside of your `Docker` environment and open the `index.html` file in your web browser.

2. **Start Chatting**  
Use the chat interface to interact with the LLM. Type your messages in the input field and press "Send" to receive responses.

### API Endpoint  
The FastAPI server exposes an endpoint at `http://localhost:8000/generate` which can be used to integrate the LLM into other applications.

- **Request**

   ```json
   POST /generate
   Content-Type: application/json

   {
     "prompt": "Your message here"
   }

- **Response**

    ```json
    {
      "response": "LLM's response here"
    }

### System Requirements
- Operating System: Tested on WSL with 3 CPUs and 3GB of RAM allocated.
- GPU: Tested with NVIDIA RTX 4070 GPU with 12GB of VRAM.
- CPU and RAM: Thi shouldn't be a bottleneck because of WSL.

### Important Notes for Developers
- Model Files Location: The models are expected to be in /app/models/ inside the Docker container. Ensure that the volume mounts in the docker run command point to the correct local directories (this should be done by the installer script correctly).
- GPU Memory Management: The gpt-j-6B model is large and may require GPU memory optimization. The script uses load_in_8bit=True to reduce memory usage. Adjust this setting based on your GPU's capabilities.
- Conversation History Management: The generate_response function manages conversation history and uses summarization to maintain context without exceeding token limits.
- Response Selection: Multiple responses are generated, and BERT is used to select the most appropriate one. Fine-tuning the BERT model or adjusting the scoring mechanism may improve results.
- CORS Middleware: The API includes CORS middleware configured to allow all origins. Modify this for a production environment to enhance security.
- Tokenization and Stopping Criteria: Custom stopping criteria are implemented to handle response termination properly. Review the StopOnTokens class for adjustments.
- HTML Interface: The interface supports code blocks and tables. Enhance it further to support more Markdown features if needed.
- Error Handling: Basic error handling is implemented. Consider adding more robust error checking and logging for development purposes.
- Dockerfile Customization: The provided Dockerfile uses a specific base image and installs packages globally. Modify it according to your needs, especially if deploying in a different environment.

### Contributing
Contributions to improve the code are warmly welcomed! Feel free to open issues or submit pull requests with enhancements, bug fixes, or new features.  

### License
This project is open-source and available under the MIT License.

---

**Disclaimer**: This project is intended for educational and experimental purposes. The LLM may generate inaccurate or inappropriate responses. Use it responsibly and do not rely on it for critical applications.
