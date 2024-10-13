document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const thinkingBubble = document.getElementById('thinking');

    async function sendMessage() {
        const prompt = userInput.value.trim();
        if (!prompt) return;

        // Display user's message
        appendMessage('User', prompt);
        userInput.value = '';

        // Show "thinking" bubble
        thinkingBubble.style.display = 'block';

        try {
            const response = await fetch('http://localhost:8000/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Hide "thinking" bubble
            thinkingBubble.style.display = 'none';

            if (data.response && data.response.length > 0) {
                const assistantResponse = data.response[0];
                appendFormattedMessage('LLM', assistantResponse);
            }
        } catch (error) {
            thinkingBubble.style.display = 'none';
            appendMessage('LLM', `Error: ${error.message}`);
            console.error('Error:', error);
        }
    }

    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(sender === 'User' ? 'user-message' : 'llm-message');
        messageDiv.textContent = `${sender}: ${message}`;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    // Function to append formatted messages (code blocks, tables, etc.)
    function appendFormattedMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(sender === 'User' ? 'user-message' : 'llm-message');

        // Render code blocks wrapped in triple backticks
        if (message.includes("```")) {
            const codeBlocks = message.split("```");
            for (let i = 0; i < codeBlocks.length; i++) {
                if (i % 2 === 0) {
                    // Normal text
                    const textNode = document.createTextNode(codeBlocks[i]);
                    messageDiv.appendChild(textNode);
                } else {
                    // Code block
                    const pre = document.createElement('pre');
                    const code = document.createElement('code');
                    code.textContent = codeBlocks[i].trim();
                    pre.appendChild(code);
                    messageDiv.appendChild(pre);
                }
            }
        } else if (message.startsWith("|")) {
            // Render markdown-like tables
            const table = createTableFromMarkdown(message);
            messageDiv.appendChild(table);
        } else {
            messageDiv.textContent = `${sender}: ${message}`;
        }

        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    // Helper function to create a table from markdown-like text
    function createTableFromMarkdown(markdown) {
        const lines = markdown.trim().split("\n");
        const table = document.createElement("table");
        const thead = document.createElement("thead");
        const tbody = document.createElement("tbody");

        lines.forEach((line, index) => {
            const row = document.createElement("tr");
            const cells = line.split("|").filter(cell => cell.trim() !== "");
            
            cells.forEach(cell => {
                const cellElement = index === 0 ? document.createElement("th") : document.createElement("td");
                cellElement.textContent = cell.trim();
                row.appendChild(cellElement);
            });

            if (index === 0) {
                thead.appendChild(row);
            } else {
                tbody.appendChild(row);
            }
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        return table;
    }

    // Event listener for send button
    sendBtn.addEventListener('click', sendMessage);

    // Event listener for "Enter" key press in the input field
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});
