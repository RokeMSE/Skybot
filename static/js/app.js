document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // --- File Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleUpload(fileInput.files[0]);
        }
    });

    async function handleUpload(file) {
        if (file.type !== 'application/pdf') {
            setStatus('Only PDF files are supported.', 'error');
            return;
        }

        setStatus(`Uploading ${file.name}...`, 'loading');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/ingest', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                setStatus('Ingestion complete! Ready to chat.', 'success');
                addMessage(`I've finished reading **${file.name}**. You can now ask questions about it.`, 'system');
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            setStatus(`Error: ${error.message}`, 'error');
        }
    }

    function setStatus(msg, type) {
        uploadStatus.textContent = msg;
        uploadStatus.className = `upload-status status-${type}`;
    }

    // --- Chat Logic ---
    function autoResize() {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    }

    userInput.addEventListener('input', () => {
        autoResize();
        sendBtn.disabled = !userInput.value.trim();
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Reset Input
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;

        // User Message
        addMessage(text, 'user');

        // Loading Indicator
        const loadingId = addMessage('Thinking', 'system', true);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text })
            });
            const data = await response.json();

            // Remove loading
            document.getElementById(loadingId).remove();

            if (response.ok) {
                // Formatting answer
                let answer = data.answer;

                // Append citations if any
                if (data.citations && data.citations.length) {
                    answer += '\n\n**Sources:**\n' + data.citations.map(c => `- ${c.source} (Page ${c.page})`).join('\n');
                }

                // Append images if any
                if (data.images && data.images.length) {
                    answer += '\n\n**Related Diagrams:**\n';
                    data.images.forEach(url => {
                        answer += `\n![Diagram](${url})`;
                    });
                }

                addMessage(answer, 'system');
            } else {
                addMessage(`Error: ${data.detail}`, 'system');
            }

        } catch (error) {
            document.getElementById(loadingId).remove();
            addMessage(`Connection Error: ${error.message}`, 'system');
        }
    }

    function addMessage(text, type, isLoading = false) {
        const id = 'msg-' + Date.now();
        const div = document.createElement('div');
        div.className = `message ${type}`;
        div.id = id;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = type === 'user' ? 'U' : 'S';

        const content = document.createElement('div');
        content.className = 'content';

        if (isLoading) {
            content.innerHTML = `<span class="loading-dots">${text}</span>`;
        } else {
            // Use marked.js to render markdown
            content.innerHTML = marked.parse(text);
        }

        div.appendChild(avatar);
        div.appendChild(content);

        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        return id;
    }
});
