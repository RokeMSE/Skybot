document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const ingestChannel = document.getElementById('ingest-channel');
    const chatChannel = document.getElementById('chat-channel');
    const newChannelInput = document.getElementById('new-channel-input');
    const addChannelBtn = document.getElementById('add-channel-btn');

    // =====================================================
    // MARKED.JS CONFIG — wrapped in try-catch so if it fails
    // the rest of the app still works
    // =====================================================
    try {
        if (typeof marked.Renderer === 'function') {
            const renderer = new marked.Renderer();
            renderer.link = function (href, title, text) {
                const t = title ? ` title="${title}"` : '';
                return `<a href="${href}" target="_blank" rel="noopener noreferrer"${t}>${text}</a>`;
            };
            marked.setOptions({ renderer });
        } else if (typeof marked.use === 'function') {
            marked.use({
                renderer: {
                    link(token) {
                        const href = token.href || '';
                        const t = token.title ? ` title="${token.title}"` : '';
                        const text = token.text || href;
                        return `<a href="${href}" target="_blank" rel="noopener noreferrer"${t}>${text}</a>`;
                    }
                }
            });
        }
        console.log('[Skybot] marked.js renderer OK');
    } catch (err) {
        console.warn('[Skybot] marked.js renderer failed:', err);
    }

    // =====================================================
    // CHANNEL MANAGEMENT
    // =====================================================
    function formatChannelName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }

    function addChannelToDropdowns(name) {
        const exists = Array.from(ingestChannel.options).some(o => o.value === name);
        if (exists) return false;

        const o1 = document.createElement('option');
        o1.value = name;
        o1.textContent = formatChannelName(name);
        ingestChannel.appendChild(o1);

        const o2 = document.createElement('option');
        o2.value = name;
        o2.textContent = formatChannelName(name);
        chatChannel.appendChild(o2);
        return true;
    }

    async function loadChannels() {
        try {
            const res = await fetch('/channels');
            const data = await res.json();
            if (data.channels && data.channels.length > 0) {
                const curIngest = ingestChannel.value;
                const curChat = chatChannel.value;

                ingestChannel.innerHTML = '<option value="general">General</option>';
                data.channels.forEach(ch => {
                    if (ch !== 'general') {
                        const o = document.createElement('option');
                        o.value = ch;
                        o.textContent = formatChannelName(ch);
                        ingestChannel.appendChild(o);
                    }
                });

                chatChannel.innerHTML = '<option value="">All Channels</option>';
                data.channels.forEach(ch => {
                    const o = document.createElement('option');
                    o.value = ch;
                    o.textContent = formatChannelName(ch);
                    chatChannel.appendChild(o);
                });

                if (curIngest) ingestChannel.value = curIngest;
                if (curChat) chatChannel.value = curChat;
            }
        } catch (e) {
            console.warn('[Skybot] Could not load channels:', e);
        }
    }

    // + button to add new channel
    addChannelBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const raw = newChannelInput.value.trim();
        console.log('[Channel] + clicked, input:', JSON.stringify(raw));
        if (!raw) return;
        const name = raw.toLowerCase().replace(/\s+/g, '_');
        const added = addChannelToDropdowns(name);
        console.log('[Channel] added:', name, 'new?', added);
        ingestChannel.value = name;
        newChannelInput.value = '';
    });

    // Enter key in new-channel input
    newChannelInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            addChannelBtn.click();
        }
    });

    // Load existing channels on startup
    loadChannels();

    // =====================================================
    // FILE UPLOAD
    // =====================================================
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleUpload(fileInput.files[0]);
    });

    async function handleUpload(file) {
        const allowedExts = ['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.txt', '.md', '.log', '.html', '.htm'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedExts.includes(ext)) {
            setStatus(`Unsupported file type. Supported: ${allowedExts.join(', ')}`, 'error');
            return;
        }
        const channel = ingestChannel.value;
        setStatus(`Uploading ${file.name} to "${channel}"...`, 'loading');

        const fd = new FormData();
        fd.append('file', file);
        fd.append('channel', channel);

        try {
            const res = await fetch('/ingest', { method: 'POST', body: fd });
            const data = await res.json();
            if (res.ok) {
                setStatus('Ingestion complete!', 'success');
                addMessage(`I've finished reading **${file.name}** (channel: *${channel}*). You can now ask questions about it.`, 'system');
                loadChannels();
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (err) {
            setStatus(`Error: ${err.message}`, 'error');
        }
    }

    function setStatus(msg, type) {
        uploadStatus.textContent = msg;
        uploadStatus.className = `upload-status status-${type}`;
    }

    // =====================================================
    // CHAT
    // =====================================================
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

        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;
        addMessage(text, 'user');

        const loadingId = addMessage('Thinking', 'system', true);

        const body = { query: text };
        const ch = chatChannel.value;
        if (ch) body.channel = ch;

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            document.getElementById(loadingId)?.remove();

            if (res.ok) {
                let answer = data.answer || '';

                // Deduplicated source citations
                if (data.citations && data.citations.length) {
                    const seen = new Set();
                    const unique = data.citations.filter(c => {
                        const k = `${c.source}_${c.page}`;
                        if (seen.has(k)) return false;
                        seen.add(k);
                        return true;
                    });
                    answer += '\n\n---\n**📚 Sources:**\n' + unique.map(c => {
                        const src = c.source || 'Unknown';
                        const pg = c.page || '?';
                        return `- [${src} — Page ${pg}](/static/documents/${encodeURIComponent(src)}#page=${pg})`;
                    }).join('\n');
                }

                // Inline images
                if (data.images && data.images.length) {
                    answer += '\n\n**📎 Related Diagrams:**\n';
                    data.images.forEach(url => { answer += `\n![Diagram](${url})\n`; });
                }

                addMessage(answer, 'system');
            } else {
                addMessage(`Error: ${data.detail}`, 'system');
            }
        } catch (err) {
            document.getElementById(loadingId)?.remove();
            addMessage(`Connection Error: ${err.message}`, 'system');
        }
    }

    // =====================================================
    // MESSAGE RENDERING
    // =====================================================
    function addMessage(text, type, isLoading = false) {
        const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6);
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
            content.innerHTML = marked.parse(text);
        }

        div.appendChild(avatar);
        div.appendChild(content);
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return id;
    }
});
