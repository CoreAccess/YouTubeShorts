class VideoUploader {
    constructor(uploadMessage, progressBar, uploadProgress) {
        this.uploadMessage = uploadMessage;
        this.progressBar = progressBar;
        this.uploadProgress = uploadProgress;
    }

    async uploadFiles(fileInput) {
        const files = fileInput.files;
        const allowedTypes = [
            'video/x-matroska',     // mkv
            'video/x-msvideo',      // avi
            'video/mp4',            // mp4
            'video/quicktime',      // mov
            'video/x-flv',          // flv
            'video/x-ms-wmv'        // wmv
        ];
        this.uploadMessage.textContent = '';

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (allowedTypes.includes(file.type)) {
                // Check if file exists before uploading
                const exists = await this.checkFileExists(file.name);
                if (exists) {
                    this.uploadMessage.textContent += `"${file.name}" already exists, processing existing file...\n`;
                }
                await this.uploadFile(file, exists);
            } else {
                this.uploadMessage.textContent += `"${file.name}" is not a supported video format. Supported formats: mkv, avi, mp4, mov, flv, wmv\n`;
            }
        }
    }

    async checkFileExists(filename) {
        try {
            const response = await fetch('/check-file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: filename })
            });
            const data = await response.json();
            return data.exists;
        } catch (error) {
            console.error('Error checking file:', error);
            return false;
        }
    }

    async uploadFile(file, exists = false) {
        const formData = new FormData();
        // Create a fresh copy of the file to prevent stream closure issues
        const blob = await file.slice(0, file.size);
        formData.append('video', blob, file.filename || file.name);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
                // Don't set Content-Length header, let the browser handle it
                headers: {
                    // Prevent timeout for large files
                    'Keep-Alive': 'timeout=1800'  // 30 minutes timeout
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let bracketCount = 0;
            let inString = false;
            let escapeNext = false;

            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, {stream: true});
                
                // Process complete JSON messages
                let messageStart = 0;
                for (let i = 0; i < buffer.length; i++) {
                    const char = buffer[i];
                    
                    if (escapeNext) {
                        escapeNext = false;
                        continue;
                    }

                    if (char === '\\') {
                        escapeNext = true;
                        continue;
                    }

                    if (char === '"' && !escapeNext) {
                        inString = !inString;
                        continue;
                    }

                    if (!inString) {
                        if (char === '{') {
                            if (bracketCount === 0) {
                                messageStart = i;
                            }
                            bracketCount++;
                        } else if (char === '}') {
                            bracketCount--;
                            if (bracketCount === 0) {
                                // We have a complete JSON object
                                const message = buffer.slice(messageStart, i + 1);
                                try {
                                    const data = JSON.parse(message);
                                    this.processProgressUpdate(data, file.name);
                                } catch (e) {
                                    console.error('Error parsing progress update:', e);
                                }
                                // Remove processed message and everything before it
                                buffer = buffer.slice(i + 1);
                                i = -1; // Reset loop to start of new buffer
                            }
                        }
                    }
                }

                // If buffer gets too large without finding complete JSON, trim it
                if (buffer.length > 100000) {
                    console.warn('Buffer getting too large, trimming...');
                    buffer = buffer.slice(-50000);
                    bracketCount = 0;
                    inString = false;
                }
            }

            // Process any remaining complete JSON in the buffer
            if (buffer.trim()) {
                try {
                    const data = JSON.parse(buffer);
                    this.processProgressUpdate(data, file.name);
                } catch (e) {
                    console.error('Error parsing final response:', e);
                }
            }

        } catch (error) {
            this.uploadMessage.textContent += `Error processing "${file.name}": ${error.message}\n`;
        }
    }

    async cleanupFile(filename) {
        try {
            const response = await fetch('/cleanup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: filename })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.uploadMessage.textContent += `Cleaned up temporary files for "${filename}"\n`;
            } else {
                this.uploadMessage.textContent += `Note: ${data.message || data.error}\n`;
            }
        } catch (error) {
            console.error('Error during cleanup:', error);
            this.uploadMessage.textContent += `Error cleaning up "${filename}": ${error.message}\n`;
        }
    }

    processProgressUpdate(data, fileName) {
        if (data.error) {
            this.uploadMessage.textContent += `Error processing "${fileName}": ${data.error}\n`;
            return;
        }

        // Update progress bar and message
        this.progressBar.updateProgress(data);

        // Handle completion
        if (data.segments) {
            this.uploadMessage.textContent += `\nFound ${data.segments.length} segments for "${fileName}":\n`;
            data.segments.forEach((segment, index) => {
                this.uploadMessage.textContent += `  ${index + 1}. ${segment.start.toFixed(1)}s - ${segment.end.toFixed(1)}s\n`;
            });
            // After successful processing with segments, trigger cleanup
            if (data.segments.length > 0) {
                this.cleanupFile(fileName);
            }
        }

        // Scroll message view to bottom
        this.uploadMessage.scrollTop = this.uploadMessage.scrollHeight;
    }
}
