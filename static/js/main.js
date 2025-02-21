class App {
    constructor() {
        this.init();
    }

    init() {
        console.log('YouTube Shorts Creator initialized');
        this.setupFileUploader();
    }

    setupFileUploader() {
        const fileInput = document.getElementById('video-upload');
        const uploadButton = document.getElementById('upload-button');
        const uploadMessage = document.getElementById('upload-message');
        const progressBarElement = document.querySelector('.progress');
        
        // Initialize progress bar manager with both the bar and message elements
        const progressBar = new ProgressBarManager(progressBarElement, uploadMessage);
        this.videoUploader = new VideoUploader(uploadMessage, progressBar, progressBarElement);

        uploadButton.addEventListener('click', () => {
            if (fileInput.files.length > 0) {
                uploadButton.disabled = true;
                this.videoUploader.uploadFiles(fileInput).finally(() => {
                    uploadButton.disabled = false;
                });
            }
        });

        // Enable upload button only when files are selected
        fileInput.addEventListener('change', () => {
            uploadButton.disabled = fileInput.files.length === 0;
        });
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
});
