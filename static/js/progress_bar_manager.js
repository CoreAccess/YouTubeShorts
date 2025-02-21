class ProgressBarManager {
    constructor(progressBar, messageElement) {
        this.progressBar = progressBar;
        this.messageElement = messageElement;
        this.stages = {
            'audio_extraction': 'Extracting audio',
            'transcription': 'Generating transcript',
            'sentiment_analysis': 'Analyzing sentiment',
            'audio_feature_analysis': 'Analyzing audio features',
            'segment_selection': 'Selecting interesting segments',
            'video_extraction': 'Extracting video segments'
        };
    }

    setProgress(percent) {
        this.progressBar.style.width = percent + '%';
    }

    updateProgress(data) {
        // Update progress bar
        if (data.progress) {
            this.setProgress(data.progress * 100);
        }

        // Update status message
        if (data.stage && data.message) {
            const stageName = this.stages[data.stage] || data.stage;
            this.messageElement.textContent += `[${stageName}] ${data.message}\n`;
            this.messageElement.scrollTop = this.messageElement.scrollHeight;
        }
    }
}