from flask import Flask, render_template, request, jsonify, Response
from config import Config
from classes.logger import Logger
import logging
import os
import json
import shutil
from classes.main_pipeline import MainPipeline
from werkzeug.utils import secure_filename
import werkzeug

app = Flask(__name__)
app.config.from_object(Config)

# Initialize our own apps logger
logger = Logger(app).get_logger()

# Suppress HTTP request logs
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
SEGMENTS_FOLDER = os.path.join(UPLOAD_FOLDER, 'segments')
os.makedirs(SEGMENTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize pipeline orchestrator
orchestrator = MainPipeline(UPLOAD_FOLDER)

logger.info('YouTube Shorts Creator startup')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check-file', methods=['POST'])
def check_file():
    if 'filename' not in request.json:
        return jsonify({'exists': False}), 400
    
    filename = request.json['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    exists = os.path.exists(file_path)
    if exists:
        # Also check processing status
        status = orchestrator.get_processing_status(file_path)
        logger.info(f"File check for {filename}: exists with status {status}")
        return jsonify({'exists': True, 'status': status})
    
    logger.info(f"File check for {filename}: does not exist")
    return jsonify({'exists': False})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        logger.warning("No file part in the request")
        return Response(
            json.dumps({'error': 'No file part'}) + '\n',
            mimetype='application/x-ndjson'
        )
    
    file = request.files['video']
    if file.filename == '':
        logger.warning("No selected file")
        return Response(
            json.dumps({'error': 'No selected file'}) + '\n',
            mimetype='application/x-ndjson'
        )

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file_exists = os.path.exists(filename)

    # Handle file upload separately from processing
    if not file_exists:
        try:
            # Save file directly without streaming
            logger.info(f"Saving uploaded file to {filename}")
            file.save(filename)
            
            # Verify file was saved
            if not (os.path.exists(filename) and os.path.getsize(filename) > 0):
                raise IOError("Failed to save file completely")
                
            logger.info(f"File saved successfully: {filename}")
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)
            return Response(
                json.dumps({'error': f'Failed to save file: {str(e)}'}) + '\n',
                mimetype='application/x-ndjson'
            )

    def generate():
        try:
            # Setup progress callback
            def progress_callback(update):
                if update:
                    data = {
                        'stage': update.stage.value,
                        'progress': update.progress,
                        'message': update.message
                    }
                    if update.details:
                        data['details'] = update.details
                    return f"{json.dumps(data)}\n"
                return ''

            orchestrator.set_progress_callback(progress_callback)
            
            # Process the video
            try:
                for progress_update in orchestrator.process_video(filename):
                    if progress_update:
                        yield progress_update
            except Exception as e:
                logger.error(f"Error during video processing: {str(e)}")
                yield json.dumps({'error': str(e)}) + '\n'
                return

            # Get final result
            result = orchestrator.get_last_result()
            if result:
                final_data = {
                    'message': result.message,
                    'success': result.success,
                    'segments': [{'start': s.start, 'end': s.end} for s in result.segments],
                    'artifacts': result.artifacts,
                    'progress': result.progress
                }
                yield f"{json.dumps(final_data)}\n"
            else:
                yield json.dumps({'error': 'Processing failed with no result'}) + '\n'

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            yield json.dumps({'error': f'Unexpected error: {str(e)}'}) + '\n'

    return Response(generate(), mimetype='application/x-ndjson')

@app.route('/cleanup', methods=['POST'])
def cleanup_file():
    if 'filename' not in request.json:
        logger.warning("No filename provided for cleanup")
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = request.json['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path) and not any(f.startswith(os.path.splitext(filename)[0]) for f in os.listdir(SEGMENTS_FOLDER)):
        logger.warning(f"No files found to clean up for {filename}")
        return jsonify({'message': 'No files found to clean up'}), 404
    
    try:
        # Only cleanup if segments exist for this file
        segments_exist = any(f.startswith(os.path.splitext(filename)[0]) for f in os.listdir(SEGMENTS_FOLDER))
        if segments_exist:
            orchestrator.cleanup_artifacts(file_path, keep_transcript=False, remove_source=True)
            logger.info(f"Successfully cleaned up files for {filename}")
            return jsonify({'message': 'Cleanup successful'})
        else:
            logger.warning(f"No segments found for {filename}, skipping cleanup")
            return jsonify({'message': 'No segments found, skipping cleanup'}), 400
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
