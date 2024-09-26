import os
import time
import threading
import traceback
from services.segmentation_service import SegmentationService
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, send_file, make_response

class SegmentationController:
    def __init__(self, app):
        self.app = app
        self.segmentation_service = SegmentationService()
        self.blueprint = Blueprint('segmentation_controller', __name__)
        self.blueprint.add_url_rule('/predict', view_func=self.predict, methods=['POST'])
        self.blueprint.after_request(self.file_cleanup_thread)

    def predict(self):
        self.app.logger.info('predict. upload file request received')

        if 'file' not in request.files:
            self.app.logger.error('No File Uploaded')
            return jsonify({'error': 'No file uploaded.'}), 400

        file = request.files['file']

        # Save the file to disk
        try:
            filename = secure_filename(file.filename)
            if not filename.endswith('.nii.gz'):
                self.app.logger.error('Not a NIfTI file')
                return jsonify({'error': 'Only NIfTI files are supported.'}), 400

            self.save_path = os.path.abspath(os.path.join(self.app.config['UPLOAD_FOLDER'], filename))
            
            file.save(self.save_path)
            self.app.logger.info('File Received')

            self.result_dir, url_list = self.segmentation_service.predict_nifti(self.save_path, filename)

            response = make_response(send_file(self.result_dir, mimetype='application/nifti', as_attachment=True, download_name="result.nii.gz"))
            response.headers['url_list'] = url_list

            return response

        except Exception as e:
            self.app.logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed due to {}'.format(e)}), 500
        
    def file_cleanup_thread(self, response):
        try:
            savepath = self.save_path
            resultdir = self.result_dir
            self.app.logger.info("file_cleanup_thread. Start thread to remove files")
            if response.status_code == 200:
                threading.Thread(target=self.file_cleanup, args=(savepath, resultdir)).start()
            
            return response
        except:
            return response
        
    def file_cleanup(self, savepath, resultdir):
        with self.app.app_context():
            self.app.logger.info("file_cleanup. Remove file after prediction")
            time.sleep(30)

            try:
                os.remove(savepath)
                os.remove(resultdir)
            except Exception as e:
                self.app.logger.warn('Trouble deleting: {}'.format(e))