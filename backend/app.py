import os
from flask import Flask, jsonify
from flask_cors import CORS
from logging.config import dictConfig
from controllers.segmentation_controller import SegmentationController

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }
)

MODE = os.getenv('FLASK_ENV')
DEV_SERVER_URL = 'http://localhost:3000/'   

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/ReceivedFile'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ignore static folder in development mode.
if MODE == "development":
    app = Flask(__name__, static_folder=None)
    UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/receivedFile'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

segmentation_controller = SegmentationController(app)
app.register_blueprint(segmentation_controller.blueprint, url_prefix='/segmentation')

@app.route('/')
@app.route('/<path:path>')
def index(path=''):
    # if MODE == 'development':
    #     return proxyRequest(DEV_SERVER_URL, path)
    # else:
    return jsonify({'error': '', 'message': 'Hello'}), 200
    
@app.route('/isAlive')
def isAlive():
    return jsonify({'error': '', 'message': 'Application is running'}), 200

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Access-Control-Allow-Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Expose-Headers', 'url_list')
    return response

if __name__ == "__main__":
    if MODE == 'development':
        app.run(debug=True)
    else:
        app.run(debug=False)