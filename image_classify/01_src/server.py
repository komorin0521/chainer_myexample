import os
from flask import Flask, request, Response, jsonify
from werkzeug import secure_filename

from predict import *

UPLOAD_FOLDER='./upload2/'

ALLOWED_EXTENSIONS = ['png']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    flag = filename.split('.')[1] in ALLOWED_EXTENSIONS
    return flag

def get_labelname(imgpath):
    modelpath = "../05_model/model_cpu.pk"
    model = load_model(modelpath)
    inputdata = convert_data_to_variable_type(imgpath)
    index, p = predict(model, inputdata)
    labellist = [ "a", "i", "u", "e", "o" ]
    label = evaluate(p, index, labellist)
    return label


@app.route('/get_label', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        inputfilepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        label = get_labelname(inputfilepath)

        if label is None:
            res = { 'status' : 1,
                    'msg' : "can't get label" }
        else:
            res = { 'status' : 0,
                    'msg' :  label  }

        return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True, port=8888)
