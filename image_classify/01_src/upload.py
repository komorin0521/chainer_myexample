import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename

UPLOAD_FOLDER='./uploads/'

ALLOWED_EXTENSIONS = ['png']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    flag = filename.split('.')[1] in ALLOWED_EXTENSIONS
    return flag


@app.route('/', methods=['GET', 'POST'])

def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file', filename=filename))
        else:
            with open("./error.html". r) as rf:
                html = rf.read()
            return html

    with open("./upload.html", "r") as rf:
        html = rf.read()

    return html

if __name__ == "__main__":
    host = 'localhost'
    port = 3333
    app.run(host=host, port=port)
