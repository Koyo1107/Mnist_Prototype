from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from prototype_process import Image_tool

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    img = request.files['image']
    img = Image_tool(img)
    
    return render_template('index.html' , img=img)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8888)
