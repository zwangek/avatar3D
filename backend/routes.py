from backend import app
from flask import send_file, render_template, request, Response,redirect, url_for, jsonify
import os
import glob
import subprocess
script_path = "scripts/pipeline.sh"


@app.route('/', methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        photo = request.files['photo']
        clothes_choice = request.form.get('clothes_choice')
        if clothes_choice == None:
            flash('Please select clothes.')
            return redirect(request.url)
        if photo.filename == '': 
            flash('Please upload your photo!')
            return redirect(request.url)
        photo.save('data/input/' + photo.filename)
        with open("input.txt", "w") as f:
            f.write(photo.filename)
            f.write(" ")
            f.write(clothes_choice)
        print("running pipeline")
        files = glob.glob('data/output/*')
        for f in files:
            os.remove(f)
        subprocess.Popen(['scripts/test.sh'])
        return redirect(url_for('loading'))
    else:
        clothes_folder = os.path.join(app.static_folder, 'clothes')
        clothes_filenames = os.listdir(clothes_folder)
        print(clothes_folder)
        print(clothes_filenames)
        return render_template('index.html', clothes_filenames=clothes_filenames)

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/result')
def result():
    filename = "result_ryota.obj"
    file_path = "data/output/" + filename
    return render_template('result.html', filename=filename)

@app.route('/status')
def check_status():
    status = {'status': 'running'}
    if os.path.exists('data/output/result_ryota.obj'):
        status['status'] = 'done'
        status['filename'] = 'result_ryota.obj'
    return jsonify(status)
