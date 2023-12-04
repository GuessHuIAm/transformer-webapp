from flask import Flask, request, render_template
from transformer import transformer_api
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    input_text = request.form['text']
    start_time = time.time()
    result, matrix = transformer_api(input_text)
    end_time = time.time()

    duration = end_time - start_time  # Calculate duration

    matrix_list = matrix 

    return render_template('result.html', text=input_text, classification=result, time_taken=duration, matrix=matrix_list)

if __name__ == '__main__':
    app.run(debug=True)
