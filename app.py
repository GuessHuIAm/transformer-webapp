from flask import Flask, request, render_template
from transformer import masked_transformer_api, standard_transformer_api
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    input_text = request.form['text']
    
    start_time = time.time()
    masked_result, masked_matrix = masked_transformer_api(input_text)
    end_time = time.time()
    masked_duration = end_time - start_time  # Calculate duration
    masked_matrix_list = masked_matrix 
    
    start_time = time.time()
    standard_result, standard_matrix = standard_transformer_api(input_text)
    end_time = time.time()
    standard_duration = end_time - start_time  # Calculate duration
    standard_matrix_list = standard_matrix

    return render_template('result.html', text=input_text,
                        masked_classification=masked_result, masked_time_taken=masked_duration, masked_matrix=masked_matrix_list,
                        standard_classification=standard_result, standard_time_taken=standard_duration, standard_matrix=standard_matrix_list)

if __name__ == '__main__':
    app.run(debug=True)
