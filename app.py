from flask import Flask, request, render_template
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    input_text = request.form['text']
    start_time = time.time()
    result = transformer_api(input_text)
    end_time = time.time()

    duration = end_time - start_time  # Calculate duration
    return render_template('result.html', text=input_text, classification=result, time_taken=duration)

def transformer_api(text):
    # Make an API request with the text
    return "TRANSFORMER NOT IMPLEMENTED"

if __name__ == '__main__':
    app.run(debug=True)
