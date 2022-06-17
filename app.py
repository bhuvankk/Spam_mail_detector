import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
cv= pickle.load(open('cv-transform.pkl','rb'))
model = pickle.load(open('spam-sms-mnb-model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    entered_text = request.form.get("entered_text")
    print(entered_text)
    message=[entered_text]
    cv_out=cv.transform(message).toarray()
    print("yes transform")
    prediction = model.predict(cv_out)

    output = prediction[0]

    return render_template('index.html', prediction_text='This is a {} message'.format(output))



if __name__ == "__main__":
    app.run(debug=True)