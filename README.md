# ML-Flask-deployment-Spam-Detector
ML model to detect if a message is Spam or ham
This repository consists of files required for end to end implementation and deployment of Machine Learning Spam message Detection web application created with flask and deployed on the AWS/Heroku platform.

The dataset was sourced from INSAID
https://raw.githubusercontent.com/insaid2018/DeepLearning/master/e2e/spam.csv

This project has three major parts :

model.py - This contains the code for Machine Learning model to predict if a message is Spam or ham.
app.py - This contains Flask APIs that computes the precited value based on the model and returns it.
templates - This folder contains the HTML template to allow user to enter the message.


AWS deployment link :  http://ec2-54-160-214-56.compute-1.amazonaws.com:8080/

Heroku deployment link : https://spam-mail-detection-flask.herokuapp.com/

