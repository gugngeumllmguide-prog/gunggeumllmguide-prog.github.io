from flask import Flask, render_template, request, redirect

app = Flask(__name__)
todos = []

@app.route('/')
def index():
    return render_template('index.html', todos=todos)

