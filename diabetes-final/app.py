from flask import Flask,render_template,request
import pickle
import numpy as np

logreg = pickle.load(open("logreg.pkl",'rb'))
linreg = pickle.load(open("regression.pkl",'rb'))


app = Flask(__name__)

@app.route("/")
def man():
    return render_template('main.html')

@app.route("/marks", methods=['POST', 'GET'])
def marks():
    return render_template('regression.html')

@app.route("/diabetes", methods=['POST', 'GET'])
def diabetes():
    return render_template('classification.html')

# {'course':3,'time':45}

@app.route("/regression.html",methods = ['POST','GET'])
def linreg_pred():
    courses = request.form['course']
    times = request.form['time']
    arr1 = np.array([[courses,times]])
    pred1 = linreg.predict(arr1)
    output_value = round(pred1[0])
    return render_template('regression.html',output = 'ESTIMATED MARKS WOULD BE {} / 60'.format(output_value))


@app.route('/classification.html', methods=['POST','GET'])
def logreg_pred():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = logreg.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == '__main__':
    app.run(debug=True)