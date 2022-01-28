import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    print(features)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)
    classes = list(model.classes_)
    idx = classes.index(prediction)

    ans = {
        'prediction': prediction[0],
        'probability': model.predict_proba(features)[0][idx].round(2)
    }

    return render_template(
        "index.html",
        prediction="Prediction: {}".format(ans['prediction']),
        probability="Probability: {}".format(ans['probability'])
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0')
