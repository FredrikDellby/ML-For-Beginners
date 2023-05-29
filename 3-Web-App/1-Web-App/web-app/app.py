from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

#model = pickle.load(open("../ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")

def train_model():
    ufos = pd.read_csv('../data/ufos.csv')
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    ufos.dropna(inplace=True)
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    selected_features = ['Seconds','Latitude','Longitude']
    X = ufos[selected_features]
    y = ufos['Country']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    model_filename = 'ufo-model.pkl'
    pickle.dump(model, open(model_filename,'wb'))
    return model

@app.route("/predict", methods=["POST"])
def predict():
    if Path("ufo-model.pkl").is_file():
        model = pickle.load(open("ufo-model.pkl", "rb"))
    else:
        model = train_model()
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)