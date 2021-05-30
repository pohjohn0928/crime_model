from flask import Flask, request, render_template
from bert_model import AlbertModel

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def home():
    return render_template("crime_model.html")

@app.route('/predictCrime', methods=["POST"])
def predictCrime():
    fact = request.values['fact']
    albert = AlbertModel()
    classes,prob = albert.predict([fact])
    dic = {'classes':classes.tolist(),'prob':prob[0].tolist()}
    return dic

if __name__ == '__main__':
    app.run(debug=True)