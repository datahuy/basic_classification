from flask import Flask, request
from waitress import serve
import sys
import logging
from datetime import date
import os
from main.classifier.industry_classifier import IndustryClassifier

if not os.path.exists('log'):
    os.mkdir('log')

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./log/product_classification' + str(date.today()) + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

app = Flask(__name__)

# Define classification models
industry_classifier = IndustryClassifier('output_model/best.pt', threshold=0.5)


@app.route('/industry_cls', methods=['POST'])
def industry_cls():
    try:
        product_name = request.form.get('product_name', type=str)
        print(product_name)
        preds, probs = industry_classifier.predict(product_name)
        result = [{"product_name": n, "industry": p, "score": str(s)} for n, p, s in zip(product_name, preds, probs)]

        return {
            "data": result,
            "status": "success",
            "status_code": 200,
            "message": "Classify industry level done"
        }, 200

    except Exception as ex:
        logging.error(f'Exception in industry_cls: {ex}')
        return {
            "data": [],
            "status": "error",
            "status_code": 500,
            "message": f"Exception in industry_cls: {ex}"
        }, 500

if __name__ == "__main__":
    app.debug = True
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 9201))
    serve(app, host=host, port=port, threads=24)

