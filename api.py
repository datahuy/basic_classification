import os
import sys
import logging
from datetime import date

from waitress import serve
from flask import Flask, request

from infer import infer

if not os.path.exists('api_log'):
    os.mkdir('api_log')

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./api_log/cls_fashion_lv1_' + str(date.today()) + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


app = Flask(__name__)
    
@app.route('/classification_product_lv1', methods=['POST'])
def predict():
    try:
        sentences = request.form.getlist('product_name')
        result = infer(
            sentences=sentences,
            return_probs=True,
        )
        result = [{"industry": industry, "score": str(score)} for (industry, score) in result]
        print(result)
        return {
            "status": "success",
            "status_code": 200,
            "message": "Infer industry name done",
            "data": result
        }, 200


    except Exception as ex:
        logging.error(f'Exception in classification_product_lv1: {ex}')
        result = []
        return {
            "status": "error",
            "status_code": 500,
            "message": "Exception in inferring product api",
            "data": result
        }, 500

if __name__ == "__main__":
    app.debug = True
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 9201))
    serve(app, host=host, port=port, threads=24)

