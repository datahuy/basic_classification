
import logging
from datetime import date
import os
import sys
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from main.classifier.industry_classifier import IndustryClassifier
# import uvicorn

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# Define classification models
industry_classifier = IndustryClassifier('model/industry_cls_best_1129.pt', batch_size=128)

class Body(BaseModel):
    product_name: Union[str, list]
    threshold: int = 0.5

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/industry_cls/")
def industry_cls(body_params: Body):
    print(body_params)
    product_name, threshold = body_params.product_name, body_params.threshold
    if type(product_name) == str:
        product_name = [product_name]
    # try:
    preds, probs = industry_classifier.predict(product_name, threshold)
    result = [{"product_name": n, "industry": p, "score": str(s)} for n, p, s in zip(product_name, preds, probs)]
    return {
        "data": result,
        "status": "success",
        "status_code": 200,
        "message": "Classify industry level done"
    }
    # except Exception as e:
    #     logging.error(e)
    #     return {
    #         "data": product_name,
    #         "status": "error",
    #         "status_code": 500,
    #         "message": f"Exception in industry_cls: {e}"
    #     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9201)