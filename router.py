
import logging
import sys
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from main.classifier.industry_classifier import IndustryClassifier
from main.classifier.FMCG_classifier import FMCGClassifier

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# Define classification models
industry_classifier = IndustryClassifier('model/industry_cls_best_1129.pt', batch_size=128)
FMCG_classifier = FMCGClassifier('model/model_fmcg_binary_best_converted.pt', batch_size=128)

class Body(BaseModel):
    product_name: Union[str, list]
    threshold_KP: int = 0.5
    threshold_FMCG: int = 0.5

app = FastAPI()

@app.get("/pd-industry-classification/")
def root():
    return {"message": "Product Classification"}
    

@app.post("/pd-industry-classification/industry_cls/")
def industry_cls(body_params: Body):
    product_name, threshold_KP, threshold_FMCG = body_params.product_name, body_params.threshold_KP, body_params.threshold_FMCG
    if type(product_name) == str:
        product_name = [product_name]
    try:
        # get prediction for 4 original KP classes
        preds_KP, probs_KP = industry_classifier.predict(product_name, threshold_KP)
        result_KP = [{"product_name": n, "industry": [p], "score": [s]} for n, p, s in zip(product_name, preds_KP, probs_KP)]

        # get prediction for FMCG
        preds_FMCG, probs_FMCG = FMCG_classifier.predict(product_name, threshold_FMCG)
        result_FMCG = [{"product_name": n, "industry": p, "score": s} for n, p, s in zip(product_name, preds_FMCG, probs_FMCG)]
        
        for i in range(len(result_KP)):
            if 'fmcg' in result_FMCG[i]['industry']:
                if 'unknown' in result_KP[i]['industry']: # Replace unknown with fmcg
                    result_KP[i]['industry'] = [result_FMCG[i]['industry']]
                    result_KP[i]['score'] = [result_FMCG[i]['score']]   
                else:
                    result_KP[i]['industry'].append(result_FMCG[i]['industry'])
                    result_KP[i]['score'].append(result_FMCG[i]['score'])
        return {
            "data": result_KP,
            "status": "success",
            "status_code": 200,
            "message": "Classify industry level done"
        }
    except Exception as e:
        logging.error(e)
        return {
            "data": product_name,
            "status": "error",
            "status_code": 500,
            "message": f"Exception in industry_cls: {e}"
        }
