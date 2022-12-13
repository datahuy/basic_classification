import logging
import sys
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
# from main.classifier.industry_classifier import IndustryClassifier
# from main.classifier.FMCG_classifier import FMCGClassifier
from main.classifier.FMCGl1_classifier import FMCGl1Classifier
import time
from main.utils.pipeline import merge_output
from main.rule.rule_mapping import read_json
from main.rule.rule_mapping import rule_predict_batch


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Define classification models
# industry_classifier = IndustryClassifier('model/industry_cls_best_1129.pt', batch_size=128)
# FMCG_classifier = FMCGClassifier('model/model_fmcg_binary_best_converted.pt', batch_size=128)
FMCG_l1_classifier = FMCGl1Classifier("output_model/l1/best.pt", batch_size=128)
keywords = read_json("main/rule/data-bin/keyword_lv1.json")


class Body(BaseModel):
    product_name: Union[str, list]
    #threshold_KP: int = 0.5
    #threshold_FMCG: int = 0.5
    batch_size: int = 128


app = FastAPI()


@app.get("/pd-industry-classification/")
def root():
    return {"message": "Product l1 Classification"}


@app.post("/pd-industry-classification/category-l2/")
def category_l2(body_params: Body):
    product_name = body_params.product_name
    batch_size = body_params.batch_size
    if type(product_name) == str:
        product_name = [product_name]
    try:
        start = time.time()
        output_model = FMCG_l1_classifier.predict(product_name, batch_size=batch_size)
        output_rule = rule_predict_batch(product_name, batch_size=batch_size, keywords=keywords)
        merged = merge_output(model_output=output_model, rule_output=output_rule)
        """result_KP = [{"product_name": n, "industry": [p], "score": [s]} for n, p, s in
                     zip(product_name, preds_KP, probs_KP)]

        for i in range(len(result_KP)):
            if 'fmcg' in result_FMCG[i]['industry']:
                if 'unknown' in result_KP[i]['industry']:  # Replace unknown with fmcg
                    result_KP[i]['industry'] = [result_FMCG[i]['industry']]
                    result_KP[i]['score'] = [result_FMCG[i]['score']]
                else:
                    result_KP[i]['industry'].append(result_FMCG[i]['industry'])
                    result_KP[i]['score'].append(result_FMCG[i]['score'])"""
        ret = {
            "data": merged,
            "status": "success",
            "status_code": 200,
            "message": "done"
        }
        product_name_for_logging = ", ".join(product_name[:10])
        product_name_for_logging += ", ..." if len(product_name) > 10 else ""
        logging.info(f"Prediction for [{product_name_for_logging}] took {(time.time() - start):.4f} seconds.")

        return ret
    except Exception as e:
        logging.error(e)
        return {
            "data": product_name,
            "status": "error",
            "status_code": 500,
            "message": f"Exception in : {e}"
        }


"""@app.post("/pd-industry-classification/industry_cls/")
def industry_cls(body_params: Body):
    product_name, threshold_KP, threshold_FMCG = body_params.product_name, body_params.threshold_KP, body_params.threshold_FMCG
    if type(product_name) == str:
        product_name = [product_name]
    try:
        start = time.time()
        # get prediction for 4 original KP classes
        preds_KP, probs_KP = industry_classifier.predict(product_name, threshold_KP)
        result_KP = [{"product_name": n, "industry": [p], "score": [s]} for n, p, s in
                     zip(product_name, preds_KP, probs_KP)]

        # get prediction for FMCG
        preds_FMCG, probs_FMCG = FMCG_classifier.predict(product_name, threshold_FMCG)
        result_FMCG = [{"product_name": n, "industry": p, "score": s} for n, p, s in
                       zip(product_name, preds_FMCG, probs_FMCG)]

        for i in range(len(result_KP)):
            if 'fmcg' in result_FMCG[i]['industry']:
                if 'unknown' in result_KP[i]['industry']:  # Replace unknown with fmcg
                    result_KP[i]['industry'] = [result_FMCG[i]['industry']]
                    result_KP[i]['score'] = [result_FMCG[i]['score']]
                else:
                    result_KP[i]['industry'].append(result_FMCG[i]['industry'])
                    result_KP[i]['score'].append(result_FMCG[i]['score'])

        ret = {
            "data": result_KP,
            "status": "success",
            "status_code": 200,
            "message": "Classify industry level done"
        }

        product_name_for_logging = ", ".join(product_name[:10])
        product_name_for_logging += ", ..." if len(product_name) > 10 else ""
        logging.info(f"Prediction for [{product_name_for_logging}] took {(time.time() - start):.4f} seconds.")

        return ret
    except Exception as e:
        logging.error(e)
        return {
            "data": product_name,
            "status": "error",
            "status_code": 500,
            "message": f"Exception in industry_cls: {e}"
        }"""
