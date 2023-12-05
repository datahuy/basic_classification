import logging
import sys
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from main.classifier.industry_classifier import IndustryClassifier
from main.classifier.FMCG_classifier import FMCGClassifier
from main.classifier.FMCG_l2_classifier import FMCGl2Classifier
import time

logging.basicConfig(
    level=logging.INFO,
    # level=logging.WARNING,
    format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Define classification models
industry_classifier = IndustryClassifier('model/industry_cls_best_1129.pt', batch_size=128)
FMCG_classifier = FMCGClassifier('model/model_fmcg_binary_best_converted.pt', batch_size=128)
FMCG_l2_classifier = FMCGl2Classifier(
    l2_model_path='model/industry_cls_l2.pt', 
    l1_model_path='model/industry_cls_l1.pt',
    l2_rule_json_path='main/rule/data-bin/keyword_lv2.json',
    l1_normal_rule_json_path='main/rule/data-bin/keyword_lv1.json',
    l1_priority_rule_json_path='main/rule/data-bin/keyword_lv1_priority.json',
    l2_category_json_path='main/rule/data-bin/self_category_lv2.json',
    l1_category_json_path='main/rule/data-bin/self_category_lv1.json',
    l1l2_mapping_json_path='main/rule/data-bin/l1l2_mapping.json',
    batch_size=128,
    default_class='Không xác định',

    )

class Level2Body(BaseModel):
    product_name: Union[str, list]
    self_category: Union[str, list] = None
    model_l2_threshold: int = 0.99
    model_l1_threshold: int = 0.99

class Level0Body(BaseModel):
    product_name: Union[str, list]
    threshold_KP: int = 0.5
    threshold_FMCG: int = 0.5

app = FastAPI()


@app.get("/pd-industry-classification/")
def root():
    return {"message": "Welcome to the PD Industry Classification API."}

@app.post("/pd-industry-classification/fmcg_sub_cat_cls/")
def sub_cat_cls(body_params: Level2Body):
    product_name = body_params.product_name
    self_category = body_params.self_category
    if type(product_name) == str:
        product_name = [product_name]
    if type(self_category) == str:
        self_category = [self_category]

    # ensure that product_name and self_category have the same length
    if self_category is not None and len(product_name) != len(self_category):
        return {
            "data": (product_name, self_category),
            "status": "error",
            "status_code": 500,
            "message": f"Length of product_name and self_category must be the same"
        }
    
    try:
        start = time.time()
        if not self_category:
            self_category = ['' for _ in range(len(product_name))]
        level1, level2, method = FMCG_l2_classifier.predict(name_input=product_name,
                                            self_category_input=self_category,
                                            model_l2_threshold=body_params.model_l2_threshold,
                                            model_l1_threshold=body_params.model_l1_threshold)
        ret = {
            "data": {
                "product_name": product_name,
                "self_category": self_category,
                "level1": level1,
                "level2": level2,
                "method": method
            },
            "status": "success",
            "status_code": 200,
            "message": "done"
        }
        product_name_for_logging = ", ".join(product_name[:10])
        product_name_for_logging += ", ..." if len(product_name) > 10 else ""
        len_product_name = len(product_name)
        logging.info(f"Prediction for [{product_name_for_logging}] - {len_product_name} products took {(time.time() - start):.4f} seconds.")

        return ret
    except Exception as e:
        logging.error(e)
        return {
            "data": (product_name, self_category),
            "status": "error",
            "status_code": 500,
            "message": f"Exception in : {e}"
        }


@app.post("/pd-industry-classification/industry_cls/")
def industry_cls(body_params: Level0Body):
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
        }