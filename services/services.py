import uvicorn as uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from loguru import logger

from business.zero_shot_clf import multiclass_zero_shot_pred, multilabel_zero_shot_pred
from utils.decorator_fastapi import ExtractValueArgsFastapi
from skllm.config import SKLLMConfig


app = FastAPI()




# class UnicornException(Exception):
#     def __init__(self, name: str):
#         self.name = name
#
# @app.exception_handler(UnicornException)
# async def unicorn_exception_handler(request: Request, exc: UnicornException):
#     return JSONResponse(
#         status_code=400,
#         content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
#     )

@app.post("/zero_shot")
@ExtractValueArgsFastapi(file=False)
def zero_shot(value, args):
    key = "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"
    name = "lvt_key"
    SKLLMConfig.set_openai_key(key)
    SKLLMConfig.set_openai_org(name)
    logger.debug(f"args::: {args}")
    multilabel = args.get("multilabel", False)
    model = args.get("model", "gpt-3.5-turbo")
    labels = args.get("labels", None)
    logger.debug(f"labels: {labels}")
    labels = labels.split(",")
    logger.debug(f"labels list preprocessed: {labels}")
    if labels is None or len(labels) < 2:
        #todo: cambiare eccezione perche' non viene vista da loko
        logger.debug(f"too few labels. Labels value: {labels}")
        raise HTTPException(status_code=400, detail="Not enough labels were specified")
    if multilabel:
        max_labels = int(args.get("max_labels", 3))
        logger.debug(f"multilabel clf... max_labels {max_labels}, model to use {model},  labels {labels}")
        pred = multilabel_zero_shot_pred(data=value, model=model, labels=labels, max_labels=max_labels)

    else:
        logger.debug(f"multiclass clf... model to use {model},  labels {labels}")
        pred = multiclass_zero_shot_pred(data=value, model=model, labels=labels)
    res = {value[i]:pred[i] for i in range(len(value))}
    logger.debug(f"results: {res}")
    return JSONResponse(content=res)


#
# @app.route("/files", methods=["POST"])
# def test2():
#     file = request.files['file']
#     fname = file.filename
#     print("You have uploaded a file called:",fname)
#     return jsonify(dict(msg=f"Hello extensions, you have uploaded the file: {fname}!"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
