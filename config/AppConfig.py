import os

from fastapi import HTTPException

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
if OPENAI_API_KEY == "<insert your OPENAI API KEY here>":
    raise HTTPException(status_code=400, detail="Both OPENAI API KEY and ORG ID must be set to run this extension")

OPENAI_API_ORG_ID = os.environ.get('OPENAI_API_ORG_ID', None)
if OPENAI_API_ORG_ID == "<insert your OPENAI API ORG ID here>":
    raise HTTPException(status_code=400, detail="Both OPENAI API KEY and ORG ID must be set to run this extension")
