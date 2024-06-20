from fastapi import FastAPI
import os
from dotenv import load_dotenv
from .service_route import service_router

app = FastAPI()

app.include_router(prefix='', router=service_router)


# if __name__ == "__main__":
#     load_dotenv("./.env")
#     print(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)
