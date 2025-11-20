from fastapi import FastAPI
from api.routes import router


# uv run -m api.main (from /Desktop/mlops_level3/ml_pipeline)
# curl 127.0.0.1:8000/ping

app = FastAPI(title="ML Pipeline API", version="1.0.0")

# Include routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)