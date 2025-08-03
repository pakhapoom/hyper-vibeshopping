FROM python:3.11-slim 

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY data/db/cust_info.xlsx ./data/db/cust_info.xlsx
COPY data/embedded_data.json ./data/embedded_data.json
RUN rm -rf ./chroma
COPY utils/set_up_vector_db.py ./utils/set_up_vector_db.py
COPY uploads ./uploads

RUN uv sync
RUN uv run utils/set_up_vector_db.py

COPY . . 

CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "3000"]