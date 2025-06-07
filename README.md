make virtual environment
```
.\venv_rag_with_kube\Scripts\activate
 ```

install poetry 
```
pip install poetry 
```

start application

```bash
poetry run uvicorn app.api.main:app --reload
````