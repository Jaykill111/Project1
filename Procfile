release: pip install -r api/requirements.txt
web: cd api && gunicorn app:app --bind 0.0.0.0:$PORT
