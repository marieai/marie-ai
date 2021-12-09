# Marie-ICR
Integrate AI-powered OCR features into your applications


## Installation 

create folder structure

```sh 
mkdir models config
```

Follow instructions from `pytorch` website

```sh
https://pytorch.org/get-started/locally/
```

Install required packages with `pip`

```sh
$ pip install -r ./requirements/requirements.txt
```

Build Docker Image

```sh
DOCKER_BUILDKIT=1 docker build .
```

Starting in Development mode

```sh
python ./app.py
```

Starting in Production mode with `gunicorn`. Config settings [https://docs.gunicorn.org/en/stable/settings.html#settings]

```sh
gunicorn -c gunicorn.conf.py wsgi:app  --log-level=debug
```


## References

https://www.toptal.com/flask/flask-production-recipes
https://apispec.readthedocs.io/en/latest/install.html
https://github.com/gregbugaj/form-processor
https://gradio.app/