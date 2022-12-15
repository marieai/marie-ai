from app import create_app

app = create_app()

if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple(hostname='0.0.0.0', port=5100, application=app)
