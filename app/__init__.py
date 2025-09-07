import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from .logger import log   # <-- import our logger

db = SQLAlchemy()
migrate = Migrate()

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # ensure instance/ exists
    os.makedirs(app.instance_path, exist_ok=True)

    # config
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret"),
        SQLALCHEMY_DATABASE_URI="sqlite:///" + os.path.join(app.instance_path, "memoir.db"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "static", "people")
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
    
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


    if test_config:
        app.config.update(test_config)

    db.init_app(app)
    migrate.init_app(app, db)

    from . import models  # noqa

    from .blueprints.glasses import bp as glasses_bp
    from .blueprints.memory_bank import bp as memory_bank_bp
    app.register_blueprint(glasses_bp, url_prefix="/glasses")
    app.register_blueprint(memory_bank_bp, url_prefix="/memory_bank")

    # Example log lines
    @app.before_request
    def log_request():
        log.info("Incoming request")

    @app.after_request
    def log_response(response):
        log.info(f"Response status: {response.status}")
        return response

    @app.get("/ping")
    def ping():
        log.debug("Ping route hit")
        return {"status": "ok"}

    log.info("Flask app created successfully")
    return app
