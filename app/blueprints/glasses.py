from flask import Blueprint
from app.logger import log

bp = Blueprint("glasses", __name__)

@bp.route("/home")
def home():
    log.info("Glasses home route accessed")
    return "<h1>Glasses blueprint success!</h1>"
