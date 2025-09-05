from flask import Blueprint
from app.logger import log

bp = Blueprint("memory_bank", __name__)

@bp.route("/home")
def home():
    log.info("Memory Bank home route accessed")
    return "<h1>Memory Bank blueprint success!</h1>"
