# app/blueprints/memory_bank.py
from flask import Blueprint, render_template, request, redirect, url_for, abort, flash, current_app
from app.logger import log
from ..models import db, Person, Conversation, TranscriptTurn
import os
from sqlalchemy import update
from werkzeug.utils import secure_filename

bp = Blueprint("memory_bank", __name__, template_folder="../templates")

ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename: str) -> bool:
    # Split once from the right; index 1 is always the extension when a dot exists
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "webp"}

# app/blueprints/memory_bank.py (function stays near the top with helpers)

def save_person_photo(file_storage, desired_stem: str = "") -> str:
    """
    Save uploaded image in static/people and return just the stored filename (e.g., 'ravi.png').
    Ensures uniqueness if a file with the same name already exists.
    """
    if not file_storage or file_storage.filename == "":
        return ""
    if not allowed_file(file_storage.filename):
        return ""

    # CORRECT indices: [1] for extension, [0] for base name
    ext = file_storage.filename.rsplit(".", 1)[1].lower()                      # [web:279]
    base = secure_filename(desired_stem or file_storage.filename.rsplit(".", 1)[0])  # [web:279]
    if not base:
        base = "photo"

    candidate = f"{base}.{ext}"
    folder = current_app.config.get("UPLOAD_FOLDER") or os.path.join(current_app.root_path, "static", "people")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, candidate)

    i = 1
    while os.path.exists(path):
        candidate = f"{base}-{i}.{ext}"
        path = os.path.join(folder, candidate)
        i += 1

    file_storage.save(path)                                                    # [web:279]
    return candidate


def photo_url_for_person(person: Person) -> str:
    """
    Build a display URL for a person's photo:
    1) If person.photo_filename exists, use static/people/<filename>
    2) Else try ID-based file like people/<id>.png|.jpg|.jpeg|.webp
    3) Else use default silhouette
    """
    if getattr(person, "photo_filename", None):
        return url_for("static", filename=f"people/{person.photo_filename}")  # [4]
    static_root = os.path.join(current_app.root_path, "static")
    for ext in ("png", "jpg", "jpeg", "webp"):
        rel = os.path.join("people", f"{person.id}.{ext}")
        if os.path.exists(os.path.join(static_root, rel)):
            return url_for("static", filename=f"people/{person.id}.{ext}")  # [4]
    return url_for("static", filename="people/default_silhouette.png")  # [4]

@bp.get("/")
def home():
    people = (
        Person.query
        .filter_by(is_unknown=False)
        .order_by(Person.display_name.asc())
        .limit(12)
        .all()
    )
    unknowns = (
        Person.query
        .filter_by(is_unknown=True)
        .order_by(Person.created_at.desc())
        .limit(6)
        .all()
    )

    people_cards = [{"person": p, "photo_url": photo_url_for_person(p)} for p in people]
    unknown_cards = [{"person": u, "photo_url": photo_url_for_person(u)} for u in unknowns]

    return render_template("memory_bank/home.html", people_cards=people_cards, unknown_cards=unknown_cards)

@bp.get("/person/<int:person_id>")
def person(person_id):
    person = Person.query.get_or_404(person_id)
    conversations = (
        Conversation.query
        .filter_by(person_id=person.id)
        .order_by(Conversation.started_at.desc())
        .limit(20)
        .all()
    )
    photo_url = photo_url_for_person(person)
    return render_template("memory_bank/person.html", person=person, conversations=conversations, photo_url=photo_url)

@bp.post("/person/<int:person_id>/update")
def person_update(person_id):
    p = Person.query.get_or_404(person_id)

    display_name = (request.form.get("display_name") or "").strip()
    relation = ((request.form.get("relation") or "").strip()) or None

    if not display_name:
        flash("Name is required.", "error")
        return redirect(url_for("memory_bank.person", person_id=p.id))

    # Optional photo change
    file = request.files.get("photo")
    if file and file.filename:
        desired_stem = secure_filename(display_name.lower().replace(" ", "_")) or f"person_{p.id}"
        stored_filename = save_person_photo(file, desired_stem)  # helper from earlier
        if stored_filename:
            p.photo_filename = stored_filename

    p.display_name = display_name
    p.relation = relation

    db.session.commit()
    flash("Profile updated.", "success")
    return redirect(url_for("memory_bank.person", person_id=p.id))

@bp.post("/person/<int:person_id>/delete")
def person_delete(person_id):
    p = Person.query.get_or_404(person_id)
    db.session.delete(p)               # cascades remove conversations (per model) [8][13]
    db.session.commit()
    flash("Person deleted.", "success")
    return redirect(url_for("memory_bank.home"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    temp_tag = request.args.get("temp_tag") or request.form.get("temp_tag")
    if request.method == "POST":
        display_name = (request.form.get("display_name") or "").strip()
        relation = ((request.form.get("relation") or "").strip()) or None

        if not display_name:
            flash("Name is required.", "error")
            return render_template("memory_bank/register.html", temp_tag=temp_tag)

        # Optional file upload
        file = request.files.get("photo")
        desired_stem = secure_filename(display_name.lower().replace(" ", "_")) or "person"
        stored_filename = save_person_photo(file, desired_stem) if file else ""

        # If temp_tag exists, convert that unknown into a real contact
        unknown = None
        if temp_tag:
            unknown = Person.query.filter_by(temp_tag=temp_tag, is_unknown=True).first()

        if unknown:
            unknown.display_name = display_name
            unknown.relation = relation
            if stored_filename:
                unknown.photo_filename = stored_filename
            unknown.is_unknown = False
            unknown.temp_tag = None
            db.session.commit()
            flash(f"Registered new contact: {unknown.display_name}", "success")
            return redirect(url_for("memory_bank.person", person_id=unknown.id))
        else:
            p = Person(
                display_name=display_name,
                relation=relation,
                photo_filename=(stored_filename or None),
                is_unknown=False,
                temp_tag=None,
            )
            db.session.add(p)
            db.session.commit()
            flash(f"Registered new contact: {p.display_name}", "success")
            return redirect(url_for("memory_bank.person", person_id=p.id))

    return render_template("memory_bank/register.html", temp_tag=temp_tag)

@bp.route("/unknowns", methods=["GET"])
def unknowns():
    rows = (
        Person.query
        .filter_by(is_unknown=True)
        .order_by(Person.created_at.desc())
        .all()
    )
    return render_template("memory_bank/unknowns.html", unknowns=rows)

@bp.get("/conversation/<int:conversation_id>")
def conversation(conversation_id):
    conv = Conversation.query.get_or_404(conversation_id)
    turns = conv.turns
    person = conv.person

    # The “person” here is the contact whose profile this conversation belongs to.
    # Treat that contact as the Visitor in UI, and the wearer/user as “User”.
    visitor_name = person.display_name if person else "Visitor"

    # Build a lightweight UI label map
    def ui_speaker_label(speaker: str) -> str:
        if speaker == "PATIENT":
            return "User"
        if speaker == "VISITOR":
            return visitor_name
        return speaker  # fallback
    
    photo_url = photo_url_for_person(person) if person else url_for("static", filename="people/default_silhouette.png")
    return render_template("memory_bank/conversation.html", 
                           conversation=conv, 
                           turns=turns, 
                           person=person, 
                           photo_url=photo_url, 
                           ui_speaker_label=ui_speaker_label,
    )

@bp.post("/conversation/<int:conversation_id>/delete")
def conversation_delete(conversation_id):
    conv = Conversation.query.get_or_404(conversation_id)
    person_id = conv.person_id
    db.session.delete(conv)            # delete one record [6]
    db.session.commit()
    flash("Conversation deleted.", "success")
    if person_id:
        return redirect(url_for("memory_bank.person", person_id=person_id))
    return redirect(url_for("memory_bank.home"))

@bp.post("/conversation/<int:conversation_id>/retry-summary")
def conversation_retry_summary(conversation_id):
    conv = Conversation.query.get_or_404(conversation_id)
    # Placeholder behavior: do nothing yet, just acknowledge.
    # In the future: enqueue a background task or call the summarization API here.
    flash("Retry requested. Summarization will run soon (placeholder).", "info")
    return redirect(url_for("memory_bank.conversation", conversation_id=conv.id))

@bp.get("/merge/<int:unknown_id>/pick")
def merge_pick(unknown_id):
    unknown = Person.query.get_or_404(unknown_id)
    if not unknown.is_unknown:
        flash("This profile is already a known contact.", "warning")
        return redirect(url_for("memory_bank.person", person_id=unknown.id))
    knowns = (
        Person.query
        .filter_by(is_unknown=False)
        .order_by(Person.display_name.asc())
        .all()
    )
    return render_template("memory_bank/merge_pick.html", unknown=unknown, knowns=knowns)

@bp.post("/merge/<int:unknown_id>/apply")
def merge_apply(unknown_id):
    unknown = Person.query.get_or_404(unknown_id)
    if not unknown.is_unknown:
        flash("This profile is already a known contact.", "warning")
        return redirect(url_for("memory_bank.person", person_id=unknown.id))

    try:
        known_id = int(request.form.get("known_id", "0"))
    except ValueError:
        known_id = 0
    if known_id <= 0 or known_id == unknown.id:
        flash("Select a valid known contact to merge into.", "error")
        return redirect(url_for("memory_bank.merge_pick", unknown_id=unknown.id))

    known = Person.query.get_or_404(known_id)
    if known.is_unknown:
        flash("Target contact must be a known contact.", "error")
        return redirect(url_for("memory_bank.merge_pick", unknown_id=unknown.id))

    keep_photo = bool(request.form.get("keep_photo"))

    db.session.execute(
        update(Conversation).where(Conversation.person_id == unknown.id).values(person_id=known.id)
    )

    if not keep_photo and unknown.photo_filename:
        known.photo_filename = unknown.photo_filename

    db.session.delete(unknown)
    db.session.commit()
    flash("Merged unknown profile into known contact successfully.", "success")
    return redirect(url_for("memory_bank.person", person_id=known.id))


