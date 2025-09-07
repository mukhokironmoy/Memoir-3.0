from flask import request  # add this
import json                # add this
from flask import Blueprint, render_template, jsonify, abort
from flask import url_for, current_app
import os
from sqlalchemy import func
from app.models import Person, Conversation, Embedding  # add Embedding

from flask import request, jsonify
from datetime import datetime
from app import db
from app.models import Conversation, TranscriptTurn


from app import db


from app.logger import log

bp = Blueprint("glasses", __name__, template_folder="../templates")

def photo_url_for_person(person: Person) -> str:
    """
    Build a display URL for a person's photo:
    1) If person.photo_filename exists, use static/people/<filename>
    2) Else try ID-based file like people/<id>.(png|jpg|jpeg|webp)
    3) Else use default silhouette
    """
    if getattr(person, "photo_filename", None):
        return url_for("static", filename=f"people/{person.photo_filename}")

    static_root = os.path.join(current_app.root_path, "static")
    for ext in ("png", "jpg", "jpeg", "webp"):
        rel = os.path.join("people", f"{person.id}.{ext}")
        if os.path.exists(os.path.join(static_root, rel)):
            return url_for("static", filename=f"people/{person.id}.{ext}")

    return url_for("static", filename="people/default_silhouette.png")



@bp.route("/")
def home():
    log.info("Glasses home route accessed")
    return render_template("glasses_base.html")

@bp.route("/memoir")
def sidebar_home():
    log.info("Memoir mode home sidebar")
    return render_template("glasses/sidebar/home.html")

@bp.get("/api/people")
def api_people():
    """
    Returns a list of people with their latest 'last_met_at' (latest conversation start time).
    """
    # subquery: latest conversation started_at per person
    last_met_subq = (
        db.session.query(
            Conversation.person_id.label("pid"),
            func.max(Conversation.started_at).label("last_met_at"),
        )
        .group_by(Conversation.person_id)
        .subquery()
    )

    # Pull Person objects + last_met_at so we can build photo URLs
    rows = (
        db.session.query(Person, last_met_subq.c.last_met_at)
        .outerjoin(last_met_subq, last_met_subq.c.pid == Person.id)
        .order_by(Person.display_name.asc())
        .all()
    )

    out = []
    for person, last_met_at in rows:
        out.append({
            "id": person.id,
            "display_name": person.display_name,
            "relation": person.relation,
            "photo_url": photo_url_for_person(person),  # ✅ real URL
            "is_unknown": bool(person.is_unknown),
            "last_summary_cached": person.last_summary_cached,
            "last_met_at": last_met_at.isoformat() if last_met_at else None,
        })
    return jsonify(out)


@bp.get("/api/people/<int:person_id>")
def api_person(person_id: int):
    """
    Returns a single person's profile data with:
    - last_met_at (latest conversation started_at)
    - latest_conversation (id, started_at, ended_at, has_summary)
    - last_summary_cached (bullets text, if any)
    """
    # base person
    person: Person | None = db.session.get(Person, person_id)
    if not person:
        abort(404, description="Person not found")

    # latest conversation for this person
    latest_conv: Conversation | None = (
        db.session.query(Conversation)
        .filter(Conversation.person_id == person.id)
        .order_by(Conversation.started_at.desc())
        .first()
    )

    last_met_at = latest_conv.started_at if latest_conv else None
    latest_conv_obj = None
    if latest_conv:
        latest_conv_obj = {
            "id": latest_conv.id,
            "started_at": latest_conv.started_at.isoformat(),
            "ended_at": latest_conv.ended_at.isoformat() if latest_conv.ended_at else None,
            "has_summary": bool(latest_conv.summary),
        }

    return jsonify({
        "id": person.id,
        "display_name": person.display_name,
        "relation": person.relation,
        "photo_url": photo_url_for_person(person),  # ✅ real URL
        "is_unknown": bool(person.is_unknown),
        "last_summary_cached": person.last_summary_cached,  # may be None
        "last_met_at": last_met_at.isoformat() if last_met_at else None,
        "latest_conversation": latest_conv_obj,             # may be None
    })


def _bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code

@bp.post("/api/face/enroll")
def api_face_enroll():
    """
    Save/replace one face embedding vector for a person (provider='local').
    Body JSON:
      {
        "person_id": 123,
        "vector": [ ... 128 floats ... ],
        "provider": "local"   # optional; defaults to "local"
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    person_id = data.get("person_id")
    vector = data.get("vector")
    provider = data.get("provider", "local")

    if not isinstance(person_id, int):
        return _bad("person_id_required")
    if not isinstance(vector, list) or not vector:
        return _bad("vector_required")

    person = db.session.get(Person, person_id)
    if not person:
        return _bad("person_not_found", 404)

    # upsert one embedding per (person, provider)
    emb = (
        db.session.query(Embedding)
        .filter(Embedding.person_id == person_id, Embedding.provider == provider)
        .first()
    )
    vec_json = json.dumps(vector)
    if emb:
        emb.vector_json = vec_json
        emb.dim = len(vector)
    else:
        emb = Embedding(
            person_id=person_id,
            provider=provider,
            vector_json=vec_json,
            dim=len(vector),
        )
        db.session.add(emb)

    db.session.commit()
    return jsonify({"ok": True, "embedding_id": emb.id})


import math

def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@bp.post("/api/face/recognize")
def api_face_recognize():
    data = request.get_json(force=True, silent=True) or {}
    vector = data.get("vector")
    provider = data.get("provider", "local")
    if not isinstance(vector, list) or not vector:
        return jsonify({"ok": False, "error": "vector_required"}), 400

    rows = db.session.query(Embedding).filter(Embedding.provider == provider).all()
    if not rows:
        return jsonify({"ok": True, "match": False, "reason": "no_enrollments"})

    best, best_d = None, 999.0
    for r in rows:
        cand = json.loads(r.vector_json)
        d = l2(vector, cand)
        if d < best_d:
            best_d, best = d, r

    THRESH = float(data.get("threshold", 0.58))
    if best and best_d <= THRESH:
        person = db.session.get(Person, best.person_id)
        return jsonify({
            "ok": True,
            "match": True,
            "distance": round(best_d, 4),
            "person": {
                "id": person.id,
                "display_name": person.display_name,
                "relation": person.relation,
                "photo_url": photo_url_for_person(person),  # ✅ real URL
            }
        })

    else:
        return jsonify({"ok": True, "match": False, "distance": round(best_d, 4)})


@bp.post("/api/unknown/ensure")
def api_unknown_ensure():
    """
    Return the most recently created unknown profile, or create a fresh one if none exist.
    Avoids hardcoding an UNKNOWN_ID.
    """
    # newest unknown if any
    unk = (
        db.session.query(Person)
        .filter(Person.is_unknown == True)
        .order_by(Person.created_at.desc())
        .first()
    )

    if not unk:
        # uses the model helper from your codebase
        unk = Person.make_unknown()
        db.session.add(unk)
        db.session.commit()


    return jsonify({
        "id": unk.id,
        "display_name": unk.display_name,
        "relation": unk.relation,
        "photo_url": photo_url_for_person(unk),
        "is_unknown": True,
    })

@bp.post("/api/conversations/start")
def api_start_conversation():
    data = request.get_json() or {}
    person_id = data.get("person_id")
    stt_lang = (data.get("stt_lang") or "en-IN").strip()

    conv = Conversation(
        person_id=person_id,
        source="glasses",
        stt_provider="web_speech",
        stt_lang=stt_lang,
    )
    db.session.add(conv)
    db.session.commit()
    return jsonify({"ok": True, "conversation_id": conv.id})


@bp.post("/api/conversations/pause")
def api_pause_conversation():
    # No DB change required; pausing is handled client-side.
    data = request.get_json() or {}
    conv_id = data.get("conversation_id")
    if not db.session.get(Conversation, conv_id):
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True})


@bp.post("/api/conversations/resume")
def api_resume_conversation():
    # No DB change required; resuming is handled client-side.
    data = request.get_json() or {}
    conv_id = data.get("conversation_id")
    if not db.session.get(Conversation, conv_id):
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True})


@bp.post("/api/conversations/stop")
def api_stop_conversation():
    data = request.get_json() or {}
    conv_id = data.get("conversation_id")
    conv = db.session.get(Conversation, conv_id)
    if not conv:
        return jsonify({"ok": False, "error": "not_found"}), 404
    conv.ended_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"ok": True})


def _speaker_to_enum(name):
    return "PATIENT" if (name or "").upper().startswith("P") else "VISITOR"


@bp.post("/api/turns/append")
def api_append_turn():
    """
    Called whenever a line is finalized (e.g., when the user flips the speaker
    toggle or when you intentionally commit the buffered text).
    """
    data = request.get_json() or {}
    conv_id = data.get("conversation_id")
    text    = (data.get("text") or "").strip()
    speaker = _speaker_to_enum(data.get("speaker"))
    conf    = data.get("confidence")
    lang    = data.get("lang")

    if not conv_id or not text:
        return jsonify({"ok": False, "error": "missing_fields"}), 400

    conv = db.session.get(Conversation, conv_id)
    if not conv:
        return jsonify({"ok": False, "error": "bad_conversation"}), 404

    turn = TranscriptTurn(
        conversation_id=conv.id,
        speaker=speaker,
        text=text,
        confidence=conf,
        lang=lang,
    )
    db.session.add(turn)
    db.session.commit()
    return jsonify({"ok": True, "turn_id": turn.id})



# @bp.route("/memoir/info")
# def sidebar_contacts():
#     return render_template("glasses/sidebar/view_contacts.html")  # placeholder for Info

# @bp.route("/memoir/profile/<int:person_id>")
# def sidebar_profile(person_id):
#     # later: pull person data from DB
#     return render_template("glasses/sidebar/profile.html", person_id=person_id)

# @bp.route("/memoir/conversation/<int:person_id>")
# def sidebar_conversation(person_id):
#     # later: pull active convo from DB
#     return render_template("glasses/sidebar/conversation.html", person_id=person_id)
