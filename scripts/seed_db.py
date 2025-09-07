# scripts/seed_db.py
"""
Hard-reset seeder for Memoir.

- Deletes the SQLite DB file at instance/memoir.db (if it exists)
- Re-creates all tables
- Inserts a coherent seed dataset (People → Conversations → TranscriptTurns → Embeddings)
"""

import os
import json
from datetime import datetime, timedelta

# --- Project imports (based on your app structure) ---
from app import create_app, db
from app.models import Person, Conversation, TranscriptTurn, Embedding


def dt(y, m, d, hh, mm, ss=0):
    """Naive datetime builder (UTC semantics for your DB)."""
    return datetime(y, m, d, hh, mm, ss)


def reset_database_file(app):
    """Delete the SQLite DB file at instance/memoir.db to guarantee a clean slate."""
    os.makedirs(app.instance_path, exist_ok=True)
    db_path = os.path.join(app.instance_path, "memoir.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing DB: {db_path}")
    else:
        print(f"No existing DB found at: {db_path}")
    return db_path


def seed_data():
    """Insert a realistic, workflow-aligned dataset."""
    # ---------------- People ----------------
    mom = Person(
        display_name="Mom",
        relation="Mother",
        photo_filename="mom.png",
        is_unknown=False,
        temp_tag=None,
        notes="Prefers morning visits; likes short walks after breakfast.",
        last_summary_cached=None,
    )

    dr = Person(
        display_name="Dr. Mehra",
        relation="Doctor",
        photo_filename="dr_mehra.png",
        is_unknown=False,
        temp_tag=None,
        notes="Family physician; clinic Tue/Thu evenings.",
        last_summary_cached=None,
    )

    unknown_live = Person(
        display_name="unknown_ab12cd34",
        relation=None,
        photo_filename="ab12cd34.png",  # snapshot frame saved by overlay
        is_unknown=True,
        temp_tag="unknown_ab12cd34",
        notes=None,
        last_summary_cached=None,
    )

    ravi = Person(
        display_name="Ravi Kumar",
        relation="Neighbor",
        photo_filename="ravi.png",
        is_unknown=False,      # registered now
        temp_tag=None,         # temp tag cleared on registration
        notes="Lives across the hall; meets on evening walks.",
        last_summary_cached=None,
    )

    db.session.add_all([mom, dr, unknown_live, ravi])
    db.session.flush()

    # ---------------- Conversations ----------------
    # A: Mom (recognized → convo → summary success)
    c1 = Conversation(
        person_id=mom.id,
        started_at=dt(2025, 9, 3, 10, 5, 0),
        ended_at=dt(2025, 9, 3, 10, 12, 15),
        summary=(
            "Morning walk plan for 6:30 AM tomorrow\n"
            "Reminder: take BP meds at 8 PM\n"
            "Discussed weekend visit by Ria"
        ),
        face_snapshot_path=None,
        source="glasses",
        stt_provider="web_speech",
        stt_lang="en-IN",
    )
    db.session.add(c1)
    db.session.flush()

    base = c1.started_at
    db.session.add_all(
        [
            TranscriptTurn(
                conversation_id=c1.id,
                speaker="VISITOR",
                text="Good morning! How are you feeling today?",
                timestamp=base + timedelta(seconds=15),
                confidence=0.95,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c1.id,
                speaker="PATIENT",
                text="I’m good. I slept well. Shall we go for a short walk later?",
                timestamp=base + timedelta(seconds=28),
                confidence=0.92,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c1.id,
                speaker="VISITOR",
                text="Yes, let’s do 6:30 AM tomorrow. Also, take your BP tablets at 8 PM.",
                timestamp=base + timedelta(minutes=2, seconds=5),
                confidence=0.93,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c1.id,
                speaker="PATIENT",
                text="Alright, I’ll set a reminder.",
                timestamp=base + timedelta(minutes=2, seconds=20),
                confidence=0.94,
                lang="en-IN",
            ),
        ]
    )

    # B: Doctor (recognized → quick check-in → summary success)
    c2 = Conversation(
        person_id=dr.id,
        started_at=dt(2025, 9, 2, 17, 30, 0),
        ended_at=dt(2025, 9, 2, 17, 40, 0),
        summary=(
            "BP readings stable over the week\n"
            "Continue current medication\n"
            "Book follow-up in two weeks"
        ),
        face_snapshot_path=None,
        source="glasses",
        stt_provider="web_speech",
        stt_lang="en-IN",
    )
    db.session.add(c2)
    db.session.flush()

    db.session.add_all(
        [
            TranscriptTurn(
                conversation_id=c2.id,
                speaker="VISITOR",
                text="Doctor, BP readings were mostly within range this week.",
                timestamp=c2.started_at + timedelta(minutes=1),
                confidence=0.93,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c2.id,
                speaker="PATIENT",
                text="I felt fine. No dizziness.",
                timestamp=c2.started_at + timedelta(minutes=2, seconds=10),
                confidence=0.9,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c2.id,
                speaker="VISITOR",
                text="Great, let’s keep the same dosage and review in two weeks.",
                timestamp=c2.started_at + timedelta(minutes=8, seconds=45),
                confidence=0.94,
                lang="en-IN",
            ),
        ]
    )

    # C: Unknown (still unknown), summary failed → Retry path in UI
    c3 = Conversation(
        person_id=unknown_live.id,  # linked to unknown person
        started_at=dt(2025, 9, 4, 9, 10, 0),
        ended_at=dt(2025, 9, 4, 9, 12, 30),
        summary=None,  # failed summarization
        face_snapshot_path="/static/snapshots/ab12cd34.jpg",
        source="glasses",
        stt_provider="web_speech",
        stt_lang="en-IN",
    )
    db.session.add(c3)
    db.session.flush()

    db.session.add_all(
        [
            TranscriptTurn(
                conversation_id=c3.id,
                speaker="VISITOR",
                text="Hello! Are you visiting Ravi’s flat?",
                timestamp=c3.started_at + timedelta(seconds=20),
                confidence=0.88,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c3.id,
                speaker="PATIENT",
                text="Yes, just dropping off a package.",
                timestamp=c3.started_at + timedelta(seconds=35),
                confidence=0.86,
                lang="en-IN",
            ),
        ]
    )

    # D: Was unknown during recording; now registered as Ravi
    c4 = Conversation(
        person_id=ravi.id,
        started_at=dt(2025, 9, 1, 18, 0, 0),
        ended_at=dt(2025, 9, 1, 18, 6, 30),
        summary=(
            "Discussed society meeting on Sunday\n"
            "Shared maintenance notice details\n"
            "Will bring forms tomorrow"
        ),
        face_snapshot_path="/static/snapshots/9f81c2ab.jpg",  # frame from the unknown moment
        source="glasses",
        stt_provider="web_speech",
        stt_lang="en-IN",
    )
    db.session.add(c4)
    db.session.flush()

    db.session.add_all(
        [
            TranscriptTurn(
                conversation_id=c4.id,
                speaker="VISITOR",
                text="There’s a society meeting this Sunday at 11 AM.",
                timestamp=c4.started_at + timedelta(seconds=50),
                confidence=0.91,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c4.id,
                speaker="PATIENT",
                text="Okay, where do we meet?",
                timestamp=c4.started_at + timedelta(minutes=1, seconds=20),
                confidence=0.9,
                lang="en-IN",
            ),
            TranscriptTurn(
                conversation_id=c4.id,
                speaker="VISITOR",
                text="Community hall on the ground floor. I’ll bring the forms tomorrow.",
                timestamp=c4.started_at + timedelta(minutes=4, seconds=5),
                confidence=0.93,
                lang="en-IN",
            ),
        ]
    )

    # ---------------- Embeddings (future-proof; minimal stubs) ----------------
    stub_vec = [0.12, -0.03, 0.41, 0.08, -0.22, 0.05, 0.31, -0.11]
    db.session.add_all(
        [
            Embedding(
                person_id=mom.id,
                provider="local_stub",
                vector_json=json.dumps(stub_vec),
                dim=len(stub_vec),
            ),
            Embedding(
                person_id=dr.id,
                provider="local_stub",
                vector_json=json.dumps(stub_vec[::-1]),  # slight variation
                dim=len(stub_vec),
            ),
        ]
    )

    # ---------------- Update cached summaries ----------------
    def set_cached_summary(person: Person):
        conv = (
            db.session.query(Conversation)
            .filter(Conversation.person_id == person.id)
            .order_by(Conversation.started_at.desc())
            .first()
        )
        if conv and conv.summary:
            person.last_summary_cached = conv.summary

    for p in (mom, dr, ravi):
        set_cached_summary(p)

    db.session.commit()

    print("Seed complete!")
    print(
        f"People: {db.session.query(Person).count()} | "
        f"Conversations: {db.session.query(Conversation).count()} | "
        f"Turns: {db.session.query(TranscriptTurn).count()} | "
        f"Embeddings: {db.session.query(Embedding).count()}"
    )


def main():
    app = create_app()
    with app.app_context():
        db_path = reset_database_file(app)
        print("Recreating tables...")
        db.create_all()
        print(f"Created new DB at: {db_path}")
        seed_data()


if __name__ == "__main__":
    main()