from datetime import datetime
from uuid import uuid4
from . import db
from sqlalchemy import Enum, UniqueConstraint, Index, CheckConstraint


# ---------- Mixins ----------
class TimestampMixin:
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# ---------- Core: People ----------
class Person(db.Model, TimestampMixin):
    """
    A contact. Unknown entries are mergeable later.
    """
    __tablename__ = "people"

    id = db.Column(db.Integer, primary_key=True)
    display_name = db.Column(db.String(120), nullable=False)          # "Mom", "Dr. Mehra", or "unknown_ab12cd34"
    relation = db.Column(db.String(80))                                # optional: Mother / Friend / Doctor
    photo_url = db.Column(db.String(512))                              # optional profile image
    is_unknown = db.Column(db.Boolean, default=False, nullable=False)
    temp_tag = db.Column(db.String(64), unique=True)                   # "unknown_<uuid8>" for merge flows
    notes = db.Column(db.Text)                                         # freeform notes for the person
    last_summary_cached = db.Column(db.Text)                           # latest recap bullets for fast sidebar

    # Relationships
    conversations = db.relationship(
        "Conversation",
        back_populates="person",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    embeddings = db.relationship(
        "Embedding",
        back_populates="person",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_people_unknown_temp_tag", "is_unknown", "temp_tag"),
    )

    @staticmethod
    def make_unknown():
        tag = f"unknown_{uuid4().hex[:8]}"
        return Person(display_name=tag, is_unknown=True, temp_tag=tag)


# ---------- Core: Conversations ----------
class Conversation(db.Model, TimestampMixin):
    """
    A single recording session in 'Conversation Mode'.
    """
    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)

    # Null if we recorded while person was unknown and wasn't merged yet.
    person_id = db.Column(
        db.Integer,
        db.ForeignKey("people.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ended_at = db.Column(db.DateTime)                                  # set when Stop is pressed

    # Summarization output; empty on failure
    summary = db.Column(db.Text)

    # Unknown-face UX: keep a snapshot path if we recorded before registration/merge
    face_snapshot_path = db.Column(db.String(512))

    # Provenance / STT details kept provider-agnostic
    source = db.Column(db.String(24), default="glasses")               # 'glasses' (overlay) or 'memory_bank'
    stt_provider = db.Column(db.String(64), default="web_speech")      # matches current browser STT
    stt_lang = db.Column(db.String(16), default="en-US")               # BCP-47

    person = db.relationship("Person", back_populates="conversations")

    turns = db.relationship(
        "TranscriptTurn",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="TranscriptTurn.timestamp",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_conversations_person_started", "person_id", "started_at"),
        CheckConstraint("(ended_at IS NULL) OR (ended_at >= started_at)", name="ck_convo_time_order"),
    )


# ---------- Core: Finalized transcript lines ----------
class TranscriptTurn(db.Model, TimestampMixin):
    """
    Finalized lines from the transcript (one active speaker at a time).
    Interim text is UI-only and not stored.
    """
    __tablename__ = "transcript_turns"

    id = db.Column(db.Integer, primary_key=True)

    conversation_id = db.Column(
        db.Integer,
        db.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    speaker = db.Column(Enum("PATIENT", "VISITOR", name="speaker_enum"), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # when the line was finalized

    # Optional hints from STT
    confidence = db.Column(db.Float)
    lang = db.Column(db.String(16))                                              # per-line BCP-47, if available

    conversation = db.relationship("Conversation", back_populates="turns")

    __table_args__ = (
        Index("ix_turns_conversation_timestamp", "conversation_id", "timestamp"),
    )


# ---------- Optional: generic embedding store (future-proofing) ----------
class Embedding(db.Model, TimestampMixin):
    """
    Provider-agnostic face embedding storage to enable switching to server-side matching later.
    """
    __tablename__ = "embeddings"

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(
        db.Integer,
        db.ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    provider = db.Column(db.String(80), nullable=False)      # 'azure_face', 'aws_rekognition', 'local', etc.
    vector_json = db.Column(db.Text, nullable=False)         # JSON list or opaque provider blob
    dim = db.Column(db.Integer)                              # optional dimension hint

    person = db.relationship("Person", back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("person_id", "provider", name="uq_embeddings_person_provider"),
    )
