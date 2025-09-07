# dump_project.py
# Creates project_dump.txt in the project root, containing all text files
# with a header line showing each file's path relative to the root.

import os
from pathlib import Path

# ---- Tweak these if you like ----
OUTPUT_FILENAME = "project_dump.txt"
MAX_FILE_SIZE_MB = 3  # skip very large files

IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env",
    "node_modules", ".cache", ".parcel-cache", ".sass-cache",
    "build", "dist", ".next", ".turbo", "target", ".gradle",
    ".idea", ".vscode", ".terraform"
}

# Common binary/heavy extensions to skip
IGNORE_FILE_EXTS = {
    # compiled/binaries
    ".pyc", ".pyo", ".class", ".o", ".obj", ".a", ".lib",
    ".so", ".dylib", ".dll", ".exe",
    # archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # media
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".wav", ".flac", ".ogg",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    # fonts
    ".ttf", ".otf", ".eot", ".woff", ".woff2",
    # dbs and others
    ".db", ".db3", ".sqlite", ".sqlite3", ".pdf", ".bin", ".iso",
    ".log"
}

# Exact file names to skip (secrets/locks/dump itself)
IGNORE_FILE_NAMES = {
    OUTPUT_FILENAME,
    ".env", ".env.local", ".env.development", ".env.production",
    "Pipfile.lock", "poetry.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Thumbs.db", ".DS_Store"
}


def should_skip_file(path: Path) -> bool:
    name = path.name
    if name in IGNORE_FILE_NAMES:
        return True
    if path.suffix.lower() in IGNORE_FILE_EXTS:
        return True
    try:
        if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return True
    except OSError:
        return True  # if we can't stat, skip it
    return False


def main():
    root = Path(__file__).resolve().parent
    out_path = root / OUTPUT_FILENAME

    with out_path.open("w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            # Prune ignored directories (by name)
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                # Skip dump file and other ignored files
                if should_skip_file(fpath):
                    continue

                rel = fpath.relative_to(root).as_posix()
                out.write(f"\n\n--- FILE: {rel} ---\n\n")

                # Best-effort read in text
                try:
                    with fpath.open("r", encoding="utf-8") as f:
                        out.write(f.read())
                except UnicodeDecodeError:
                    try:
                        with fpath.open("r", encoding="latin-1") as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f"[Could not read file: {e}]\n")
                except Exception as e:
                    out.write(f"[Could not read file: {e}]\n")

    print(f"✅ Dump complete → {out_path}")


if __name__ == "__main__":
    main()
