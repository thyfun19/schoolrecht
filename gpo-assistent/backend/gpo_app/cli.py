import argparse
import json
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError

from .database import create_engine
from .ingestion import import_existing, register_records
from .repository import counts


def main():
    parser = argparse.ArgumentParser(
        description="Lokaal GPO-bronregister; geen downloads of modelaanroepen."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser("import-existing")
    importer.add_argument(
        "--catalog", type=Path, default=Path("../sources/catalogus.json")
    )
    importer.add_argument(
        "--pdf-root", type=Path, default=Path("../../onderzoek_gpo/data/pdf")
    )
    importer.add_argument(
        "--pages", type=Path, default=Path("../../onderzoek_gpo/data/pages.json")
    )
    importer.add_argument("--corpus", type=Path, default=Path("data/e5/corpus.json"))
    records = sub.add_parser("register-records")
    records.add_argument("path", type=Path)
    sub.add_parser("counts")
    sub.add_parser("audit-chunks")
    chunks = sub.add_parser("prepare-chunks")
    chunks.add_argument(
        "--tokenizer", type=Path, default=Path("data/e5/model/tokenizer.json")
    )
    chunks.add_argument("--token-limit", type=int, default=384)
    args = parser.parse_args()
    engine = create_engine()
    try:
        if args.command == "import-existing":
            result = import_existing(
                engine, args.catalog, args.pdf_root, args.pages, args.corpus
            )
        elif args.command == "register-records":
            result = register_records(
                engine, json.loads(args.path.read_text(encoding="utf-8"))
            )
        elif args.command == "prepare-chunks":
            from .chunking import LocalTokenCounter, prepare_chunks

            result = prepare_chunks(
                engine, LocalTokenCounter(args.tokenizer), args.token_limit
            )
        elif args.command == "audit-chunks":
            from .chunk_audit import audit

            result = audit(engine)
        else:
            result = counts(engine)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except ValueError as error:
        parser.exit(1, str(error) + "\n")
    except SQLAlchemyError:
        parser.exit(
            1,
            "PostgreSQL niet beschikbaar of registratie geweigerd; controleer configuratie/migratie.\n",
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
