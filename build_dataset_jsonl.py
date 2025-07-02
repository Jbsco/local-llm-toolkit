import json
from pathlib import Path

# Extensions to include
exts = {
    '.adb', '.ads', '.py', '.yaml', '.yml', '.md', '.tex',
    '.sh', '.dockerfile', '.gpr', '.cpp', '.h'
}

# Set input and output paths
INPUT_DIR = Path("codebase")
OUTPUT_FILE = Path("codebase_dataset.jsonl")

def valid_file(file: Path):
    if file.is_file():
        ext = file.suffix.lower()
        # handle Dockerfile separately, which may have no suffix
        if file.name.lower() == "dockerfile":
            return True
        return ext in exts
    return False

def process_codebase(input_dir: Path, output_file: Path):
    count = 0
    with output_file.open("w", encoding="utf-8") as out:
        for file in input_dir.rglob("*"):
            if not valid_file(file):
                continue
            try:
                text = file.read_text(encoding="utf-8").strip()
                if text:
                    item = {
                        "path": str(file.relative_to(input_dir)),
                        "text": text
                    }
                    json.dump(item, out, ensure_ascii=False)
                    out.write("\n")
                    count += 1
            except Exception as e:
                print(f"[!] Skipping {file}: {e}")
    print(f"✅ Wrote {count} entries to {output_file}")

if __name__ == "__main__":
    process_codebase(INPUT_DIR, OUTPUT_FILE)
