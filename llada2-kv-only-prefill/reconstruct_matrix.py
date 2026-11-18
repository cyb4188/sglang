"""Reconstruct selected full prompts from the pinned public source files."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_messages(source_dir, selected):
    import pyarrow.parquet as pq

    messages = {}
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    for dataset, filename in (("ShareGPT", "sharegpt.json"),
                              ("OpenHermes", "openhermes_all.json")):
        rows = json.loads((source_dir / filename).read_text(encoding="utf-8"))
        by_id = {row["id"]: row for row in rows} if dataset == "ShareGPT" else None
        for item in selected:
            if item["dataset"] != dataset:
                continue
            uid, turn = item["uid"].removeprefix("raw/").rsplit("/turn", 1)
            row = by_id[uid] if by_id is not None else rows[int(uid)]
            messages[dataset, item["uid"]] = [
                {"role": roles[m["from"]], "content": m["value"]}
                for m in row["conversations"][:int(turn) + 1]
            ]
        del rows, by_id
    wanted = {item["uid"] for item in selected if item["dataset"] == "OpenOrca"}
    for batch in pq.ParquetFile(source_dir / "openorca.parquet").iter_batches(
        columns=["id", "system_prompt", "question"]
    ):
        for row in batch.to_pylist():
            if row["id"] in wanted:
                conversation = []
                if row["system_prompt"]:
                    conversation.append({"role": "system", "content": row["system_prompt"]})
                conversation.append({"role": "user", "content": row["question"]})
                messages["OpenOrca", row["id"]] = conversation
    wild = [item for item in selected if item["dataset"] == "WildChat"]
    max_row = max(int(item["uid"].split("/")[0]) for item in wild)
    rows = []
    for batch in pq.ParquetFile(source_dir / "wildchat.parquet").iter_batches(
        batch_size=2048, columns=["conversation"]
    ):
        rows.extend(batch.to_pylist())
        if len(rows) > max_row:
            break
    for item in wild:
        uid, turn = item["uid"].rsplit("/turn", 1)
        messages["WildChat", item["uid"]] = [
            {"role": m["role"], "content": m["content"]}
            for m in rows[int(uid)]["conversation"][:int(turn) + 1]
        ]
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    selected = [json.loads(line) for line in (root / "matrix_source_manifest.jsonl").read_text().splitlines()]
    sources = json.loads((root / "source_files.json").read_text())
    for source in sources.values():
        if digest(args.source_dir / source["filename"]) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {source['filename']}")
    messages = load_messages(args.source_dir, selected)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with args.output.open("w", encoding="utf-8") as output:
        for item in selected:
            ids = tokenizer.apply_chat_template(messages[item["dataset"], item["uid"]],
                                                tokenize=True, add_generation_prompt=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if len(ids) != item["input_len"] or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != item["input_sha"]:
                raise ValueError(f"Tokenization mismatch: {item['dataset']}/{item['uid']}")
            output.write(json.dumps(dict(item, input_ids=ids)) + "\n")
    print(f"Reconstructed and verified {len(selected)} inputs")


if __name__ == "__main__":
    main()
