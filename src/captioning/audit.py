import html
from pathlib import Path

import pandas as pd

from captioning.clean import load_config


def main(n_images=60):
    cfg = load_config()
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "captions_split.csv")
    train = df[df["split"] == "train"]

    seed = cfg.get("seed", 42)
    sample = (train["image"].drop_duplicates()
              .sample(n=n_images, random_state=seed).tolist())

    image_dir = Path(cfg["paths"]["raw_images"]).resolve()
    page = ["<html><body style='font-family:sans-serif; max-width:700px'>",
            "<h1>Label audit: 60 random training images</h1>"]
    rows = []
    for number, name in enumerate(sample, start=1):
        captions = train.loc[train["image"] == name, "caption"].tolist()
        page.append(f"<h3>{number}. {name}</h3>")
        page.append(f"<img src='{(image_dir / name).as_uri()}' width='450'>")
        page.append("<ol>")
        page += [f"<li>{html.escape(c)}</li>" for c in captions]
        page.append("</ol><hr>")
        rows.append({"number": number, "image": name,
                     "verdict": "", "error_type": "", "note": ""})
    page.append("</body></html>")

    html_path = Path(cfg["paths"]["processed"]) / "label_audit.html"
    html_path.write_text("\n".join(page), encoding="utf-8")

    csv_path = Path("reports") / "label_audit.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Sampled {len(sample)} training images (seed {seed})")
    print(f"Open this in your browser: {html_path.resolve()}")
    print(f"Fill in your verdicts here: {csv_path.resolve()}")


if __name__ == "__main__":
    main()