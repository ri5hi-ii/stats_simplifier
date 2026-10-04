import io

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


def load(file):
    name = file.filename.lower()
    data = io.BytesIO(file.read())
    if name.endswith(".csv"):
        return pd.read_csv(data)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(data)
    raise ValueError("please upload a .csv or .xlsx file")


def fmt(x):
    s = f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def summarise(df):
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = df.columns.map(str)
    rows, cols = df.shape

    # 1. Big picture
    overview = [f"Your data has {rows:,} rows and {cols} columns."]
    missing = int(df.isna().sum().sum())
    if missing:
        worst = df.isna().sum().idxmax()
        overview.append(f"{missing:,} cells are empty, most of them in “{worst}”.")
    else:
        overview.append("No cells are empty.")
    dupes = int(df.duplicated().sum())
    if dupes:
        overview.append(f"{dupes:,} rows are exact duplicates.")

    # 2. Number columns
    num = df.select_dtypes("number")
    columns = []
    for c in num.columns[:8]:
        s = num[c].dropna()
        if s.empty:
            continue
        pts = [
            f"Average is {fmt(s.mean())}; the middle value is {fmt(s.median())}.",
            f"Values run from {fmt(s.min())} to {fmt(s.max())}.",
        ]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if out:
                pts.append(f"{out} unusual value{'s' if out > 1 else ''} sit far from the rest.")
        if len(s) >= 6:
            h = len(s) // 2
            a, b = s.iloc[:h].mean(), s.iloc[h:].mean()
            if a:
                ch = (b - a) / abs(a) * 100
                if abs(ch) >= 5:
                    pts.append(
                        f"Later rows average {abs(ch):.0f}% {'higher' if ch > 0 else 'lower'} than earlier rows."
                    )
        columns.append({"name": c, "points": pts})

    # 3. Category columns
    for c in df.select_dtypes(exclude="number").columns[:4]:
        s = df[c].dropna()
        if s.empty or s.nunique() > 50:
            continue
        top = s.value_counts()
        share = top.iloc[0] / len(s) * 100
        columns.append({
            "name": c,
            "points": [
                f"{s.nunique()} different values; “{top.index[0]}” is the most common ({share:.0f}%).",
            ],
        })

    # 4. Strongest link between two number columns
    links = []
    if num.shape[1] >= 2:
        corr = num.iloc[:, :15].corr().abs()
        best = None
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v) and (best is None or v > best[2]):
                    best = (a, b, v)
        if best and best[2] >= 0.5:
            r = num[best[0]].corr(num[best[1]])
            way = "rise together" if r > 0 else "move in opposite directions"
            links.append(f"“{best[0]}” and “{best[1]}” {way} (strength {abs(r):.2f} out of 1).")

    preview = df.head(8).to_html(index=False, border=0)
    return {"overview": overview, "columns": columns, "links": links, "preview": preview}


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {}
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            ctx["error"] = "Choose a file first."
        else:
            try:
                ctx.update(summarise(load(f)))
                ctx["filename"] = f.filename
            except Exception as e:
                ctx["error"] = f"Couldn’t read that file: {e}."
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
