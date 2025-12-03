import base64
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px


SAMPLE_DIR = "../run/sample/lh_birthdeath_02-12-25"
PCA_MODE = 0  # 0 = parameters, 1 = images

# ---------------------------
# Load data
# ---------------------------
# must include "filename" for each image
df = pd.read_csv(os.path.join(SAMPLE_DIR, "report_0.csv"))
df.columns = df.columns.str.strip()  # remove any whitespace from column names

# construct per-row image filenames from the 'step' column (ensure it's string)
df["img_filename"] = "out_0_" + df["step"].astype(str) + "_0.png"
excl = ["walk_id", "step", "attempt", "status", "target", "img_filename"]
param_cols = [c for c in df.columns if c not in excl]

img_vectors = []
print("Loading images...")
for img_filename in df["img_filename"]:
    img = Image.open(os.path.join(SAMPLE_DIR, img_filename)
                     ).convert("L").resize((64, 64))
    img_vectors.append(np.array(img).flatten())
img_vectors = np.vstack(img_vectors)
print("Images loaded.")


# # define parameter columns (exclude the filename and any image columns)
# param_cols = [c for c in df.columns if c not in ("filename", "img", "imgdata")]

# # PCA on parameters
print("Starting PCA...")
pca = PCA(n_components=2)
df[["pca1_p", "pca2_p"]] = pca.fit_transform(
    df[param_cols])  # PCA on parameters
var_ratio_p = pca.explained_variance_ratio_
df[["pca1_i", "pca2_i"]] = pca.fit_transform(img_vectors)  # PCA on images
var_ratio_i = pca.explained_variance_ratio_
print("PCA complete.")

xlab_i = f"PCA 1 ({var_ratio_i[0]*100:.1f} % var)"
ylab_i = f"PCA 2 ({var_ratio_i[1]*100:.1f} % var)"
xlab_p = f"PCA 1 ({var_ratio_p[0]*100:.1f} % var)"
ylab_p = f"PCA 2 ({var_ratio_p[1]*100:.1f} % var)"


# ---------------------------
# Encode images for display
# ---------------------------
def encode_image(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext = path.split(".")[-1]
    return f"data:image/{ext};base64,{encoded}"


df["imgdata"] = df["img_filename"].apply(
    lambda name: encode_image(os.path.join(SAMPLE_DIR, name))
)


# ---------------------------
# Build Dash app
# ---------------------------
app = dash.Dash(__name__)


def make_fig(mode):

    if int(mode) == 0:
        xcol, ycol, colorcol = "pca1_p", "pca2_p", "pca1_i"
        labels = {
            "pca1_p": f"PCA 1 (params) ({var_ratio_p[0]*100:.1f}% var)",
            "pca2_p": f"PCA 2 (params) ({var_ratio_p[1]*100:.1f}% var)",
            "pca1_i": f"Image PC1 ({var_ratio_i[0]*100:.1f}% var)",
        }
        title = "PCA of Parameter Space (hover a point to view the image)"
    else:
        xcol, ycol, colorcol = "pca1_i", "pca2_i", "pca1_p"
        labels = {
            "pca1_i": f"PCA 1 (images) ({var_ratio_i[0]*100:.1f}% var)",
            "pca2_i": f"PCA 2 (images) ({var_ratio_i[1]*100:.1f}% var)",
            "pca1_p": f"Parameter PC1 ({var_ratio_p[0]*100:.1f}% var)",
        }
        title = "PCA of Image Space (hover a point to view the parameters)"

    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        # hover_data=param_cols,
        color=colorcol,   # color by other PCA component
        color_continuous_scale="Viridis",
        # embed row index so we can reliably find the row on hover
        custom_data=[df.index.tolist()],
        labels=labels,
        title=title,
    )
    return fig


fig = make_fig(PCA_MODE)

# layout: container height is 85% of viewport; use flex to give scatter most vertical space
app.layout = html.Div(
    [
        dcc.Store(id="pca-mode", data=PCA_MODE),
        # main content area
        dcc.Graph(
            id="scatter",
            figure=fig,
            config={"responsive": True},
            style={
                "flex": "0 0 55%",   # take ~65% of horizontal space
                "height": "100%",    # fill container height
                "minWidth": "600px",
            },
        ),
        html.Div(
            id="image-container",
            style={
                "flex": "0 0 45%",   # remaining horizontal space
                "height": "100%",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "overflow": "hidden",
                "padding": "8px",
            },
        ),
        # bottom-left fixed toggle button (doesn't change layout sizes)
        html.Button(
            "Switch PCA Mode",
            id="mode-toggle-btn",
            n_clicks=0,
            style={
                "position": "fixed",
                "left": "12px",
                "bottom": "12px",
                "zIndex": "9999",
                "padding": "8px 12px",
                "fontSize": "14px",
            },
        ),
    ],
    style={
        "display": "flex",
        "flexDirection": "row",
        "height": "85vh",   # controls vertical proportion of the page
        "boxSizing": "border-box",
    },
)


# ---------------------------
# Callback: update image on click
# ---------------------------
@app.callback(
    Output('image-container', 'children'),
    Input('scatter', 'hoverData')
)
def update_image(hoverData):
    if hoverData is None:
        return html.Div("Hover a point to view its image.", style={"padding": "8px", "textAlign": "center"})

    pt = hoverData["points"][0]
    # prefer customdata if present, fallback to pointIndex
    idx = None
    if "customdata" in pt and pt["customdata"] is not None:
        # customdata may be a scalar or list
        cd = pt["customdata"]
        idx = int(cd[0]) if isinstance(
            cd, (list, tuple, np.ndarray)) else int(cd)
    elif "pointIndex" in pt:
        idx = int(pt["pointIndex"])
    else:
        return html.Div("Could not determine point index from hoverData.")

    img_src = df.iloc[idx]["imgdata"]
    # build parameter display box, split into 1-3 columns and allow scrolling
    params = list(df.iloc[idx][param_cols].items())
    max_lines_per_col = 12

    def make_item(k, v):
        return html.Div(f"{k}: {v}", style={
            "fontSize": "12px",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "padding": "2px 0",
        })

    n = len(params)
    if n == 0:
        params_box = html.Div("No parameters", style={"fontSize": "12px"})
    else:
        cols = 1 if n <= max_lines_per_col else min(
            3, int(np.ceil(n / max_lines_per_col)))
        chunks = np.array_split(params, cols)
        col_divs = []
        for chunk in chunks:
            items = [make_item(k, v) for k, v in chunk]
            # minWidth:0 keeps columns from overflowing their flex container
            col_divs.append(
                html.Div(items, style={"flex": "1", "minWidth": "0"}))
        params_box = html.Div(
            col_divs,
            style={
                "display": "flex",
                "gap": "8px",
                "overflowY": "auto",
                "maxHeight": "100%",
                "padding": "4px",
                "fontFamily": "monospace",
                "fontSize": "12px",
            },
        )

    return html.Div(
        [
            # image area — give it explicit height fraction and center the image
            html.Div(
                html.Img(
                    src=img_src,
                    style={
                        "width": "100%",
                        "height": "100%",
                        "objectFit": "contain",   # or "cover" to fill/crop
                        "display": "block",
                    },
                ),
                style={
                    "flex": "0 0 60%",
                    "width": "100%",
                    "height": "60%",           # explicit height so img height:100% works
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "textAlign": "center",
                    "paddingBottom": "6px",
                },
            ),
            # params area — allow scrolling if content is large
            html.Div(
                params_box,
                style={
                    "flex": "0 0 40%",
                    "width": "100%",
                    "height": "40%",         # explicit height for scrolling
                    "padding": "8px",
                    "borderTop": "1px solid #ddd",
                    "overflow": "auto",
                },
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "height": "100%"},
    )


# ---------------------------
# ---------------------------
# Toggle button callback: flip PCA mode and update scatter figure + button label
# ---------------------------
@app.callback(
    Output("scatter", "figure"),
    Output("mode-toggle-btn", "children"),
    Output("pca-mode", "data"),
    Input("mode-toggle-btn", "n_clicks"),
    State("pca-mode", "data"),
)
def toggle_pca_mode(n_clicks, mode):
    # Flip mode on each click; if no stored mode, use default
    if mode is None:
        mode = PCA_MODE
    try:
        mode_int = int(mode)
    except Exception:
        mode_int = PCA_MODE
    new_mode = 1 - mode_int
    fig = make_fig(new_mode)
    # Button label indicates the action (what will be shown next time)
    btn_label = "Show Image PCA" if new_mode == 0 else "Show Param PCA"
    return fig, btn_label, new_mode


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
