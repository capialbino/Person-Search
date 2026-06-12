import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── palette ──────────────────────────────────────────────────────────
C = {
    "input":    "#2E86AB",
    "backbone": "#7B2D8B",
    "fpn_rpn":  "#E94F37",
    "roi":      "#F18F01",
    "det":      "#3A86FF",
    "reid":     "#C73E1D",
    "loss":     "#2D6A4F",
    "output":   "#44BBA4",
    "match":    "#44BBA4",
    "bg":       "#1A1A2E",
    "text":     "white",
}

BOX_H = 0.55
BOX_W = 2.6
SMALL_W = 2.2
FONTSIZE = 9


def box(ax, cx, cy, text, color, w=BOX_W, h=BOX_H, fontsize=FONTSIZE, sub=None):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="white", linewidth=1.2, zorder=3,
    )
    ax.add_patch(rect)
    if sub:
        ax.text(cx, cy + 0.08, text, ha="center", va="center",
                fontsize=fontsize, color=C["text"], fontweight="bold", zorder=4)
        ax.text(cx, cy - 0.14, sub, ha="center", va="center",
                fontsize=fontsize - 1.5, color="lightgray", zorder=4, style="italic")
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, color=C["text"], fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#AAAAAA", lw=1.4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=12),
        zorder=2,
    )


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 — TRAINING
# ═══════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(6, 10))
fig1.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 6)
ax.set_ylim(-0.5, 10.5)
ax.axis("off")

ax.text(3, 10.1, "TRAINING PIPELINE", ha="center", va="center",
        fontsize=13, color="white", fontweight="bold")

CX = 3.0
steps = [
    (9.3, "Gallery Image\n(full scene)", C["input"]),
    (8.3, "Backbone", C["backbone"], "ResNet-101  or  DINOv2"),
    (7.3, "FPN", C["fpn_rpn"], "Feature Pyramid Network"),
    (6.3, "RPN", C["fpn_rpn"], "Region Proposal Network"),
    (5.3, "ROI Align", C["roi"]),
]

for entry in steps:
    y = entry[0]
    label = entry[1]
    color = entry[2]
    sub = entry[3] if len(entry) > 3 else None
    box(ax, CX, y, label, color, sub=sub)

# arrows top section
for y_top, y_bot in [(9.0, 8.6), (8.0, 7.6), (7.0, 6.6), (6.0, 5.6)]:
    arrow(ax, CX, y_top, CX, y_bot)

# split into two branches at ROI Align
LEFT_CX = 1.6
RIGHT_CX = 4.4

arrow(ax, CX, 5.02, LEFT_CX, 4.5)
arrow(ax, CX, 5.02, RIGHT_CX, 4.5)

box(ax, LEFT_CX, 4.2, "Detection Head", C["det"], w=SMALL_W)
box(ax, RIGHT_CX, 4.2, "Re-ID Head", C["reid"], w=SMALL_W)

arrow(ax, LEFT_CX, 3.92, LEFT_CX, 3.4)
arrow(ax, RIGHT_CX, 3.92, RIGHT_CX, 3.4)

box(ax, LEFT_CX, 3.1, "cls + reg loss", C["det"], w=SMALL_W, h=0.5)
box(ax, RIGHT_CX, 3.1, "OIM / CircleLoss", C["reid"], w=SMALL_W, h=0.5)

# converge to joint loss
arrow(ax, LEFT_CX, 2.85, 2.2, 2.25)
arrow(ax, RIGHT_CX, 2.85, 3.8, 2.25)

box(ax, CX, 2.0, "Joint Loss", C["loss"], w=2.0, h=0.5)

fig1.tight_layout(pad=0.5)
fig1.savefig("assets/pipeline_training.png", dpi=150,
             bbox_inches="tight", facecolor=C["bg"])
print("Saved: assets/pipeline_training.png")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — INFERENCE
# ═══════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(9, 10))
fig2.patch.set_facecolor(C["bg"])
ax2.set_facecolor(C["bg"])
ax2.set_xlim(0, 9)
ax2.set_ylim(-0.5, 10.5)
ax2.axis("off")

ax2.text(4.5, 10.1, "INFERENCE PIPELINE", ha="center", va="center",
         fontsize=13, color="white", fontweight="bold")

# ── column labels ──
ax2.text(2.2, 9.7, "QUERY", ha="center", fontsize=10,
         color="#AAAAFF", fontweight="bold")
ax2.text(6.5, 9.7, "GALLERY", ha="center", fontsize=10,
         color="#FFAAAA", fontweight="bold")

# ── LEFT column (query) ──
QX = 2.2
q_steps = [
    (9.2, "Query Image\n+ GT bounding box", C["input"]),
    (7.9, "Backbone", C["backbone"], "shared weights"),
    (6.9, "ROI Align\n(GT box)", C["roi"]),
    (5.9, "Re-ID Head", C["reid"]),
    (4.9, "query_feat\n(D-dim vector)", C["output"]),
]
for entry in q_steps:
    y = entry[0]
    label = entry[1]
    color = entry[2]
    sub = entry[3] if len(entry) > 3 else None
    box(ax2, QX, y, label, color, w=SMALL_W, sub=sub)

for y_top, y_bot in [(8.92, 8.17), (8.17, 7.5 - 0.22), (6.67, 6.17), (5.67, 5.17)]:
    arrow(ax2, QX, y_top, QX, y_bot)

# ── RIGHT column (gallery) ──
GX = 6.5
g_steps_top = [
    (9.2, "Gallery Images\n(full scenes)", C["input"]),
    (7.9, "Backbone + FPN", C["backbone"], "shared weights"),
    (6.9, "RPN", C["fpn_rpn"]),
    (5.9, "ROI Align", C["roi"]),
]
for entry in g_steps_top:
    y = entry[0]
    label = entry[1]
    color = entry[2]
    sub = entry[3] if len(entry) > 3 else None
    box(ax2, GX, y, label, color, w=SMALL_W, sub=sub)

for y_top, y_bot in [(8.92, 8.17), (8.17, 7.22), (6.67, 6.17)]:
    arrow(ax2, GX, y_top, GX, y_bot)

# split gallery into Det and ReID
GL = 5.3
GR = 7.7
arrow(ax2, GX, 5.67, GL, 5.17)
arrow(ax2, GX, 5.67, GR, 5.17)

box(ax2, GL, 4.87, "Det. Head", C["det"], w=1.9)
box(ax2, GR, 4.87, "Re-ID Head", C["reid"], w=1.9)

arrow(ax2, GL, 4.59, GL, 4.09)
arrow(ax2, GR, 4.59, GR, 4.09)

box(ax2, GL, 3.8, "boxes + scores", C["det"], w=1.9, h=0.5)
box(ax2, GR, 3.8, "det_feats", C["reid"], w=1.9, h=0.5)

# ── matching zone ──
MX = 4.35
MY = 2.6

arrow(ax2, QX, 4.67, QX, MY + 0.27)
arrow(ax2, GR, 3.55, GR, MY + 0.27)

# horizontal lines to center
ax2.annotate("", xy=(MX, MY), xytext=(QX, MY),
             arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=1.4), zorder=2)
ax2.annotate("", xy=(MX, MY), xytext=(GR, MY),
             arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=1.4), zorder=2)
arrow(ax2, MX, MY, MX, MY - 0.35)

box(ax2, MX, 2.0, "Cosine Similarity", C["match"], w=3.0, h=0.55)
arrow(ax2, MX, 1.72, MX, 1.22)
box(ax2, MX, 1.0, "Ranked Results  →  mAP, Top-1", C["output"], w=4.2, h=0.55)

fig2.tight_layout(pad=0.5)
fig2.savefig("assets/pipeline_inference.png", dpi=150,
             bbox_inches="tight", facecolor=C["bg"])
print("Saved: assets/pipeline_inference.png")
