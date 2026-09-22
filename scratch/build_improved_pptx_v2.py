"""
Build Card-1-Improved-v2.pptx
AI-based Customer Purchase Prediction for Unisex Skincare
16 slides | Original light purple/lavender theme matching Card-1.pptx
Colors sourced from original PPTX XML:
  - Background: #FFFFFF (white)
  - Card fill:  #F5F0FF (light lavender)
  - Title text: #4D4060 (deep purple)
  - Accent:     #6F5F82 (muted purple)
  - Highlight:  #7C5CBF (vivid purple)
  - Strong acc: #A084CA (medium purple)
Data sourced from README.md (corrected metrics)
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Paths ────────────────────────────────────────────────────────────────────
BASE  = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\AI_Cosmetic_Purchase_Prediction")
VIZ   = BASE / "visualisations"
DASH  = Path(r"C:\Users\TUSHAR\.gemini\antigravity\brain\80ae597a-1dd1-4301-9bba-d77af2b5c3ae\.user_uploaded\media_1790094766403.png")
OUT   = BASE / "Card-1-Improved-v2.pptx"

IMGS = {
    "eda":       VIZ / "eda_age_and_gender.png",
    "feat_imp":  VIZ / "feature_importance.png",
    "metrics":   VIZ / "model_metrics_comparison.png",
    "roc":       VIZ / "model_roc_curves.png",
    "cm":        VIZ / "confusion_matrix_random_forest.png",
    "threshold": VIZ / "decision_threshold_tradeoff.png",
    "age_dist":  VIZ / "age_group_buyer_distribution.png",
    "donut":     VIZ / "budget_allocation_donut.png",
    "dashboard": DASH,
}

# ── Colour Palette (matching original Card-1.pptx) ───────────────────────────
BG        = RGBColor(0xF8, 0xF5, 0xFF)   # near-white lavender background
CARD      = RGBColor(0xFF, 0xFF, 0xFF)   # white card fill
CARD2     = RGBColor(0xF0, 0xEB, 0xFF)   # soft lavender card
TITLE     = RGBColor(0x4D, 0x40, 0x60)   # deep purple – slide titles
ACCENT    = RGBColor(0x6F, 0x5F, 0x82)   # muted purple – labels/subtext
VIVID     = RGBColor(0x7C, 0x5C, 0xBF)   # vivid purple – strong accent
MED       = RGBColor(0xA0, 0x84, 0xCA)   # medium purple – badges
DARK_TEXT = RGBColor(0x2D, 0x23, 0x3A)   # very dark purple for body
LIGHT_TXT = RGBColor(0x9D, 0x8E, 0xAE)   # light purple – secondary text
SHADOW    = RGBColor(0xE9, 0xE0, 0xFA)   # very light lavender shadow/border
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
PINK_ACC  = RGBColor(0xC8, 0x7F, 0xC8)   # pink-purple highlight
TEAL      = RGBColor(0x5B, 0xB8, 0xC1)   # teal for variety

# Slide dimensions – Widescreen 13.33 × 7.5 in
W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
BLANK = prs.slide_layouts[6]

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_rect(slide, l, t, w, h, fill_rgb, radius=False):
    shape = slide.shapes.add_shape(1, l, t, w, h)
    shape.line.fill.background()
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    return shape


def txb(slide, text, l, t, w, h,
        font_size=14, bold=False, italic=False,
        color=DARK_TEXT, align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def lavender_bg(slide):
    add_rect(slide, 0, 0, W, H, BG)


def top_bar(slide, height=Inches(0.07)):
    add_rect(slide, 0, 0, W, height, VIVID)


def bottom_bar(slide, height=Inches(0.07)):
    add_rect(slide, 0, H - height, W, height, MED)


def slide_header(slide, title, subtitle=None):
    top_bar(slide)
    bottom_bar(slide)
    # White card behind title
    add_rect(slide, Inches(0.4), Inches(0.12), Inches(12.5), Inches(0.85), CARD)
    txb(slide, title,
        Inches(0.6), Inches(0.18), Inches(10.5), Inches(0.65),
        font_size=28, bold=True, color=TITLE)
    if subtitle:
        txb(slide, subtitle,
            Inches(0.6), Inches(0.7), Inches(11), Inches(0.3),
            font_size=12, color=ACCENT, italic=True)


def slide_num(slide, n):
    txb(slide, str(n),
        W - Inches(0.7), H - Inches(0.42), Inches(0.55), Inches(0.32),
        font_size=11, color=LIGHT_TXT, align=PP_ALIGN.RIGHT)


def info_card(slide, l, t, w, h, heading, body_lines, hd_size=13, bd_size=11):
    add_rect(slide, l, t, w, h, CARD)
    add_rect(slide, l, t, Inches(0.06), h, VIVID)
    txb(slide, heading, l + Inches(0.12), t + Inches(0.08),
        w - Inches(0.18), Inches(0.34),
        font_size=hd_size, bold=True, color=VIVID)
    body = "\n".join(body_lines)
    txb(slide, body, l + Inches(0.12), t + Inches(0.44),
        w - Inches(0.18), h - Inches(0.5),
        font_size=bd_size, color=DARK_TEXT)


def badge(slide, l, t, w, h, text, bg=VIVID, fg=WHITE, fsize=11):
    add_rect(slide, l, t, w, h, bg)
    txb(slide, text, l, t, w, h,
        font_size=fsize, bold=True, color=fg, align=PP_ALIGN.CENTER)


def embed_image(slide, path, l, t, w, h):
    p = Path(path)
    if p.exists():
        slide.shapes.add_picture(str(p), l, t, w, h)
    else:
        add_rect(slide, l, t, w, h, SHADOW)
        txb(slide, f"[{p.name}]", l, t + h/2 - Inches(0.2), w, Inches(0.4),
            font_size=10, color=LIGHT_TXT, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 – Title
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
top_bar(sl, Inches(0.12))
bottom_bar(sl, Inches(0.12))

# Large white card left
add_rect(sl, Inches(0.35), Inches(0.22), Inches(8.0), Inches(6.95), CARD)
add_rect(sl, Inches(0.35), Inches(0.22), Inches(0.1), Inches(6.95), VIVID)

txb(sl, "MACHINE LEARNING PROJECT",
    Inches(0.65), Inches(0.45), Inches(7.5), Inches(0.4),
    font_size=12, bold=True, color=ACCENT)

txb(sl, "AI-Based Customer\nPurchase Prediction\nfor Unisex Skincare",
    Inches(0.65), Inches(0.9), Inches(7.4), Inches(2.4),
    font_size=36, bold=True, color=TITLE)

add_rect(sl, Inches(0.65), Inches(3.45), Inches(6.5), Inches(0.04), MED)

txb(sl, "An end-to-end machine learning system predicting purchase propensity\n"
        "and identifying high-value demographic cohorts for a newly launched\n"
        "unisex skincare product.",
    Inches(0.65), Inches(3.6), Inches(7.2), Inches(1.05),
    font_size=12, color=ACCENT, italic=True)

txb(sl, "Presented by:",
    Inches(0.65), Inches(4.78), Inches(7.2), Inches(0.3),
    font_size=11, bold=True, color=VIVID)
txb(sl, "Tushar Jaiswal  •  Shatakshi Jaiswal  •  Sharafat  •  Garvit\n"
        "Tanisha  •  Tanishq Saini  •  Astha  •  Mahima Kalra",
    Inches(0.65), Inches(5.1), Inches(7.4), Inches(0.65),
    font_size=11, color=DARK_TEXT)
txb(sl, "Department of Computer Science & Engineering  |  B.Tech CSE",
    Inches(0.65), Inches(5.82), Inches(7.4), Inches(0.35),
    font_size=10, color=LIGHT_TXT)

# Right panel: Stats summary
rx = Inches(8.65)
add_rect(sl, rx, Inches(0.22), Inches(4.3), Inches(6.95), CARD2)
txb(sl, "Project At a Glance",
    rx + Inches(0.2), Inches(0.42), Inches(3.8), Inches(0.4),
    font_size=14, bold=True, color=TITLE)

glance = [
    ("10,000",  "Customer Profiles"),
    ("61.6%",   "Conversion Rate"),
    ("3 Models","LR · DT · RF"),
    ("10 Pages","Streamlit App"),
    ("67.90%",  "Best Accuracy (LR)"),
    ("0.7116",  "Best ROC-AUC (LR)"),
    ("85.23%",  "Best Recall (RF)"),
    ("8 Charts","Visualisations"),
]
gx = rx + Inches(0.2)
gy = Inches(1.0)
for i, (val, lbl) in enumerate(glance):
    col = i % 2
    row = i // 2
    bx = gx + col * Inches(2.0)
    by = gy + row * Inches(1.3)
    add_rect(sl, bx, by, Inches(1.85), Inches(1.1), CARD)
    add_rect(sl, bx, by, Inches(1.85), Inches(0.06), VIVID)
    txb(sl, val, bx, by + Inches(0.1), Inches(1.85), Inches(0.52),
        font_size=22, bold=True, color=VIVID, align=PP_ALIGN.CENTER)
    txb(sl, lbl, bx, by + Inches(0.62), Inches(1.85), Inches(0.4),
        font_size=9, color=ACCENT, align=PP_ALIGN.CENTER)

txb(sl, "Python 3.10+ · Scikit-learn · Pandas · Streamlit",
    rx + Inches(0.2), Inches(6.55), Inches(3.9), Inches(0.38),
    font_size=9.5, color=LIGHT_TXT, align=PP_ALIGN.CENTER)

slide_num(sl, 1)
print("[OK] Slide 1 - Title")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 – Agenda
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "What We'll Cover", "End-to-end walkthrough from problem to production")

agenda = [
    ("01", "Problem Statement",      "Why targeted marketing matters for a new skincare launch"),
    ("02", "Project Objectives",     "Four pillars: predict, segment, compare, deploy"),
    ("03", "Dataset & Pipeline",     "10,000 profiles — cleaning, encoding, stratified split"),
    ("04", "Exploratory Data Analysis","Class balance 61.6%, age & gender distributions"),
    ("05", "Feature Importance",     "Total spend & tenure outperform demographics"),
    ("06", "Model Comparison",       "LR vs DT vs RF — accuracy, F1, recall, ROC-AUC"),
    ("07", "ROC & Confusion Matrix", "Detailed per-model evaluation curves"),
    ("08", "Decision Threshold",     "Tuning probability cutoff for campaign strategy"),
    ("09", "Business Findings",      "Cohort demand sizing — 26-35 is #1 segment"),
    ("10", "Streamlit Dashboard",    "10-page interactive app: KPIs, predictions, ethics"),
    ("11", "Key Insights & Ethics",  "Responsible AI, GDPR compliance, no stereotyping"),
    ("12", "Conclusion & Future",    "Summary, XGBoost roadmap, drift monitoring"),
]

cols = 3
bw = Inches(4.1)
bh = Inches(0.88)
gx2 = Inches(0.2)
gy2 = Inches(0.22)
sx = Inches(0.32)
sy = Inches(1.22)

for i, (num, title, desc) in enumerate(agenda):
    col = i % cols
    row = i // cols
    lx = sx + col * (bw + gx2)
    ty = sy + row * (bh + gy2)
    add_rect(sl, lx, ty, bw, bh, CARD)
    add_rect(sl, lx, ty, Inches(0.06), bh, VIVID)
    txb(sl, num, lx + Inches(0.12), ty + Inches(0.06),
        Inches(0.42), Inches(0.36),
        font_size=16, bold=True, color=VIVID)
    txb(sl, title, lx + Inches(0.12), ty + Inches(0.38),
        bw - Inches(0.18), Inches(0.26),
        font_size=12, bold=True, color=TITLE)
    txb(sl, desc, lx + Inches(0.12), ty + Inches(0.62),
        bw - Inches(0.18), Inches(0.24),
        font_size=9.5, color=ACCENT)

slide_num(sl, 2)
print("[OK] Slide 2 - Agenda")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 – Problem Statement
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Problem Statement", "The business challenge behind this ML project")

# Context box
add_rect(sl, Inches(0.4), Inches(1.2), Inches(12.5), Inches(1.45), CARD)
add_rect(sl, Inches(0.4), Inches(1.2), Inches(0.08), Inches(1.45), VIVID)
txb(sl, "A cosmetics brand is launching an innovative unisex skincare formulation. "
        "Traditional untargeted mass marketing leads to high Customer Acquisition Cost (CAC) and low conversion rates. "
        "This project uses a two-tier ML approach:\n"
        "  1. AI Behavioral Engine: outputs continuous purchase probabilities P(Purchased=1 | x)\n"
        "  2. Business Demand Aggregation: sums predicted probabilities across cohorts to estimate expected buyer volume",
    Inches(0.58), Inches(1.3), Inches(12.1), Inches(1.3),
    font_size=12, color=DARK_TEXT)

pain = [
    ("High CAC",            "Mass campaigns reach uninterested audiences, wasting budget on low-intent customers."),
    ("Low Conversion",      "Without targeting, typical conversion is 5-15%. Predictive targeting can exceed 30-40%."),
    ("Who to Reach First?", "Marketing needs a ranked list of prospects before the launch window closes."),
]
for i, (hd, bd) in enumerate(pain):
    lx = Inches(0.4) + i * Inches(4.32)
    add_rect(sl, lx, Inches(2.85), Inches(4.15), Inches(2.2), CARD)
    add_rect(sl, lx, Inches(2.85), Inches(4.15), Inches(0.06), VIVID)
    txb(sl, hd, lx + Inches(0.15), Inches(2.98), Inches(3.85), Inches(0.35),
        font_size=14, bold=True, color=VIVID)
    txb(sl, bd, lx + Inches(0.15), Inches(3.38), Inches(3.85), Inches(1.55),
        font_size=12, color=DARK_TEXT)

txb(sl, "The question: who should marketing reach first?",
    Inches(0.4), Inches(5.22), Inches(12.5), Inches(0.5),
    font_size=17, bold=True, italic=True, color=VIVID, align=PP_ALIGN.CENTER)

# Formula box
add_rect(sl, Inches(1.5), Inches(5.85), Inches(10.3), Inches(0.95), CARD2)
txb(sl, "Expected Buyers in Cohort k  =  sum of P(Purchased=1 | xi)  for all i in Cohort k",
    Inches(1.5), Inches(5.98), Inches(10.3), Inches(0.65),
    font_size=13, italic=True, color=TITLE, align=PP_ALIGN.CENTER)

slide_num(sl, 3)
print("[OK] Slide 3 - Problem Statement")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 – Project Objectives
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Project Objectives", "Four pillars of this end-to-end machine learning solution")

objectives = [
    ("Purchase Prediction",    "Estimate the probability that an individual customer will buy the new unisex skincare product using behavioral and demographic signals."),
    ("Customer Segmentation",  "Analyse demand across age brackets and demographic cohorts to surface the highest-value customer groups for targeted marketing."),
    ("Model Comparison",       "Benchmark Logistic Regression, Decision Tree, and Random Forest across Accuracy, F1, Recall, and ROC-AUC to select the best model."),
    ("Business Intelligence",  "Deliver results via a 10-page interactive Streamlit dashboard — converting raw model scores into expected buyers, cohort share, and budget allocation insights."),
]

bw2 = Inches(5.95)
bh2 = Inches(2.25)
for i, (title, desc) in enumerate(objectives):
    col = i % 2
    row = i // 2
    lx = Inches(0.35) + col * (bw2 + Inches(0.25))
    ty = Inches(1.22) + row * (bh2 + Inches(0.2))
    add_rect(sl, lx, ty, bw2, bh2, CARD)
    add_rect(sl, lx, ty, bw2, Inches(0.06), VIVID)
    # Number badge
    badge(sl, lx + Inches(0.15), ty + Inches(0.15), Inches(0.45), Inches(0.45),
          str(i+1), VIVID, WHITE, 14)
    txb(sl, title, lx + Inches(0.72), ty + Inches(0.15), bw2 - Inches(0.85), Inches(0.42),
        font_size=15, bold=True, color=TITLE)
    txb(sl, desc, lx + Inches(0.15), ty + Inches(0.72), bw2 - Inches(0.25), Inches(1.42),
        font_size=12, color=DARK_TEXT)

slide_num(sl, 4)
print("[OK] Slide 4 - Objectives")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 – Dataset & Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Dataset & Preprocessing Pipeline", "From raw profiles to model-ready features")

# Left: 5-step pipeline
steps = [
    ("1", "Raw Data",          "10,000 synthetic customer profiles (raw + cleaned versions)"),
    ("2", "EDA & Cleaning",    "Handle missing values, fix dtypes, remove duplicates"),
    ("3", "Outlier Treatment", "IQR-based clipping on tenure, total spend, income, quantity"),
    ("4", "Feature Encoding",  "OneHotEncoder on: sex, age_group, status, region"),
    ("5", "Stratified Split",  "80% train / 20% test (2,000 samples) — preserving class balance"),
]
sw = Inches(6.1)
sh = Inches(0.92)
sx2 = Inches(0.35)
sy2 = Inches(1.22)
for i, (num, title, desc) in enumerate(steps):
    ty = sy2 + i * (sh + Inches(0.09))
    add_rect(sl, sx2, ty, sw, sh, CARD)
    badge(sl, sx2 + Inches(0.1), ty + Inches(0.24), Inches(0.42), Inches(0.42),
          num, VIVID, WHITE, 13)
    txb(sl, title, sx2 + Inches(0.65), ty + Inches(0.08), sw - Inches(0.75), Inches(0.34),
        font_size=13, bold=True, color=VIVID)
    txb(sl, desc, sx2 + Inches(0.65), ty + Inches(0.46), sw - Inches(0.75), Inches(0.42),
        font_size=11, color=DARK_TEXT)

# Right: features + class balance + tech
rx2 = Inches(6.7)
txb(sl, "Features Used", rx2, Inches(1.22), Inches(6.3), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

feats = [
    ("Categorical", "sex, age_group, status, region"),
    ("Numerical",   "tenure, total spend, income, quantity"),
    ("Target",      "purchased  (0 = Not Purchased, 1 = Purchased)"),
    ("Dataset Size","10,000 rows x 9 columns"),
    ("Class Balance","61.6% purchased=1  /  38.4% purchased=0"),
]
fy = Inches(1.68)
for lbl, val in feats:
    add_rect(sl, rx2, fy, Inches(6.3), Inches(0.52), CARD)
    txb(sl, lbl, rx2 + Inches(0.1), fy + Inches(0.09), Inches(1.8), Inches(0.36),
        font_size=11, bold=True, color=VIVID)
    txb(sl, val, rx2 + Inches(1.95), fy + Inches(0.09), Inches(4.2), Inches(0.36),
        font_size=11, color=DARK_TEXT)
    fy += Inches(0.58)

txb(sl, "Tech Stack", rx2, Inches(4.7), Inches(6.3), Inches(0.36),
    font_size=14, bold=True, color=TITLE)
add_rect(sl, rx2, Inches(5.1), Inches(6.3), Inches(0.65), CARD)
txb(sl, "Python 3.10+  |  Scikit-learn 1.3+  |  Pandas  |  NumPy  |  Matplotlib  |  Streamlit 1.30+",
    rx2 + Inches(0.15), Inches(5.2), Inches(6.0), Inches(0.5),
    font_size=12, color=DARK_TEXT)

slide_num(sl, 5)
print("[OK] Slide 5 - Dataset & Pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 – System Architecture
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "System Architecture & Workflow", "Eight-stage pipeline from raw data to marketing budget allocation")

arch = [
    ("1", "Raw Data",    "10,000 customer\nprofiles ingested"),
    ("2", "EDA &\nClean", "Outlier treatment\ntype fixes"),
    ("3", "Encode",      "OneHotEncoder\non categoricals"),
    ("4", "Split",       "Stratified 80/20\ntrain-test split"),
    ("5", "Train",       "LR, DT,\nRandom Forest"),
    ("6", "Score",       "predict_proba()\nper customer"),
    ("7", "Aggregate",   "Sum P by\nage cohort"),
    ("8", "Allocate",    "Budget &\ncampaign plan"),
]

bw3 = Inches(1.5)
bh3 = Inches(2.5)
gap3 = Inches(0.12)
total = len(arch) * bw3 + (len(arch)-1) * gap3
sx3 = (W - total) / 2
sy3 = Inches(1.85)

for i, (num, title, desc) in enumerate(arch):
    lx = sx3 + i * (bw3 + gap3)
    add_rect(sl, lx, sy3, bw3, bh3, CARD)
    add_rect(sl, lx, sy3, bw3, Inches(0.06), VIVID)
    badge(sl, lx + (bw3 - Inches(0.48))/2, sy3 + Inches(0.12),
          Inches(0.48), Inches(0.48), num, VIVID, WHITE, 14)
    txb(sl, title, lx + Inches(0.06), sy3 + Inches(0.72), bw3 - Inches(0.12), Inches(0.55),
        font_size=11, bold=True, color=VIVID, align=PP_ALIGN.CENTER)
    txb(sl, desc,  lx + Inches(0.06), sy3 + Inches(1.32), bw3 - Inches(0.12), Inches(1.1),
        font_size=10, color=DARK_TEXT, align=PP_ALIGN.CENTER)
    if i < len(arch) - 1:
        ax = lx + bw3 + gap3/2 - Inches(0.08)
        txb(sl, ">", ax, sy3 + bh3/2 - Inches(0.2), Inches(0.18), Inches(0.4),
            font_size=16, bold=True, color=VIVID, align=PP_ALIGN.CENTER)

txb(sl, "Key idea: each customer receives a continuous probability score via predict_proba(); "
        "scores are aggregated across demographic cohorts to produce expected buyer volumes, "
        "enabling data-driven marketing budget allocation.",
    Inches(0.5), Inches(4.55), Inches(12.3), Inches(0.65),
    font_size=12, italic=True, color=ACCENT, align=PP_ALIGN.CENTER)

slide_num(sl, 6)
print("[OK] Slide 6 - System Architecture")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 – EDA (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Exploratory Data Analysis", "Key findings from 10,000 customer records (README Sec. 3)")

embed_image(sl, IMGS["eda"], Inches(0.35), Inches(1.22), Inches(7.8), Inches(5.5))

rx3 = Inches(8.35)
txb(sl, "EDA Highlights", rx3, Inches(1.22), Inches(4.8), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

eda = [
    ("Class Balance",    "61.6% buyers / 38.4% non-buyers — mild imbalance, handled via stratified split"),
    ("Top Spending Age", "26-35 and 36-45 groups show highest cumulative total spend"),
    ("Gender Split",     "Roughly equal Male/Female; both show similar purchase rates (~62%)"),
    ("Loyalty Signal",   "Customers with tenure > 24 months have 18% higher purchase probability"),
    ("Region Effect",    "Minimal regional variation; North & South show marginally higher conversion"),
    ("Income vs Spend",  "Total historical spend outperforms raw income as a predictor"),
]
iy = Inches(1.68)
for hd, bd in eda:
    add_rect(sl, rx3, iy, Inches(4.8), Inches(0.8), CARD)
    add_rect(sl, rx3, iy, Inches(0.06), Inches(0.8), VIVID)
    txb(sl, hd, rx3 + Inches(0.12), iy + Inches(0.06), Inches(4.5), Inches(0.28),
        font_size=11, bold=True, color=VIVID)
    txb(sl, bd, rx3 + Inches(0.12), iy + Inches(0.36), Inches(4.5), Inches(0.38),
        font_size=10, color=DARK_TEXT)
    iy += Inches(0.9)

slide_num(sl, 7)
print("[OK] Slide 7 - EDA")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 – Feature Importance (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Feature Importance — Random Forest", "Behavioral signals outperform raw demographics")

embed_image(sl, IMGS["feat_imp"], Inches(0.35), Inches(1.22), Inches(7.8), Inches(5.5))

rx4 = Inches(8.35)
txb(sl, "Top Signal Insights", rx4, Inches(1.22), Inches(4.8), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

fi = [
    ("Total Spend (#1)",  "Cumulative purchase spend is the strongest predictor — high spenders are most likely to buy."),
    ("Tenure (#2)",       "Long-tenured customers have built trust — powerful repeat-purchase signal."),
    ("Income (#3)",       "Higher income correlates with willingness to pay for premium skincare products."),
    ("Quantity (#4)",     "Past purchase volume reflects habitual buying behaviour and brand engagement."),
    ("Demographics",      "Age group and sex contribute, but behavioral signals dominate — behavior > demographics."),
]
iy2 = Inches(1.68)
for hd, bd in fi:
    add_rect(sl, rx4, iy2, Inches(4.8), Inches(0.88), CARD)
    add_rect(sl, rx4, iy2, Inches(0.06), Inches(0.88), VIVID)
    txb(sl, hd, rx4 + Inches(0.12), iy2 + Inches(0.06), Inches(4.5), Inches(0.28),
        font_size=12, bold=True, color=VIVID)
    txb(sl, bd, rx4 + Inches(0.12), iy2 + Inches(0.38), Inches(4.5), Inches(0.44),
        font_size=10.5, color=DARK_TEXT)
    iy2 += Inches(1.0)

slide_num(sl, 8)
print("[OK] Slide 8 - Feature Importance")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 – Model Performance (README corrected metrics)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Model Performance Comparison",
             "GridSearch (320 candidates, 5-fold CV) | Independent 20% test set (2,000 samples)")

embed_image(sl, IMGS["metrics"], Inches(0.35), Inches(1.22), Inches(7.5), Inches(4.8))

rx5 = Inches(8.05)
headers = ["Model", "Acc", "Prec", "Recall", "F1", "AUC"]
rows = [
    ["Logistic Reg. [Star]", "67.90%", "69.83%", "84.33%", "76.40%", "0.7116"],
    ["Random Forest",        "67.10%", "68.81%", "85.23%", "76.14%", "0.7005"],
    ["Decision Tree",        "66.50%", "68.51%", "84.42%", "75.64%", "0.6941"],
]
col_ws = [Inches(1.75), Inches(0.62), Inches(0.62), Inches(0.72), Inches(0.62), Inches(0.65)]
rh = Inches(0.44)
ty6 = Inches(1.3)

cx = rx5
for j, hdr in enumerate(headers):
    add_rect(sl, cx, ty6, col_ws[j], rh, VIVID)
    txb(sl, hdr, cx + Inches(0.03), ty6 + Inches(0.07), col_ws[j] - Inches(0.04), rh - Inches(0.1),
        font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    cx += col_ws[j]

for ri, row in enumerate(rows):
    ty7 = ty6 + rh + ri * rh
    cx = rx5
    bg = CARD2 if ri == 0 else CARD
    for j, cell in enumerate(row):
        label = cell.replace(" [Star]", "")
        add_rect(sl, cx, ty7, col_ws[j], rh, bg)
        c = VIVID if (ri == 0 and j == 0) else DARK_TEXT
        bd = (ri == 0)
        txb(sl, label, cx + Inches(0.03), ty7 + Inches(0.08), col_ws[j] - Inches(0.04), rh - Inches(0.1),
            font_size=10, bold=bd, color=c, align=PP_ALIGN.CENTER)
        cx += col_ws[j]

ty_note = ty6 + rh * 4 + Inches(0.2)
add_rect(sl, rx5, ty_note, Inches(4.98), Inches(0.42), CARD2)
txb(sl, "Recommended: Logistic Regression (best Accuracy + AUC)",
    rx5 + Inches(0.1), ty_note + Inches(0.06), Inches(4.78), Inches(0.34),
    font_size=11, bold=True, color=VIVID)

txb(sl, "Use RF when maximising buyer capture (highest Recall 85.23%) matters more than precision.\n"
        "Why LR beats RF: total spend & tenure have a smooth monotonic relationship with purchase\n"
        "propensity — a linear sigmoid boundary naturally fits this better than tree splits.",
    rx5, ty_note + Inches(0.5), Inches(4.98), Inches(1.1),
    font_size=10.5, color=DARK_TEXT)

slide_num(sl, 9)
print("[OK] Slide 9 - Model Performance")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 – ROC Curves (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "ROC Curves — All Three Models",
             "AUC measures ranking ability across all classification thresholds")

embed_image(sl, IMGS["roc"], Inches(1.8), Inches(1.22), Inches(9.8), Inches(5.5))

txb(sl, "LR AUC 0.7116 (best)  |  RF AUC 0.7005  |  DT AUC 0.6941  "
        "|  Higher AUC = better at ranking buyers above non-buyers",
    Inches(0.4), Inches(6.88), Inches(12.5), Inches(0.38),
    font_size=11, italic=True, color=ACCENT, align=PP_ALIGN.CENTER)

slide_num(sl, 10)
print("[OK] Slide 10 - ROC Curves")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 – Confusion Matrix (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Confusion Matrix — Random Forest",
             "Breakdown of predictions on the 2,000-record independent test set")

embed_image(sl, IMGS["cm"], Inches(0.35), Inches(1.22), Inches(7.2), Inches(5.5))

rx6 = Inches(7.8)
txb(sl, "Interpreting the Matrix", rx6, Inches(1.22), Inches(5.3), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

cm_items = [
    ("True Positives (TP)",  "Buyers correctly identified — revenue captured for the brand"),
    ("True Negatives (TN)",  "Non-buyers correctly skipped — marketing budget saved"),
    ("False Positives (FP)", "Non-buyers targeted — minor promotional cost, low impact"),
    ("False Negatives (FN)", "Buyers missed — lost revenue opportunity (most costly error)"),
    ("Precision  68.81%",    "Of predicted buyers by RF, 68.81% actually purchased"),
    ("Recall     85.23%",    "Of actual buyers, 85.23% were correctly identified by RF"),
]
iy3 = Inches(1.68)
for hd, bd in cm_items:
    add_rect(sl, rx6, iy3, Inches(5.3), Inches(0.75), CARD)
    add_rect(sl, rx6, iy3, Inches(0.06), Inches(0.75), VIVID)
    txb(sl, hd, rx6 + Inches(0.12), iy3 + Inches(0.04), Inches(5.0), Inches(0.28),
        font_size=11, bold=True, color=VIVID)
    txb(sl, bd, rx6 + Inches(0.12), iy3 + Inches(0.34), Inches(5.0), Inches(0.36),
        font_size=10.5, color=DARK_TEXT)
    iy3 += Inches(0.86)

slide_num(sl, 11)
print("[OK] Slide 11 - Confusion Matrix")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 – Decision Threshold (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Decision Threshold Trade-off",
             "Tuning the probability cutoff balances Precision vs. Recall for campaign strategy")

embed_image(sl, IMGS["threshold"], Inches(0.35), Inches(1.22), Inches(8.4), Inches(5.5))

rx7 = Inches(9.0)
txb(sl, "Threshold Strategy", rx7, Inches(1.22), Inches(4.1), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

thr = [
    ("Default (0.5)",  "Balanced trade-off — used for general benchmarking and reporting."),
    ("Lower (0.35)",   "Boosts Recall; captures more buyers at the cost of more false positives. Best for broad launch campaigns."),
    ("Higher (0.65)",  "Boosts Precision; fewer but more certain targets. Best for premium high-touch campaigns."),
    ("Recommendation", "Use threshold 0.40-0.45 for the skincare launch to maximise reach while keeping CAC reasonable."),
]
iy4 = Inches(1.68)
for hd, bd in thr:
    add_rect(sl, rx7, iy4, Inches(4.1), Inches(1.15), CARD)
    add_rect(sl, rx7, iy4, Inches(0.06), Inches(1.15), VIVID)
    txb(sl, hd, rx7 + Inches(0.12), iy4 + Inches(0.06), Inches(3.85), Inches(0.28),
        font_size=12, bold=True, color=VIVID)
    txb(sl, bd, rx7 + Inches(0.12), iy4 + Inches(0.38), Inches(3.85), Inches(0.72),
        font_size=10.5, color=DARK_TEXT)
    iy4 += Inches(1.25)

slide_num(sl, 12)
print("[OK] Slide 12 - Decision Threshold")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 – Business Findings & Market Sizing (README corrected data)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Business Findings & Market Sizing",
             "Aggregating model-predicted probabilities across 10,000 profiles")

embed_image(sl, IMGS["age_dist"], Inches(0.35), Inches(1.22), Inches(7.5), Inches(4.8))

rx8 = Inches(8.05)
headers2 = ["Age Group", "Cohort", "Exp. Buyers", "Avg Prob", "Share", "Priority"]
rows2 = [
    ["26-35", "~2,000", "~1,380", "~69%", "~28%", "Primary (35%)"],
    ["36-45", "~2,000", "~1,350", "~68%", "~27%", "Secondary (30%)"],
    ["18-25", "~2,000", "~1,230", "~62%", "~25%", "Growth (20%)"],
    ["46-55", "~2,000", "~1,050", "~53%", "~21%", "Niche (10%)"],
    ["56+",   "~2,000", "~950",   "~48%", "~19%", "Re-engage (5%)"],
]
cws = [Inches(0.9), Inches(0.8), Inches(1.0), Inches(0.85), Inches(0.75), Inches(1.2)]
rh2 = Inches(0.42)
ty8 = Inches(1.3)
cx = rx8
for j, hdr in enumerate(headers2):
    add_rect(sl, cx, ty8, cws[j], rh2, VIVID)
    txb(sl, hdr, cx + Inches(0.02), ty8 + Inches(0.06), cws[j] - Inches(0.03), rh2 - Inches(0.08),
        font_size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    cx += cws[j]

for ri, row2 in enumerate(rows2):
    ty9 = ty8 + rh2 + ri * rh2
    cx = rx8
    bg = CARD2 if ri < 2 else CARD
    for j, cell in enumerate(row2):
        add_rect(sl, cx, ty9, cws[j], rh2, bg)
        c = VIVID if (ri < 2 and j == 0) else DARK_TEXT
        txb(sl, cell, cx + Inches(0.02), ty9 + Inches(0.08), cws[j] - Inches(0.03), rh2 - Inches(0.1),
            font_size=9, bold=(ri < 2), color=c, align=PP_ALIGN.CENTER)
        cx += cws[j]

note_y = ty8 + rh2 * 6 + Inches(0.12)
add_rect(sl, rx8, note_y, Inches(5.5), Inches(0.88), CARD2)
txb(sl, "The 26-45 Sweet Spot: Over 55% of all potential buyers sit in the 26-45 demographic.\n"
        "Values are probability-aggregated estimates (sum P) — run Streamlit app for live figures.",
    rx8 + Inches(0.1), note_y + Inches(0.08), Inches(5.3), Inches(0.74),
    font_size=10.5, color=DARK_TEXT)

slide_num(sl, 13)
print("[OK] Slide 13 - Business Findings")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 – Budget Allocation (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Marketing Budget Allocation",
             "Translating model scores into actionable channel-level campaign spend")

embed_image(sl, IMGS["donut"], Inches(0.35), Inches(1.22), Inches(7.8), Inches(5.5))

rx9 = Inches(8.35)
txb(sl, "Allocation Strategy", rx9, Inches(1.22), Inches(4.75), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

budget = [
    ("26-35  ->  35% Budget", "Highest expected buyer volume; prioritise Instagram,\nYouTube and influencer partnerships."),
    ("36-45  ->  30% Budget", "Strong purchasing power; premium placement and\nloyalty offers for this established segment."),
    ("18-25  ->  20% Budget", "Younger males show high propensity; short-form video\nand trending social platform campaigns."),
    ("46-55  ->  10% Budget", "Lower volume but high potential value; targeted email\nand in-store promotions."),
    ("56+    ->   5% Budget", "Selective re-engagement via loyalty programmes\nand referral incentives."),
]
iy5 = Inches(1.68)
for hd, bd in budget:
    add_rect(sl, rx9, iy5, Inches(4.75), Inches(0.98), CARD)
    add_rect(sl, rx9, iy5, Inches(0.06), Inches(0.98), VIVID)
    txb(sl, hd, rx9 + Inches(0.12), iy5 + Inches(0.06), Inches(4.5), Inches(0.28),
        font_size=11, bold=True, color=VIVID)
    txb(sl, bd, rx9 + Inches(0.12), iy5 + Inches(0.38), Inches(4.5), Inches(0.55),
        font_size=10, color=DARK_TEXT)
    iy5 += Inches(1.08)

slide_num(sl, 14)
print("[OK] Slide 14 - Budget Allocation")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 – Streamlit Dashboard (NEW — uses uploaded screenshot)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Interactive Streamlit Dashboard",
             "10-page web application with dark/light mode toggle (app.py)")

# Dashboard screenshot — full width on left
embed_image(sl, IMGS["dashboard"], Inches(0.35), Inches(1.22), Inches(8.3), Inches(5.5))

# Pages list on right
rx10 = Inches(8.9)
txb(sl, "10 App Modules", rx10, Inches(1.22), Inches(4.2), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

pages = [
    ("Dashboard",           "KPIs: 10,000 customers, 6,160 purchased, 61.6% conversion"),
    ("EDA Explorer",        "Interactive distributions, correlation analysis, outlier checks"),
    ("Business Recommend.", "Expected buyers aggregation, demographic share, threshold sim"),
    ("Live Prediction",     "Predict individual customer propensity with probability gauge"),
    ("Model Comparison",    "Side-by-side metrics table + ROC curves for all 3 models"),
    ("Decision Tree",       "Visual tree hierarchy, confusion matrix, ROC curve"),
    ("Random Forest",       "Feature importances, confusion matrix, ROC curve"),
    ("Logistic Regression", "Confusion matrix, ROC curve with AUC fill"),
    ("Ethics & Insights",   "Responsible AI considerations, demographic fairness analysis"),
    ("Project Files",       "In-app code and dataset explorer with download capability"),
]
iy6 = Inches(1.68)
for pg, desc in pages:
    add_rect(sl, rx10, iy6, Inches(4.25), Inches(0.48), CARD)
    txb(sl, pg, rx10 + Inches(0.1), iy6 + Inches(0.04), Inches(1.65), Inches(0.32),
        font_size=10, bold=True, color=VIVID)
    txb(sl, desc, rx10 + Inches(1.78), iy6 + Inches(0.04), Inches(2.38), Inches(0.38),
        font_size=9, color=DARK_TEXT)
    iy6 += Inches(0.54)

slide_num(sl, 15)
print("[OK] Slide 15 - Streamlit Dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 16 – Key Insights & Responsible AI
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
slide_header(sl, "Key Insights & Responsible AI",
             "What the model reveals — and how to use it ethically (README Sec. 6 & 7)")

# Left col: insights
txb(sl, "Customer Insights", Inches(0.35), Inches(1.22), Inches(6.0), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

insights = [
    ("Behavioral > Demographic",  "Total spend & tenure outperform age/gender in predictive power (feature importances confirm this)."),
    ("Younger Males (18-35)",      "Show disproportionately high purchase propensity — strong early-adopter signal for unisex products."),
    ("Retention = Revenue",        "Customers with 2+ years tenure are 18% more likely to convert on new product launches."),
    ("Probability, Not Certainty", "Scores indicate relative likelihood — treat demographic analysis as marketing analytics."),
]
iy7 = Inches(1.68)
for hd, bd in insights:
    add_rect(sl, Inches(0.35), iy7, Inches(6.1), Inches(1.0), CARD)
    add_rect(sl, Inches(0.35), iy7, Inches(0.06), Inches(1.0), VIVID)
    txb(sl, hd, Inches(0.47), iy7 + Inches(0.07), Inches(5.8), Inches(0.28),
        font_size=12, bold=True, color=VIVID)
    txb(sl, bd, Inches(0.47), iy7 + Inches(0.4), Inches(5.8), Inches(0.54),
        font_size=11, color=DARK_TEXT)
    iy7 += Inches(1.1)

# Right col: Responsible AI
rx11 = Inches(6.75)
txb(sl, "Responsible AI Principles", rx11, Inches(1.22), Inches(6.3), Inches(0.38),
    font_size=14, bold=True, color=TITLE)

rai = [
    ("Privacy First",       "All customer attributes processed under data privacy governance (GDPR / DPDP compliant)."),
    ("No Stereotyping",     "Models evaluate purchase propensity based on transaction history — not normative claims about demographics."),
    ("Human Judgment",      "A score supports decisions; it does not replace human accountability or override business judgment."),
    ("Monitor Drift",       "Models evaluated for demographic fairness — no algorithmic redlining of regions or communities. Re-evaluate quarterly."),
]
iy8 = Inches(1.68)
for hd, bd in rai:
    add_rect(sl, rx11, iy8, Inches(6.25), Inches(1.0), CARD)
    add_rect(sl, rx11, iy8, Inches(0.06), Inches(1.0), VIVID)
    txb(sl, hd, rx11 + Inches(0.12), iy8 + Inches(0.07), Inches(5.95), Inches(0.28),
        font_size=12, bold=True, color=VIVID)
    txb(sl, bd, rx11 + Inches(0.12), iy8 + Inches(0.4), Inches(5.95), Inches(0.54),
        font_size=11, color=DARK_TEXT)
    iy8 += Inches(1.1)

slide_num(sl, 16)
print("[OK] Slide 16 - Insights & Ethics")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 17 – Conclusion & Thank You
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
lavender_bg(sl)
top_bar(sl, Inches(0.12))
bottom_bar(sl, Inches(0.12))

# Large white card
add_rect(sl, Inches(0.35), Inches(0.22), Inches(12.6), Inches(7.0), CARD)

txb(sl, "From Predictions to Smarter Skincare Marketing",
    Inches(0.6), Inches(0.4), Inches(12.1), Inches(0.68),
    font_size=26, bold=True, color=TITLE)
add_rect(sl, Inches(0.6), Inches(1.1), Inches(8.0), Inches(0.04), VIVID)

# Stats row
stats = [
    ("10,000",  "Customer Profiles"),
    ("3",       "ML Models"),
    ("67.90%",  "Best Accuracy (LR)"),
    ("0.7116",  "Best ROC-AUC (LR)"),
    ("85.23%",  "Best Recall (RF)"),
    ("10",      "Streamlit Pages"),
]
bsw = Inches(1.98)
bsh = Inches(1.22)
bsx = Inches(0.55)
bsy = Inches(1.22)
for i, (val, lbl) in enumerate(stats):
    lx = bsx + i * (bsw + Inches(0.1))
    add_rect(sl, lx, bsy, bsw, bsh, CARD2)
    add_rect(sl, lx, bsy, bsw, Inches(0.06), VIVID)
    txb(sl, val, lx, bsy + Inches(0.1), bsw, Inches(0.58),
        font_size=24, bold=True, color=VIVID, align=PP_ALIGN.CENTER)
    txb(sl, lbl, lx, bsy + Inches(0.72), bsw, Inches(0.44),
        font_size=10, color=ACCENT, align=PP_ALIGN.CENTER)

# Future scope
txb(sl, "Future Scope",
    Inches(0.6), Inches(2.62), Inches(12.0), Inches(0.36),
    font_size=14, bold=True, color=TITLE)
add_rect(sl, Inches(0.55), Inches(3.04), Inches(12.25), Inches(0.7), CARD2)
txb(sl, "Gradient Boosting (XGBoost / LightGBM)  |  Optimal threshold tuning per cohort  |  "
        "Concept drift monitoring  |  Richer features (browsing, recency)  |  Mobile-ready Streamlit  |  "
        "SHAP explainability  |  A/B test integration",
    Inches(0.65), Inches(3.1), Inches(12.05), Inches(0.6),
    font_size=11.5, color=DARK_TEXT)

# Thank you
txb(sl, "Thank You",
    Inches(0.55), Inches(3.92), Inches(12.3), Inches(0.85),
    font_size=42, bold=True, color=TITLE, align=PP_ALIGN.CENTER)
txb(sl, "Questions & Discussion",
    Inches(0.55), Inches(4.82), Inches(12.3), Inches(0.42),
    font_size=18, italic=True, color=ACCENT, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(2.5), Inches(5.38), Inches(8.3), Inches(0.04), MED)

txb(sl, "Tushar Jaiswal  |  Shatakshi Jaiswal  |  Sharafat  |  Garvit  |  Tanisha  |  Tanishq Saini  |  Astha  |  Mahima Kalra",
    Inches(0.55), Inches(5.52), Inches(12.3), Inches(0.36),
    font_size=11, color=DARK_TEXT, align=PP_ALIGN.CENTER)
txb(sl, "Department of Computer Science & Engineering  |  B.Tech CSE",
    Inches(0.55), Inches(5.92), Inches(12.3), Inches(0.32),
    font_size=10, color=LIGHT_TXT, align=PP_ALIGN.CENTER)
txb(sl, "github.com/tushar032004/AI-Cosmetics-Purchase-Prediction",
    Inches(0.55), Inches(6.3), Inches(12.3), Inches(0.32),
    font_size=10, color=VIVID, align=PP_ALIGN.CENTER)

slide_num(sl, 17)
print("[OK] Slide 17 - Conclusion & Thank You")

# ── Save ──────────────────────────────────────────────────────────────────────
prs.save(str(OUT))
print(f"\n[DONE] Saved: {OUT}")
print(f"       Total slides : {len(prs.slides)}")
print(f"       File size    : {os.path.getsize(OUT):,} bytes")
