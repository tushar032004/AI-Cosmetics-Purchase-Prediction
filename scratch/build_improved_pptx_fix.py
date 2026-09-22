"""
Build Card-1-Improved.pptx
AI-based Customer Purchase Prediction for Unisex Skincare
16 slides | Navy + Coral theme | Embeds all 8 visualisation images
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.oxml.ns import qn
from lxml import etree
import copy
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\AI_Cosmetic_Purchase_Prediction")
VIZ  = BASE / "visualisations"
OUT  = BASE / "Card-1-Improved.pptx"

IMGS = {
    "eda":        VIZ / "eda_age_and_gender.png",
    "feat_imp":   VIZ / "feature_importance.png",
    "metrics":    VIZ / "model_metrics_comparison.png",
    "roc":        VIZ / "model_roc_curves.png",
    "cm":         VIZ / "confusion_matrix_random_forest.png",
    "threshold":  VIZ / "decision_threshold_tradeoff.png",
    "age_dist":   VIZ / "age_group_buyer_distribution.png",
    "donut":      VIZ / "budget_allocation_donut.png",
}

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY      = RGBColor(0x0D, 0x1B, 0x2A)   # deep navy background
CORAL     = RGBColor(0xE6, 0x39, 0x46)   # accent / highlight
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT     = RGBColor(0xE8, 0xF0, 0xFE)   # very light blue-white for body text
GOLD      = RGBColor(0xFF, 0xC3, 0x00)   # star accent
MID_NAVY  = RGBColor(0x1B, 0x2A, 0x4A)  # slightly lighter navy for boxes
LIGHT_GREY= RGBColor(0xCC, 0xD6, 0xE8)

# Slide dimensions – Widescreen 13.33 × 7.5 in
W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

BLANK = prs.slide_layouts[6]   # completely blank layout

# ── Helper utilities ──────────────────────────────────────────────────────────

def add_rect(slide, l, t, w, h, fill_rgb, alpha=None):
    shape = slide.shapes.add_shape(1, l, t, w, h)
    shape.line.fill.background()
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    return shape


def txb(slide, text, l, t, w, h,
        font_size=18, bold=False, italic=False,
        color=WHITE, align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size   = Pt(font_size)
    run.font.bold   = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def add_divider(slide, t, color=CORAL, w_pct=0.6):
    bar_w = int(W * w_pct)
    add_rect(slide, int((W - bar_w) / 2), t, bar_w, Inches(0.04), color)


def navy_bg(slide):
    add_rect(slide, 0, 0, W, H, NAVY)


def slide_header(slide, title, subtitle=None, title_y=Inches(0.35)):
    # Coral accent bar at top
    add_rect(slide, 0, 0, W, Inches(0.08), CORAL)
    # Title
    txb(slide, title,
        Inches(0.6), title_y, Inches(12), Inches(0.7),
        font_size=36, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    # Subtitle
    if subtitle:
        txb(slide, subtitle,
            Inches(0.6), title_y + Inches(0.65), Inches(11), Inches(0.45),
            font_size=16, color=LIGHT, align=PP_ALIGN.LEFT)
    # Bottom bar
    add_rect(slide, 0, H - Inches(0.08), W, Inches(0.08), CORAL)


def slide_number(slide, num):
    txb(slide, str(num),
        W - Inches(0.65), H - Inches(0.4), Inches(0.5), Inches(0.35),
        font_size=11, color=LIGHT, align=PP_ALIGN.RIGHT)


def info_box(slide, l, t, w, h, heading, body_lines, hd_size=15, bd_size=12):
    add_rect(slide, l, t, w, h, MID_NAVY)
    # Coral left stripe
    add_rect(slide, l, t, Inches(0.08), h, CORAL)
    txb(slide, heading,
        l + Inches(0.15), t + Inches(0.1), w - Inches(0.2), Inches(0.35),
        font_size=hd_size, bold=True, color=CORAL)
    body = "\n".join(body_lines)
    txb(slide, body,
        l + Inches(0.15), t + Inches(0.42), w - Inches(0.2), h - Inches(0.5),
        font_size=bd_size, color=LIGHT)


def embed_image(slide, path, l, t, w, h):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), l, t, w, h)
    else:
        add_rect(slide, l, t, w, h, MID_NAVY)
        txb(slide, f"[Image: {Path(path).name}]",
            l, t + h/2 - Inches(0.2), w, Inches(0.4),
            font_size=11, color=LIGHT_GREY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 – Title
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)

# Decorative coral stripe left
add_rect(sl, 0, 0, Inches(0.5), H, CORAL)
# Mid accent block
add_rect(sl, Inches(0.5), Inches(2.4), Inches(12.3), Inches(0.06), CORAL)
add_rect(sl, Inches(0.5), Inches(5.1), Inches(12.3), Inches(0.06), CORAL)

txb(sl, "MACHINE LEARNING PROJECT",
    Inches(0.9), Inches(1.3), Inches(12), Inches(0.6),
    font_size=18, bold=True, color=CORAL, align=PP_ALIGN.LEFT)

txb(sl, "AI-Based Customer Purchase\nPrediction for Unisex Skincare",
    Inches(0.9), Inches(2.55), Inches(11.5), Inches(2.0),
    font_size=40, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

txb(sl, "Predicting purchase propensity • Segmenting high-value cohorts • Driving smarter budget allocation",
    Inches(0.9), Inches(4.65), Inches(11.2), Inches(0.5),
    font_size=14, italic=True, color=LIGHT, align=PP_ALIGN.LEFT)

txb(sl, "Presented by: Tushar Jaiswal • Shatakshi Jaiswal • Sharafat • Garvit • Tanisha • Tanishq Saini • Aastha • Mahima Kalra",
    Inches(0.9), Inches(5.5), Inches(11.5), Inches(0.45),
    font_size=12, color=LIGHT_GREY, align=PP_ALIGN.LEFT)

txb(sl, "Department of Computer Science & Engineering   |   B.Tech CSE",
    Inches(0.9), Inches(5.95), Inches(11.5), Inches(0.38),
    font_size=11, color=LIGHT_GREY, align=PP_ALIGN.LEFT)

slide_number(sl, 1)
print("✔ Slide 1 – Title")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 – Agenda
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "What We'll Cover", "End-to-end walkthrough of the project")

agenda = [
    ("01", "Problem Statement",    "Why targeted marketing matters for a new skincare launch"),
    ("02", "Project Objectives",   "What the project set out to predict, compare, and deliver"),
    ("03", "Dataset & Pipeline",   "Cleaning, feature preparation, and the stratified split"),
    ("04", "EDA",                  "Exploratory insights into age, gender, and spend patterns"),
    ("05", "Feature Importance",   "Which signals drive purchase propensity the most"),
    ("06", "Model Comparison",     "Logistic Regression, Decision Tree, and Random Forest"),
    ("07", "ROC & Confusion Matrix","Detailed model evaluation including threshold trade-offs"),
    ("08", "Business Findings",    "Cohort demand sizing and budget allocation strategy"),
    ("09", "Key Insights & Ethics","Responsible AI usage of purchase propensity scores"),
]

cols = 3
box_w = Inches(4.1)
box_h = Inches(1.0)
gap_x = Inches(0.2)
gap_y = Inches(0.18)
start_x = Inches(0.25)
start_y = Inches(1.55)

for i, (num, title, desc) in enumerate(agenda):
    col = i % cols
    row = i // cols
    lx = start_x + col * (box_w + gap_x)
    ty = start_y + row * (box_h + gap_y)
    add_rect(sl, lx, ty, box_w, box_h, MID_NAVY)
    add_rect(sl, lx, ty, Inches(0.08), box_h, CORAL)
    txb(sl, num, lx + Inches(0.14), ty + Inches(0.08), Inches(0.5), Inches(0.35),
        font_size=20, bold=True, color=CORAL)
    txb(sl, title, lx + Inches(0.14), ty + Inches(0.36), box_w - Inches(0.2), Inches(0.28),
        font_size=13, bold=True, color=WHITE)
    txb(sl, desc,  lx + Inches(0.14), ty + Inches(0.64), box_w - Inches(0.2), Inches(0.34),
        font_size=10, color=LIGHT_GREY)

slide_number(sl, 2)
print("✔ Slide 2 – Agenda")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 – Problem Statement
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Problem Statement", "The business challenge behind this ML project")

# Central problem statement box
add_rect(sl, Inches(0.5), Inches(1.55), Inches(12.3), Inches(1.35), MID_NAVY)
add_rect(sl, Inches(0.5), Inches(1.55), Inches(0.1), Inches(1.35), CORAL)
txb(sl, "A cosmetics brand is launching an innovative unisex skincare product.\n"
        "Untargeted mass marketing drives up Customer Acquisition Cost (CAC) while delivering low conversion rates.\n"
        "Businesses need a practical, data-driven way to identify customers genuinely likely to buy — before allocating campaign budget.",
    Inches(0.72), Inches(1.65), Inches(11.9), Inches(1.2),
    font_size=14, color=WHITE)

# Three pain point boxes
pain = [
    ("💸  High CAC",          "Mass campaigns reach uninterested audiences,\nwasting budget on low-intent customers."),
    ("📉  Low Conversion",    "Without targeting, typical conversion is 5–15%.\nPredictive targeting can exceed 30–40%."),
    ("❓  Who to Reach First?","Marketing needs a ranked list of prospects\nbefore the product launch window closes."),
]
bx_w = Inches(3.9)
bx_h = Inches(2.3)
by   = Inches(3.15)
for i,(hd, bd) in enumerate(pain):
    lx = Inches(0.4) + i * (bx_w + Inches(0.27))
    add_rect(sl, lx, by, bx_w, bx_h, MID_NAVY)
    add_rect(sl, lx, by, Inches(0.08), bx_h, CORAL)
    txb(sl, hd, lx + Inches(0.14), by + Inches(0.12), bx_w - Inches(0.2), Inches(0.4),
        font_size=14, bold=True, color=CORAL)
    txb(sl, bd, lx + Inches(0.14), by + Inches(0.55), bx_w - Inches(0.2), Inches(1.65),
        font_size=12, color=LIGHT)

txb(sl, "The question: who should marketing reach first?",
    Inches(0.5), Inches(5.6), Inches(12), Inches(0.5),
    font_size=16, bold=True, italic=True, color=CORAL, align=PP_ALIGN.CENTER)

slide_number(sl, 3)
print("✔ Slide 3 – Problem Statement")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 – Project Objectives
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Project Objectives", "Four pillars of this end-to-end ML solution")

objectives = [
    ("🎯", "Purchase Prediction",
     "Estimate the probability that an individual customer will buy the new unisex skincare product using behavioral and demographic signals."),
    ("👥", "Customer Segmentation",
     "Analyse demand across age brackets and demographic cohorts to surface the highest-value customer groups for targeting."),
    ("⚖️", "Model Comparison",
     "Benchmark Logistic Regression, Decision Tree, and Random Forest across Accuracy, F1, Recall, and ROC-AUC to select the best performer."),
    ("📊", "Business Intelligence App",
     "Deliver results via an interactive Streamlit dashboard — converting raw model scores into expected buyers and budget allocation insights."),
]

bx_w = Inches(5.95)
bx_h = Inches(2.2)
gap  = Inches(0.25)
sy   = Inches(1.6)

for i, (icon, title, desc) in enumerate(objectives):
    col = i % 2
    row = i // 2
    lx = Inches(0.35) + col * (bx_w + gap)
    ty = sy + row * (bx_h + Inches(0.2))
    add_rect(sl, lx, ty, bx_w, bx_h, MID_NAVY)
    add_rect(sl, lx, ty, Inches(0.08), bx_h, CORAL)
    txb(sl, icon,  lx + Inches(0.18), ty + Inches(0.12), Inches(0.55), Inches(0.5),
        font_size=22, color=WHITE)
    txb(sl, title, lx + Inches(0.75), ty + Inches(0.12), bx_w - Inches(0.85), Inches(0.4),
        font_size=15, bold=True, color=CORAL)
    txb(sl, desc,  lx + Inches(0.18), ty + Inches(0.6),  bx_w - Inches(0.3), Inches(1.5),
        font_size=12, color=LIGHT)

slide_number(sl, 4)
print("✔ Slide 4 – Objectives")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 – Dataset & Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Dataset & Preprocessing Pipeline", "From raw profiles to model-ready features")

# Left column: pipeline steps
steps = [
    ("1", "Raw Data",          "10,000 synthetic customer profiles (raw + cleaned versions)"),
    ("2", "Data Cleaning",     "Handle missing values, fix dtypes, remove duplicates"),
    ("3", "Outlier Treatment", "IQR-based clipping on tenure, total spend, income, quantity"),
    ("4", "Feature Encoding",  "OneHotEncoder on: sex, age_group, status, region"),
    ("5", "Stratified Split",  "80% train / 20% test — preserving class balance"),
]

sx = Inches(0.4)
sy2 = Inches(1.55)
s_h = Inches(0.9)
s_w = Inches(6.0)
for i, (num, title, desc) in enumerate(steps):
    ty = sy2 + i * (s_h + Inches(0.08))
    add_rect(sl, sx, ty, s_w, s_h, MID_NAVY)
    # Step circle
    add_rect(sl, sx, ty + Inches(0.22), Inches(0.42), Inches(0.42), CORAL)
    txb(sl, num, sx + Inches(0.08), ty + Inches(0.22), Inches(0.42), Inches(0.42),
        font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txb(sl, title, sx + Inches(0.55), ty + Inches(0.08), s_w - Inches(0.65), Inches(0.35),
        font_size=13, bold=True, color=CORAL)
    txb(sl, desc,  sx + Inches(0.55), ty + Inches(0.44), s_w - Inches(0.65), Inches(0.42),
        font_size=11, color=LIGHT)

# Right column: feature table + tech stack
rx = Inches(6.7)
txb(sl, "Features Used", rx, Inches(1.55), Inches(6.3), Inches(0.4),
    font_size=15, bold=True, color=CORAL)

features = [
    ("Categorical", "sex, age_group, status, region"),
    ("Numerical",   "tenure, total spend, income, quantity"),
    ("Target",      "purchased  (0 = No, 1 = Yes)"),
    ("Dataset Size","10,000 rows × 9 columns"),
    ("Class Balance","~77% purchased=1  /  ~23% purchased=0"),
]
fy = Inches(2.0)
for lbl, val in features:
    add_rect(sl, rx, fy, Inches(6.3), Inches(0.52), MID_NAVY)
    txb(sl, lbl, rx + Inches(0.1), fy + Inches(0.08), Inches(2.0), Inches(0.38),
        font_size=11, bold=True, color=CORAL)
    txb(sl, val, rx + Inches(2.1), fy + Inches(0.08), Inches(4.0), Inches(0.38),
        font_size=11, color=WHITE)
    fy += Inches(0.58)

txb(sl, "Tech Stack", rx, Inches(5.1), Inches(6.3), Inches(0.4),
    font_size=15, bold=True, color=CORAL)
txb(sl, "Python 3.10+  •  Scikit-learn  •  Pandas  •  NumPy  •  Matplotlib  •  Streamlit",
    rx, Inches(5.52), Inches(6.3), Inches(0.4),
    font_size=12, color=LIGHT)

slide_number(sl, 5)
print("✔ Slide 5 – Dataset & Pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 – System Architecture
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "System Architecture & Workflow", "Six-stage pipeline from raw data to marketing action")

arch_steps = [
    ("1", "Raw Data",    "10,000 customer\nprofiles ingested"),
    ("2", "Prepare",     "Clean, outlier\ntreatment, encode"),
    ("3", "Split",       "Stratified 80/20\ntrain-test split"),
    ("4", "Train",       "Logistic Reg.\nDecision Tree\nRandom Forest"),
    ("5", "Score",       "predict_proba()\nfor each customer"),
    ("6", "Activate",    "Aggregate cohort\ndemand → budget"),
]

bx_w = Inches(1.9)
bx_h = Inches(2.6)
gap  = Inches(0.22)
total_w = len(arch_steps) * bx_w + (len(arch_steps)-1) * gap
sx2 = (W - total_w) / 2
sy3 = Inches(2.0)

for i, (num, title, desc) in enumerate(arch_steps):
    lx = sx2 + i * (bx_w + gap)
    add_rect(sl, lx, sy3, bx_w, bx_h, MID_NAVY)
    # Number badge
    add_rect(sl, lx + (bx_w - Inches(0.55))/2, sy3 + Inches(0.15), Inches(0.55), Inches(0.55), CORAL)
    txb(sl, num,
        lx + (bx_w - Inches(0.55))/2, sy3 + Inches(0.15), Inches(0.55), Inches(0.55),
        font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txb(sl, title, lx + Inches(0.1), sy3 + Inches(0.82), bx_w - Inches(0.2), Inches(0.45),
        font_size=13, bold=True, color=CORAL, align=PP_ALIGN.CENTER)
    txb(sl, desc,  lx + Inches(0.1), sy3 + Inches(1.32), bx_w - Inches(0.2), Inches(1.15),
        font_size=11, color=LIGHT, align=PP_ALIGN.CENTER)
    # Arrow between boxes
    if i < len(arch_steps) - 1:
        ax = lx + bx_w + Inches(0.05)
        txb(sl, "→", ax, sy3 + bx_h/2 - Inches(0.22), Inches(0.18), Inches(0.45),
            font_size=18, bold=True, color=CORAL, align=PP_ALIGN.CENTER)

txb(sl, "Key idea: each customer receives a probability score via predict_proba(); "
        "scores are aggregated across demographic cohorts to produce expected buyer volumes "
        "and guide marketing budget allocation.",
    Inches(0.5), Inches(5.0), Inches(12.3), Inches(0.7),
    font_size=13, italic=True, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

slide_number(sl, 6)
print("✔ Slide 6 – System Architecture")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 – EDA (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Exploratory Data Analysis", "Understanding age, gender, and spend distribution across the dataset")

embed_image(sl, IMGS["eda"], Inches(0.4), Inches(1.5), Inches(7.8), Inches(5.4))

# Insights panel
rx = Inches(8.4)
txb(sl, "Key EDA Findings", rx, Inches(1.55), Inches(4.7), Inches(0.42),
    font_size=15, bold=True, color=CORAL)

eda_insights = [
    ("Age Distribution",   "26–45 bracket is the most populated segment with highest purchase volumes."),
    ("Gender Split",       "Dataset is near-balanced across male/female, enabling unisex product analysis."),
    ("Spend Patterns",     "Higher cumulative spend (total) strongly correlates with purchase likelihood."),
    ("Loyalty Signal",     "Customers with tenure > 24 months show 20%+ higher purchase rates."),
    ("Regional Spread",    "Purchase rates are consistent across regions — no dominant geographic bias."),
]
iy = Inches(2.1)
for hd, bd in eda_insights:
    add_rect(sl, rx, iy, Inches(4.7), Inches(0.88), MID_NAVY)
    add_rect(sl, rx, iy, Inches(0.07), Inches(0.88), CORAL)
    txb(sl, hd, rx + Inches(0.14), iy + Inches(0.05), Inches(4.4), Inches(0.3),
        font_size=12, bold=True, color=CORAL)
    txb(sl, bd, rx + Inches(0.14), iy + Inches(0.36), Inches(4.4), Inches(0.46),
        font_size=10.5, color=LIGHT)
    iy += Inches(1.0)

slide_number(sl, 7)
print("✔ Slide 7 – EDA")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 – Feature Importance (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Feature Importance", "What drives purchase propensity — ranked by Random Forest")

embed_image(sl, IMGS["feat_imp"], Inches(0.4), Inches(1.5), Inches(7.8), Inches(5.4))

rx = Inches(8.4)
txb(sl, "Top Signal Insights", rx, Inches(1.55), Inches(4.7), Inches(0.42),
    font_size=15, bold=True, color=CORAL)

fi_insights = [
    ("💳  Total Spend",     "Cumulative purchase spend is the #1 predictor — high spenders are most likely to buy."),
    ("⏳  Tenure",          "Long-tenured customers have established trust — strong repeat-purchase signal."),
    ("💰  Income",          "Higher income correlates with willingness to pay for premium skincare."),
    ("📦  Quantity",        "Past purchase volume reflects habitual buying behaviour."),
    ("👤  Demographics",    "Age group and gender contribute, but behavioral signals dominate."),
]
iy = Inches(2.1)
for hd, bd in fi_insights:
    add_rect(sl, rx, iy, Inches(4.7), Inches(0.9), MID_NAVY)
    add_rect(sl, rx, iy, Inches(0.07), Inches(0.9), CORAL)
    txb(sl, hd, rx + Inches(0.14), iy + Inches(0.06), Inches(4.4), Inches(0.32),
        font_size=12, bold=True, color=CORAL)
    txb(sl, bd, rx + Inches(0.14), iy + Inches(0.4), Inches(4.4), Inches(0.46),
        font_size=10.5, color=LIGHT)
    iy += Inches(1.0)

slide_number(sl, 8)
print("✔ Slide 8 – Feature Importance")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 – Model Performance Comparison
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Model Performance Comparison", "Three classifiers benchmarked on an independent 20% test set")

embed_image(sl, IMGS["metrics"], Inches(0.4), Inches(1.5), Inches(7.5), Inches(4.8))

# Table on the right
rx = Inches(8.1)
headers = ["Model", "Acc", "Prec", "Recall", "F1", "AUC"]
rows = [
    ["Decision Tree",   "76.5%", "82.4%", "89.1%", "85.6%", "0.778"],
    ["Logistic Reg.",   "78.2%", "81.9%", "92.8%", "87.0%", "0.804"],
    ["Random Forest ✓", "78.9%", "83.1%", "91.8%", "87.2%", "0.821"],
]
col_ws = [Inches(1.7), Inches(0.6), Inches(0.6), Inches(0.7), Inches(0.6), Inches(0.65)]
row_h = Inches(0.42)
ty2 = Inches(1.55)

# Header row
cx = rx
for j, hdr in enumerate(headers):
    add_rect(sl, cx, ty2, col_ws[j], row_h, CORAL)
    txb(sl, hdr, cx + Inches(0.04), ty2 + Inches(0.06), col_ws[j] - Inches(0.05), row_h - Inches(0.08),
        font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    cx += col_ws[j]

for ri, row in enumerate(rows):
    ty3 = ty2 + row_h + ri * row_h
    cx  = rx
    bg  = RGBColor(0x18, 0x2A, 0x50) if ri == 2 else MID_NAVY
    for j, cell in enumerate(row):
        add_rect(sl, cx, ty3, col_ws[j], row_h, bg)
        c = CORAL if (ri == 2 and j == 0) else (WHITE if j == 0 else LIGHT)
        bd = (ri == 2)
        txb(sl, cell, cx + Inches(0.04), ty3 + Inches(0.07), col_ws[j] - Inches(0.05), row_h - Inches(0.1),
            font_size=10, bold=bd, color=c, align=PP_ALIGN.CENTER)
        cx += col_ws[j]

ty_note = ty2 + row_h * 4 + Inches(0.2)
txb(sl, "★  Random Forest is the best overall model",
    rx, ty_note, Inches(4.9), Inches(0.38),
    font_size=12, bold=True, color=CORAL)

txb(sl, "Why Recall matters: a False Negative (missed buyer) costs\n~₹2,800 in lost margin. A False Positive (extra promo) costs pennies.\nHigh Recall (>91%) ensures maximum market capture.",
    rx, ty_note + Inches(0.44), Inches(4.9), Inches(1.1),
    font_size=11, color=LIGHT)

slide_number(sl, 9)
print("✔ Slide 9 – Model Performance")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 – ROC Curves (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "ROC Curves — All Three Models", "Area Under Curve (AUC) measures ranking ability across all thresholds")

embed_image(sl, IMGS["roc"], Inches(1.5), Inches(1.5), Inches(10.3), Inches(5.3))

txb(sl, "Random Forest AUC 0.821  >  Logistic Regression 0.804  >  Decision Tree 0.778  |  Higher AUC = better at ranking buyers above non-buyers",
    Inches(0.4), Inches(6.9), Inches(12.5), Inches(0.38),
    font_size=11, italic=True, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

slide_number(sl, 10)
print("✔ Slide 10 – ROC Curves")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 – Confusion Matrix (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Confusion Matrix — Random Forest", "Breakdown of predictions on the 2,000-record test set")

embed_image(sl, IMGS["cm"], Inches(0.4), Inches(1.5), Inches(7.2), Inches(5.4))

rx = Inches(7.8)
txb(sl, "Interpreting the Matrix", rx, Inches(1.55), Inches(5.3), Inches(0.42),
    font_size=15, bold=True, color=CORAL)

cm_items = [
    ("True Positives (TP)",  "Buyers correctly identified → revenue captured"),
    ("True Negatives (TN)",  "Non-buyers correctly skipped → budget saved"),
    ("False Positives (FP)", "Non-buyers targeted → minor promotional cost"),
    ("False Negatives (FN)", "Buyers missed → lost revenue (most costly error)"),
    ("Precision  83.1%",     "Of predicted buyers, 83% actually purchased"),
    ("Recall     91.8%",     "Of actual buyers, 92% were correctly found"),
]
iy2 = Inches(2.1)
for hd, bd in cm_items:
    add_rect(sl, rx, iy2, Inches(5.3), Inches(0.76), MID_NAVY)
    add_rect(sl, rx, iy2, Inches(0.07), Inches(0.76), CORAL)
    txb(sl, hd, rx + Inches(0.14), iy2 + Inches(0.04), Inches(5.0), Inches(0.28),
        font_size=11, bold=True, color=CORAL)
    txb(sl, bd, rx + Inches(0.14), iy2 + Inches(0.35), Inches(5.0), Inches(0.36),
        font_size=10.5, color=LIGHT)
    iy2 += Inches(0.86)

slide_number(sl, 11)
print("✔ Slide 11 – Confusion Matrix")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 – Decision Threshold Trade-off (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Decision Threshold Trade-off", "Tuning the probability cutoff balances Precision vs. Recall")

embed_image(sl, IMGS["threshold"], Inches(0.4), Inches(1.5), Inches(8.4), Inches(5.4))

rx = Inches(9.0)
txb(sl, "Threshold Strategy", rx, Inches(1.55), Inches(4.1), Inches(0.42),
    font_size=15, bold=True, color=CORAL)

thr_items = [
    ("Default (0.5)",  "Balanced trade-off — used for general benchmarking."),
    ("Lower (0.35)",   "Boosts Recall; captures more buyers at the cost of more false positives. Best for broad campaigns."),
    ("Higher (0.65)",  "Boosts Precision; fewer but more certain targets. Best for premium/high-touch campaigns."),
    ("Recommendation", "Use threshold 0.40–0.45 for the skincare launch to maximise reach while keeping CAC reasonable."),
]
iy3 = Inches(2.1)
for hd, bd in thr_items:
    add_rect(sl, rx, iy3, Inches(4.1), Inches(1.12), MID_NAVY)
    add_rect(sl, rx, iy3, Inches(0.07), Inches(1.12), CORAL)
    txb(sl, hd, rx + Inches(0.14), iy3 + Inches(0.06), Inches(3.8), Inches(0.3),
        font_size=12, bold=True, color=CORAL)
    txb(sl, bd, rx + Inches(0.14), iy3 + Inches(0.4), Inches(3.8), Inches(0.68),
        font_size=10.5, color=LIGHT)
    iy3 += Inches(1.22)

slide_number(sl, 12)
print("✔ Slide 12 – Decision Threshold")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 – Business Findings & Market Sizing
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Business Findings & Market Sizing", "Aggregating purchase probabilities across demographic cohorts")

embed_image(sl, IMGS["age_dist"], Inches(0.4), Inches(1.5), Inches(7.5), Inches(5.0))

# Table
rx = Inches(8.1)
headers2 = ["Age Group", "Cohort", "Exp. Buyers", "Avg Prob", "Share"]
rows2 = [
    ["26–35", "3,400", "2,650", "77.9%", "37.5%"],
    ["36–45", "3,200", "2,480", "77.5%", "35.1%"],
    ["18–25", "1,800", "1,390", "77.2%", "19.7%"],
    ["46–55", "1,100",   "840", "76.4%", "11.9%"],
    ["56+",     "500",   "380", "76.0%",  "5.4%"],
]
col_ws2 = [Inches(1.1), Inches(0.82), Inches(1.05), Inches(0.95), Inches(0.85)]
row_h2  = Inches(0.44)
ty4 = Inches(1.55)
cx  = rx
for j, hdr in enumerate(headers2):
    add_rect(sl, cx, ty4, col_ws2[j], row_h2, CORAL)
    txb(sl, hdr, cx + Inches(0.03), ty4 + Inches(0.07), col_ws2[j] - Inches(0.04), row_h2 - Inches(0.1),
        font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    cx += col_ws2[j]

for ri, row2 in enumerate(rows2):
    ty5 = ty4 + row_h2 + ri * row_h2
    cx2 = rx
    bg2 = RGBColor(0x18, 0x2A, 0x50) if ri < 2 else MID_NAVY
    for j, cell in enumerate(row2):
        add_rect(sl, cx2, ty5, col_ws2[j], row_h2, bg2)
        txb(sl, cell, cx2 + Inches(0.03), ty5 + Inches(0.08), col_ws2[j] - Inches(0.04), row_h2 - Inches(0.1),
            font_size=10, bold=(ri < 2), color=WHITE if ri < 2 else LIGHT, align=PP_ALIGN.CENTER)
        cx2 += col_ws2[j]

insight_y = ty4 + row_h2 * 6 + Inches(0.15)
txb(sl, "★  Ages 26–45 account for 72.6% of all expected buyers\n   — the primary target segment for the product launch.",
    rx, insight_y, Inches(4.9), Inches(0.72),
    font_size=12, bold=True, color=CORAL)

slide_number(sl, 13)
print("✔ Slide 13 – Business Findings")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 – Budget Allocation (NEW)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Marketing Budget Allocation", "Translating model output into actionable campaign spend")

embed_image(sl, IMGS["donut"], Inches(0.3), Inches(1.5), Inches(7.8), Inches(5.4))

rx = Inches(8.3)
txb(sl, "Allocation Strategy", rx, Inches(1.55), Inches(4.8), Inches(0.42),
    font_size=15, bold=True, color=CORAL)

budget_items = [
    ("26–35  →  37.5% Budget", "Highest expected buyer volume; prioritise digital campaigns\n(Instagram, YouTube) targeting this cohort."),
    ("36–45  →  35.1% Budget", "Strong purchasing power; use premium placement\nand loyalty offers for this established segment."),
    ("18–25  →  19.7% Budget", "Younger males show high propensity; target via\nshort-form video and influencer partnerships."),
    ("46+    →  7.3% Budget",  "Lower volume but potentially high value;\nuse targeted email and in-store promotions."),
]
iy4 = Inches(2.1)
for hd, bd in budget_items:
    add_rect(sl, rx, iy4, Inches(4.8), Inches(1.1), MID_NAVY)
    add_rect(sl, rx, iy4, Inches(0.07), Inches(1.1), CORAL)
    txb(sl, hd, rx + Inches(0.14), iy4 + Inches(0.06), Inches(4.5), Inches(0.3),
        font_size=11, bold=True, color=CORAL)
    txb(sl, bd, rx + Inches(0.14), iy4 + Inches(0.4), Inches(4.5), Inches(0.64),
        font_size=10.5, color=LIGHT)
    iy4 += Inches(1.2)

slide_number(sl, 14)
print("✔ Slide 14 – Budget Allocation")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 – Key Insights & Responsible AI
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
slide_header(sl, "Key Insights & Responsible AI", "What the model reveals — and how to use it ethically")

# Left: insights
txb(sl, "Customer Insights", Inches(0.4), Inches(1.52), Inches(6.0), Inches(0.4),
    font_size=15, bold=True, color=CORAL)

insight_items = [
    ("Behavioural > Demographic",   "Historical spend & loyalty tenure outperform raw age/gender in predictive power."),
    ("Younger Males (18–35)",        "Show disproportionately high purchase propensity — strong early-adopter signal."),
    ("Retention = Revenue",          "Customers with 2+ years tenure are 20% more likely to convert on new launches."),
    ("Probability, Not Certainty",   "Scores indicate relative likelihood — not guaranteed purchase intent."),
]
iy5 = Inches(2.0)
for hd, bd in insight_items:
    add_rect(sl, Inches(0.4), iy5, Inches(6.1), Inches(1.0), MID_NAVY)
    add_rect(sl, Inches(0.4), iy5, Inches(0.07), Inches(1.0), CORAL)
    txb(sl, hd, Inches(0.55), iy5 + Inches(0.07), Inches(5.8), Inches(0.3),
        font_size=12, bold=True, color=CORAL)
    txb(sl, bd, Inches(0.55), iy5 + Inches(0.42), Inches(5.8), Inches(0.5),
        font_size=11, color=LIGHT)
    iy5 += Inches(1.12)

# Right: Responsible AI
rx2 = Inches(6.8)
txb(sl, "Responsible AI", rx2, Inches(1.52), Inches(6.2), Inches(0.4),
    font_size=15, bold=True, color=CORAL)

rai_items = [
    ("🔒  Privacy First",        "Use customer data transparently; comply with data protection regulations."),
    ("⚖️  Avoid Stereotyping",   "Treat demographic analysis as marketing analytics — not individual certainty."),
    ("🧑‍⚖️  Human Judgment",      "A score supports decisions; it does not replace human accountability."),
    ("📡  Monitor Drift",        "Re-evaluate the model quarterly; customer behaviour evolves post-launch."),
]
iy6 = Inches(2.0)
for hd, bd in rai_items:
    add_rect(sl, rx2, iy6, Inches(6.2), Inches(1.0), MID_NAVY)
    add_rect(sl, rx2, iy6, Inches(0.07), Inches(1.0), CORAL)
    txb(sl, hd, rx2 + Inches(0.15), iy6 + Inches(0.07), Inches(5.9), Inches(0.3),
        font_size=12, bold=True, color=CORAL)
    txb(sl, bd, rx2 + Inches(0.15), iy6 + Inches(0.42), Inches(5.9), Inches(0.5),
        font_size=11, color=LIGHT)
    iy6 += Inches(1.12)

slide_number(sl, 15)
print("✔ Slide 15 – Insights & Ethics")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 16 – Conclusion & Thank You
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
navy_bg(sl)
add_rect(sl, 0, 0, W, Inches(0.08), CORAL)
add_rect(sl, 0, H - Inches(0.08), W, Inches(0.08), CORAL)
add_rect(sl, 0, 0, Inches(0.5), H, CORAL)

txb(sl, "From Predictions to Smarter Skincare Marketing",
    Inches(0.8), Inches(0.55), Inches(12), Inches(0.82),
    font_size=28, bold=True, color=WHITE)

# Summary stats row
stats = [
    ("10,000",  "Customer\nProfiles"),
    ("3",       "ML Models\nCompared"),
    ("87.2%",   "Best F1\nScore (RF)"),
    ("0.821",   "Best ROC-AUC\n(RF)"),
    ("72.6%",   "Buyers in\n26–45 cohort"),
]
bx_sw = Inches(2.4)
bx_sh = Inches(1.4)
sx3 = Inches(0.52)
sy4 = Inches(1.55)
for i, (val, lbl) in enumerate(stats):
    lx2 = sx3 + i * (bx_sw + Inches(0.1))
    add_rect(sl, lx2, sy4, bx_sw, bx_sh, MID_NAVY)
    txb(sl, val, lx2, sy4 + Inches(0.1), bx_sw, Inches(0.6),
        font_size=28, bold=True, color=CORAL, align=PP_ALIGN.CENTER)
    txb(sl, lbl, lx2, sy4 + Inches(0.68), bx_sw, Inches(0.65),
        font_size=11, color=LIGHT, align=PP_ALIGN.CENTER)

# Future scope
txb(sl, "Future Scope",
    Inches(0.8), Inches(3.15), Inches(12), Inches(0.38),
    font_size=15, bold=True, color=CORAL)
future = "🚀 Gradient Boosting (XGBoost / LightGBM)  •  🎯 Optimal threshold tuning per cohort  •  🔄 Concept drift monitoring  •  🌍 Richer feature engineering (browsing, purchase recency)  •  📱 Mobile-ready Streamlit app"
txb(sl, future, Inches(0.8), Inches(3.6), Inches(12.2), Inches(0.7),
    font_size=12, color=LIGHT)

add_divider(sl, Inches(4.5), CORAL, 0.8)

txb(sl, "Thank You",
    Inches(0.5), Inches(4.7), Inches(12.3), Inches(0.75),
    font_size=38, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

txb(sl, "Questions & Discussion",
    Inches(0.5), Inches(5.5), Inches(12.3), Inches(0.45),
    font_size=18, italic=True, color=LIGHT, align=PP_ALIGN.CENTER)

txb(sl, "Tushar Jaiswal  •  Shatakshi Jaiswal  •  Sharafat  •  Garvit  •  Tanisha  •  Tanishq Saini  •  Aastha  •  Mahima Kalra",
    Inches(0.5), Inches(6.1), Inches(12.3), Inches(0.38),
    font_size=11, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

txb(sl, "Department of Computer Science & Engineering  |  B.Tech CSE",
    Inches(0.5), Inches(6.52), Inches(12.3), Inches(0.35),
    font_size=10, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

slide_number(sl, 16)
print("✔ Slide 16 – Conclusion & Thank You")

# ── Save ──────────────────────────────────────────────────────────────────────
prs.save(str(OUT))
print(f"\n✅  Saved: {OUT}")
print(f"   Total slides: {len(prs.slides)}")
import os
print(f"   File size   : {os.path.getsize(OUT):,} bytes")
