"""
Build an interactive single-file HTML dashboard for TrustGate results.
Reads CSVs and summary.json from results/, outputs results/dashboard.html.
"""

import os
import json
import csv
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RISK_REGISTER_CSV = RESULTS_DIR / "risk_register.csv"
EROSION_CURVE_CSV = RESULTS_DIR / "erosion_curve.csv"
DOMAIN_EROSION_CSV = RESULTS_DIR / "domain_erosion.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
OUTPUT_HTML = RESULTS_DIR / "dashboard.html"


def read_data():
    """Read all result files and return (risk_register, erosion_curve, domain_erosion, summary)."""
    risk_register = []
    with open(RISK_REGISTER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            risk_register.append(row)

    erosion_curve = []
    with open(EROSION_CURVE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            erosion_curve.append(row)

    domain_erosion = []
    with open(DOMAIN_EROSION_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain_erosion.append(row)

    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return risk_register, erosion_curve, domain_erosion, summary


def build_html(risk_register, erosion_curve, domain_erosion, summary):
    """Build the complete single-file HTML dashboard string."""
    today = datetime.date.today().isoformat()
    total_mas = summary.get("total_mas", len(risk_register))
    total_sig = summary.get("total_significant", 0)
    erosion_70 = summary.get("erosion_rate_at_70", 0)
    red_flag_count = summary.get("red_flag_count", 0)
    hidden_gem_count = summary.get("hidden_gem_count", 0)

    # Embed data as JSON — escape </ to prevent </script> breaking the script block
    def _safe_json(obj):
        return json.dumps(obj).replace('</', '<\\/')

    register_json = _safe_json(risk_register)
    erosion_curve_json = _safe_json(erosion_curve)
    domain_erosion_json = _safe_json(domain_erosion)
    summary_json = _safe_json(summary)

    # Guideline data counts
    who_count = sum(1 for r in risk_register if r.get("who_essential", "").lower() == "true")
    nice_count = sum(1 for r in risk_register if int(r.get("nice_guideline_count", 0) or 0) > 0)
    has_guideline_data = who_count > 0 or nice_count > 0

    # Domain heatmap: detect if all "Unknown"
    has_real_domains = any(
        r.get("domain", "Unknown").strip() not in ("Unknown", "")
        for r in domain_erosion
    )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TrustGate Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.6;
}}
.tg-container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}}

/* Hero */
.tg-hero {{
    text-align: center;
    padding: 60px 20px 40px;
    background: linear-gradient(135deg, #16213e 0%, #1a1a2e 50%, #0f3460 100%);
    border-bottom: 2px solid #0f3460;
    margin-bottom: 30px;
}}
.tg-hero h1 {{
    font-size: 2.8rem;
    font-weight: 700;
    color: #e0e0e0;
    margin-bottom: 10px;
}}
.tg-hero h1 span {{ color: #00d4aa; }}
.tg-hero p {{
    font-size: 1.2rem;
    color: #8892b0;
    max-width: 700px;
    margin: 0 auto;
}}
.tg-stats-row {{
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 30px;
    flex-wrap: wrap;
}}
.tg-stat-card {{
    text-align: center;
    padding: 15px 25px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    min-width: 140px;
    backdrop-filter: blur(4px);
}}
.tg-stat-card .tg-stat-value {{
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d4aa;
}}
.tg-stat-card .tg-stat-label {{
    font-size: 0.8rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}}

/* Sections */
.tg-section {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 25px;
}}
.tg-section h2 {{
    font-size: 1.5rem;
    color: #e0e0e0;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}

/* Chart containers */
.tg-chart-row {{
    display: flex;
    gap: 25px;
    flex-wrap: wrap;
}}
.tg-chart-box {{
    flex: 1;
    min-width: 300px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}}
.tg-chart-box h3 {{
    font-size: 1.1rem;
    color: #8892b0;
    margin-bottom: 15px;
}}
canvas {{ max-width: 100%; }}

/* Filter buttons */
.tg-filter-row {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 15px;
    align-items: center;
}}
.tg-filter-btn {{
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.05);
    color: #e0e0e0;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
}}
.tg-filter-btn:hover {{ background: rgba(255,255,255,0.1); }}
.tg-filter-btn.active {{
    background: rgba(0,212,170,0.2);
    border-color: #00d4aa;
    color: #00d4aa;
}}

/* Search */
.tg-search-box {{
    flex: 1;
    min-width: 200px;
    padding: 8px 14px;
    font-size: 0.95rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    color: #e0e0e0;
    outline: none;
}}
.tg-search-box:focus {{
    border-color: #00d4aa;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.2);
}}
.tg-search-box::placeholder {{ color: #555; }}

/* Tables */
.tg-table-wrap {{
    overflow-x: auto;
    max-height: 600px;
    overflow-y: auto;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}}
th, td {{
    padding: 9px 13px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
th {{
    background: rgba(255,255,255,0.05);
    color: #8892b0;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    z-index: 2;
}}
tr:hover td {{ background: rgba(255,255,255,0.03); }}

/* Grade badges */
.tg-grade {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.82rem;
    text-align: center;
    min-width: 36px;
}}
.tg-grade-aplus {{ background: #059669; color: #fff; }}
.tg-grade-a {{ background: #10b981; color: #fff; }}
.tg-grade-b {{ background: #3b82f6; color: #fff; }}
.tg-grade-c {{ background: #f59e0b; color: #1a1a2e; }}
.tg-grade-d {{ background: #f97316; color: #1a1a2e; }}
.tg-grade-f {{ background: #ef4444; color: #fff; }}

/* Quadrant badges */
.tg-quad {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}}
.tg-quad-redflag {{ background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid #ef4444; }}
.tg-quad-hiddengem {{ background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid #10b981; }}
.tg-quad-safe {{ background: rgba(5,150,105,0.2); color: #059669; border: 1px solid #059669; }}
.tg-quad-lowstakes {{ background: rgba(136,146,176,0.2); color: #8892b0; border: 1px solid #8892b0; }}
.tg-quad-moderate {{ background: rgba(59,130,246,0.2); color: #3b82f6; border: 1px solid #3b82f6; }}

/* Methodology */
.tg-method {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 15px;
}}
.tg-method-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 18px;
}}
.tg-method-card h4 {{
    font-size: 1rem;
    color: #00d4aa;
    margin-bottom: 8px;
}}
.tg-method-card p {{
    font-size: 0.84rem;
    color: #8892b0;
    line-height: 1.5;
}}
.tg-method-card .tg-tag {{
    display: inline-block;
    padding: 2px 8px;
    background: rgba(0,212,170,0.15);
    border-radius: 4px;
    font-size: 0.78rem;
    color: #00d4aa;
    margin-bottom: 8px;
}}

/* Placeholder */
.tg-placeholder {{
    text-align: center;
    padding: 40px;
    color: #555;
    font-style: italic;
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 8px;
}}

/* Footer */
.tg-footer {{
    text-align: center;
    padding: 30px;
    color: #555;
    font-size: 0.85rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 20px;
}}
.tg-footer strong {{ color: #8892b0; }}

/* Showing count */
.tg-showing {{ color: #8892b0; font-size: 0.83rem; margin-top: 10px; }}

/* Responsive */
@media (max-width: 768px) {{
    .tg-hero h1 {{ font-size: 1.8rem; }}
    .tg-stats-row {{ gap: 12px; }}
    .tg-stat-card {{ min-width: 110px; padding: 10px 14px; }}
    .tg-stat-card .tg-stat-value {{ font-size: 1.6rem; }}
    .tg-chart-row {{ flex-direction: column; }}
}}
</style>
</head>
<body>

<!-- Section 1: Hero Stats Bar -->
<div class="tg-hero">
    <h1><span>TrustGate</span> Dashboard</h1>
    <p>Trust-weighted meta-meta-analysis. Significance erosion analysis across {total_mas:,} Cochrane meta-analyses.</p>
    <div class="tg-stats-row">
        <div class="tg-stat-card">
            <div class="tg-stat-value">{total_mas:,}</div>
            <div class="tg-stat-label">MAs Analyzed</div>
        </div>
        <div class="tg-stat-card">
            <div class="tg-stat-value">{erosion_70:.1f}%</div>
            <div class="tg-stat-label">Erosion at B+ (70)</div>
        </div>
        <div class="tg-stat-card">
            <div class="tg-stat-value" id="tg-redflag-count">{red_flag_count:,}</div>
            <div class="tg-stat-label">Red Flags</div>
        </div>
        <div class="tg-stat-card">
            <div class="tg-stat-value" id="tg-hiddengem-count">{hidden_gem_count:,}</div>
            <div class="tg-stat-label">Hidden Gems</div>
        </div>
        <div class="tg-stat-card">
            <div class="tg-stat-value">{total_sig:,}</div>
            <div class="tg-stat-label">Sig. MAs</div>
        </div>
    </div>
</div>

<div class="tg-container">

<!-- Section 2: Significance Erosion Curve -->
<div class="tg-section">
    <h2>Significance Erosion Curve</h2>
    <div class="tg-chart-box" style="max-width:900px;margin:0 auto;">
        <canvas id="tg-erosion-canvas" width="860" height="380"></canvas>
    </div>
    <p style="color:#8892b0;font-size:0.85rem;margin-top:12px;text-align:center;">
        As the trust threshold rises, significant MAs are classified as Surviving (trust &ge; threshold), Weakened (below threshold but still significant), or Excluded.
    </p>
</div>

<!-- Section 3: Domain Heatmap -->
<div class="tg-section">
    <h2>Domain Erosion Heatmap</h2>
    <div id="tg-domain-section">
        {"" if has_real_domains else '<div class="tg-placeholder">Domain breakdown not available in this dataset — all MAs are in the &ldquo;Unknown&rdquo; domain group. Run with review-group metadata to see per-domain erosion.</div>'}
        <canvas id="tg-heatmap-canvas" width="860" height="300" style="{"display:block" if has_real_domains else "display:none"}"></canvas>
    </div>
</div>

<!-- Section 4: Risk Register Table -->
<div class="tg-section">
    <h2>Risk Register</h2>
    <div class="tg-filter-row">
        <button class="tg-filter-btn active" data-quad="All" onclick="tgSetFilter(this,'All')">All</button>
        <button class="tg-filter-btn" data-quad="Red Flag" onclick="tgSetFilter(this,'Red Flag')">Red Flag</button>
        <button class="tg-filter-btn" data-quad="Hidden Gem" onclick="tgSetFilter(this,'Hidden Gem')">Hidden Gem</button>
        <button class="tg-filter-btn" data-quad="Safe" onclick="tgSetFilter(this,'Safe')">Safe</button>
        <button class="tg-filter-btn" data-quad="Low Stakes" onclick="tgSetFilter(this,'Low Stakes')">Low Stakes</button>
        <button class="tg-filter-btn" data-quad="Moderate" onclick="tgSetFilter(this,'Moderate')">Moderate</button>
        <input type="text" class="tg-search-box" id="tg-search" placeholder="Search MA ID or Review ID..." aria-label="Search meta-analyses">
    </div>
    <div class="tg-table-wrap">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>MA ID</th>
                    <th>Review</th>
                    <th>Score</th>
                    <th>Grade</th>
                    <th>Influence</th>
                    <th>Quadrant</th>
                    <th>Citations</th>
                </tr>
            </thead>
            <tbody id="tg-register-body"></tbody>
        </table>
    </div>
    <div class="tg-showing" id="tg-showing"></div>
</div>

<!-- Section 5: Guideline Source Breakdown -->
<div class="tg-section">
    <h2>Guideline Source Breakdown</h2>
    {"''" if has_guideline_data else '<div class="tg-placeholder">Guideline sources pending data enrichment &mdash; WHO Essential Medicines and NICE guideline citation data not yet populated for this dataset.</div>'}
    <canvas id="tg-guideline-canvas" width="640" height="280" style="{"display:block;margin:0 auto;" if has_guideline_data else "display:none"}"></canvas>
</div>

<!-- Section 6: Scatter Plot — Trust vs Influence -->
<div class="tg-section">
    <h2>Trust Score vs Influence Score</h2>
    <div class="tg-chart-box" style="max-width:900px;margin:0 auto;">
        <canvas id="tg-scatter-canvas" width="860" height="500"></canvas>
    </div>
    <p style="color:#8892b0;font-size:0.85rem;margin-top:12px;text-align:center;">
        Quadrant boundaries: Trust 60 (low/mid), Trust 80 (mid/high), Influence 30 (low/mid), Influence 70 (mid/high).
        Points colored by grade.
    </p>
</div>

<!-- Section 7: Methodology Panel -->
<div class="tg-section">
    <h2>Methodology</h2>
    <div class="tg-method">
        <div class="tg-method-card">
            <h4>Trust-Weighting Formula</h4>
            <div class="tg-tag">EvidenceScore</div>
            <p>TrustScore = 0.30&times;Audit + 0.20&times;Consistency + 0.20&times;Robustness + 0.15&times;Stability + 0.15&times;Power. Scale 0&ndash;100. Grades: A+ (&ge;90), A (&ge;80), B (&ge;70), C (&ge;60), D (&ge;50), F (&lt;50).</p>
        </div>
        <div class="tg-method-card">
            <h4>Erosion Analysis</h4>
            <div class="tg-tag">Thresholds 50&ndash;90</div>
            <p>For each trust threshold T, significant MAs are split into Surviving (score &ge; T), Weakened (significant but score &lt; T), and Excluded (non-significant or missing). Erosion rate = (excluded + weakened) / total significant.</p>
        </div>
        <div class="tg-method-card">
            <h4>Influence Score Formula</h4>
            <div class="tg-tag">Citation + Size</div>
            <p>Influence = 0.5&times;CitationPercentile + 0.3&times;GroupSizePercentile + 0.2&times;WHObonus. Reflects both academic impact (citation rank) and practical importance (study sample size and guideline relevance).</p>
        </div>
        <div class="tg-method-card">
            <h4>Risk Register Quadrants</h4>
            <div class="tg-tag">2&times;2 Matrix</div>
            <p>Red Flag: high trust + high influence (needs urgent scrutiny). Hidden Gem: high trust but low influence (under-cited). Safe: high trust + adequate influence. Low Stakes: low influence overall. Moderate: middle band.</p>
        </div>
        <div class="tg-method-card">
            <h4>Data Sources</h4>
            <div class="tg-tag">Cochrane + OpenAlex</div>
            <p>Meta-analyses from Cochrane Reviews (6,229 MAs across 501 reviews). Trust scores from MetaAudit 11-detector pipeline. Citations from OpenAlex. WHO Essential Medicines and NICE guideline flags from curated reference lists.</p>
        </div>
    </div>
</div>

<!-- Footer -->
<div class="tg-footer">
    <strong>TrustGate</strong>: Trust-weighted meta-meta-analysis.
    Covering <strong>{total_mas:,}</strong> Cochrane MAs.
    Generated <strong>{today}</strong>.
</div>

</div><!-- end tg-container -->

<script>
// ── Embedded Data ─────────────────────────────────────────────────────────
var TG_REGISTER = {register_json};
var TG_EROSION = {erosion_curve_json};
var TG_DOMAIN = {domain_erosion_json};
var TG_SUMMARY = {summary_json};

// ── HTML escape helper (prevent XSS from CSV data) ───────────────────────
function tgEsc(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
        .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

// ── Grade helpers ─────────────────────────────────────────────────────────
function tgGradeClass(g) {{
    var m = {{'A+':'tg-grade-aplus','A':'tg-grade-a','B':'tg-grade-b',
              'C':'tg-grade-c','D':'tg-grade-d','F':'tg-grade-f'}};
    return m[g] || 'tg-grade-f';
}}
function tgGradeColor(g) {{
    var m = {{'A+':'#059669','A':'#10b981','B':'#3b82f6',
              'C':'#f59e0b','D':'#f97316','F':'#ef4444'}};
    return m[g] || '#ef4444';
}}
function tgQuadClass(q) {{
    if (!q) return 'tg-quad-moderate';
    var key = q.toLowerCase().replace(/[^a-z]/g,'');
    var m = {{
        'redflag': 'tg-quad-redflag',
        'hiddengem': 'tg-quad-hiddengem',
        'safe': 'tg-quad-safe',
        'lowstakes': 'tg-quad-lowstakes',
        'moderate': 'tg-quad-moderate'
    }};
    return m[key] || 'tg-quad-moderate';
}}

// ── Section 2: Erosion Curve ───────────────────────────────────────────────
(function() {{
    var canvas = document.getElementById('tg-erosion-canvas');
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    var pad = {{t:40, r:140, b:60, l:65}};
    var cW = W - pad.l - pad.r;
    var cH = H - pad.t - pad.b;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    if (!TG_EROSION || TG_EROSION.length === 0) {{
        ctx.fillStyle = '#555';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No erosion data available', W/2, H/2);
        return;
    }}

    var thresholds = TG_EROSION.map(function(r) {{ return parseInt(r.threshold); }});
    var surviving = TG_EROSION.map(function(r) {{ return parseInt(r.n_surviving || 0); }});
    var weakened = TG_EROSION.map(function(r) {{ return parseInt(r.n_weakened || 0); }});
    var excluded = TG_EROSION.map(function(r) {{ return parseInt(r.n_excluded || 0); }});
    var totalSig = parseInt(TG_EROSION[0].n_total_sig || 1);

    // Stacked areas: surviving (bottom), weakened (mid), excluded (top)
    var n = thresholds.length;
    function xPos(i) {{ return pad.l + (i / (n - 1)) * cW; }}
    function yPos(v) {{ return pad.t + cH - (v / totalSig) * cH; }}

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (var g = 0; g <= 5; g++) {{
        var gy = pad.t + (g / 5) * cH;
        ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(pad.l + cW, gy); ctx.stroke();
        ctx.fillStyle = '#8892b0';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(totalSig * (5-g) / 5), pad.l - 8, gy + 4);
    }}

    // Draw stacked areas: excluded (red, full height), then weakened+surviving (yellow), then surviving (green)
    // Area 1: excluded region (from surviving+weakened to totalSig)
    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(surviving[0] + weakened[0]));
    for (var i = 1; i < n; i++) ctx.lineTo(xPos(i), yPos(surviving[i] + weakened[i]));
    ctx.lineTo(xPos(n-1), yPos(0));
    ctx.lineTo(xPos(0), yPos(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(239,68,68,0.25)';
    ctx.fill();

    // Area 2: weakened region (from surviving to surviving+weakened)
    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(surviving[0]));
    for (var i = 1; i < n; i++) ctx.lineTo(xPos(i), yPos(surviving[i]));
    for (var i = n-1; i >= 0; i--) ctx.lineTo(xPos(i), yPos(surviving[i] + weakened[i]));
    ctx.closePath();
    ctx.fillStyle = 'rgba(245,158,11,0.3)';
    ctx.fill();

    // Area 3: surviving region (from 0 to surviving)
    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(0));
    for (var i = 0; i < n; i++) ctx.lineTo(xPos(i), yPos(surviving[i]));
    ctx.lineTo(xPos(n-1), pad.t + cH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(16,185,129,0.25)';
    ctx.fill();

    // Lines
    var series = [
        {{ data: surviving, color: '#10b981', label: 'Surviving' }},
        {{ data: weakened.map(function(w,i) {{ return surviving[i]+w; }}), color: '#f59e0b', label: 'Weakened boundary' }},
        {{ data: excluded.map(function(e,i) {{ return surviving[i]+weakened[i]; }}), color: '#ef4444', label: 'Excluded boundary' }}
    ];
    series.forEach(function(s) {{
        ctx.beginPath();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.5;
        for (var i = 0; i < n; i++) {{
            var x = xPos(i), y = yPos(s.data[i]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        ctx.stroke();
        // Dots
        for (var i = 0; i < n; i++) {{
            ctx.beginPath();
            ctx.arc(xPos(i), yPos(s.data[i]), 4, 0, 2*Math.PI);
            ctx.fillStyle = s.color;
            ctx.fill();
        }}
    }});

    // X-axis labels
    ctx.fillStyle = '#8892b0';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    for (var i = 0; i < n; i++) {{
        ctx.fillText(thresholds[i], xPos(i), pad.t + cH + 22);
    }}
    ctx.fillText('Trust Threshold', pad.l + cW/2, H - 5);

    // Y-axis label
    ctx.save();
    ctx.translate(14, pad.t + cH/2);
    ctx.rotate(-Math.PI/2);
    ctx.textAlign = 'center';
    ctx.fillText('Count', 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = '#e0e0e0';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Significance Erosion by Trust Threshold', pad.l + cW/2, 20);

    // Legend
    var lx = W - pad.r + 15, ly = pad.t + 10;
    var legItems = [
        {{ color: '#10b981', label: 'Surviving' }},
        {{ color: '#f59e0b', label: 'Weakened' }},
        {{ color: '#ef4444', label: 'Excluded' }}
    ];
    legItems.forEach(function(item, i) {{
        ctx.fillStyle = item.color;
        ctx.fillRect(lx, ly + i*22, 12, 12);
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(item.label, lx + 16, ly + i*22 + 10);
    }});
}})();

// ── Section 3: Domain Heatmap ──────────────────────────────────────────────
(function() {{
    var canvas = document.getElementById('tg-heatmap-canvas');
    if (!canvas || canvas.style.display === 'none') return;
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    if (!TG_DOMAIN || TG_DOMAIN.length === 0) return;

    // Build domain x threshold matrix
    var domains = [];
    var thresholds = [];
    TG_DOMAIN.forEach(function(r) {{
        if (domains.indexOf(r.domain) === -1) domains.push(r.domain);
        var t = parseInt(r.threshold);
        if (thresholds.indexOf(t) === -1) thresholds.push(t);
    }});
    thresholds.sort(function(a,b) {{ return a-b; }});

    var rates = {{}};
    TG_DOMAIN.forEach(function(r) {{
        var key = r.domain + '|' + r.threshold;
        rates[key] = parseFloat(r.erosion_rate || 0);
    }});

    var padL = 120, padT = 40, padR = 20, padB = 40;
    var cellW = (W - padL - padR) / thresholds.length;
    var cellH = Math.min(40, (H - padT - padB) / domains.length);

    // Column headers (thresholds)
    ctx.fillStyle = '#8892b0';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    thresholds.forEach(function(t, j) {{
        ctx.fillText(t, padL + j*cellW + cellW/2, padT - 8);
    }});

    // Rows
    domains.forEach(function(dom, i) {{
        // Row label
        ctx.fillStyle = '#8892b0';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        var label = dom.length > 16 ? dom.substring(0,14)+'...' : dom;
        ctx.fillText(label, padL - 6, padT + i*cellH + cellH/2 + 4);

        thresholds.forEach(function(t, j) {{
            var rate = rates[dom + '|' + t] || 0;
            // Color: 0%=green, 100%=red
            var r2 = Math.round(rate * 2.39);
            var g2 = Math.round((100 - rate) * 1.85);
            var color = 'rgb(' + r2 + ',' + g2 + ',80)';
            ctx.fillStyle = color;
            ctx.fillRect(padL + j*cellW + 1, padT + i*cellH + 1, cellW - 2, cellH - 2);
            ctx.fillStyle = '#fff';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(rate.toFixed(0) + '%', padL + j*cellW + cellW/2, padT + i*cellH + cellH/2 + 3);
        }});
    }});

    // Axis label
    ctx.fillStyle = '#8892b0';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Trust Threshold', padL + (W-padL-padR)/2, H - 5);
}})();

// ── Section 4: Risk Register Table ────────────────────────────────────────
var tgActiveFilter = 'All';
var tgSearchVal = '';
var TG_SHOW_MAX = 200;

function tgRenderRow(d, idx) {{
    var gc = tgGradeClass(d.grade);
    var qc = tgQuadClass(d.quadrant);
    var score = parseFloat(d.final_score || 0);
    var sc = score >= 90 ? '#059669' : score >= 80 ? '#10b981' : score >= 70 ? '#3b82f6' :
             score >= 60 ? '#f59e0b' : score >= 50 ? '#f97316' : '#ef4444';
    var inf = parseFloat(d.influence_score || 0).toFixed(1);
    var cit = parseInt(d.citation_count || 0);
    return '<tr>' +
        '<td style="color:#555;">' + (idx+1) + '</td>' +
        '<td style="font-family:monospace;font-size:0.82rem;">' + tgEsc(d.ma_id || '') + '</td>' +
        '<td style="color:#8892b0;">' + tgEsc(d.review_id || '') + '</td>' +
        '<td><strong style="color:' + sc + '">' + score.toFixed(0) + '</strong></td>' +
        '<td><span class="tg-grade ' + gc + '">' + tgEsc(d.grade || '?') + '</span></td>' +
        '<td>' + inf + '</td>' +
        '<td><span class="tg-quad ' + qc + '">' + tgEsc(d.quadrant || 'Moderate') + '</span></td>' +
        '<td>' + cit + '</td>' +
    '</tr>';
}}

function tgRenderTable() {{
    var body = document.getElementById('tg-register-body');
    var html = '';
    var count = 0, total = 0;
    var filterLC = tgActiveFilter.toLowerCase();
    var searchLC = tgSearchVal.toLowerCase();
    for (var i = 0; i < TG_REGISTER.length; i++) {{
        var d = TG_REGISTER[i];
        if (tgActiveFilter !== 'All') {{
            var q = (d.quadrant || '').toLowerCase();
            if (q !== filterLC) continue;
        }}
        if (searchLC) {{
            var mid = (d.ma_id || '').toLowerCase();
            var rid = (d.review_id || '').toLowerCase();
            if (mid.indexOf(searchLC) === -1 && rid.indexOf(searchLC) === -1) continue;
        }}
        total++;
        if (count < TG_SHOW_MAX) {{
            html += tgRenderRow(d, count);
            count++;
        }}
    }}
    body.innerHTML = html;
    document.getElementById('tg-showing').textContent =
        'Showing ' + count + ' of ' + total + ' meta-analyses' +
        (total < TG_REGISTER.length ? ' (filtered)' : '') +
        (count < total ? ' \u2014 refine search to see more' : '');
}}

function tgSetFilter(btn, quad) {{
    document.querySelectorAll('.tg-filter-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');
    tgActiveFilter = quad;
    TG_SHOW_MAX = 200;
    tgRenderTable();
}}

document.getElementById('tg-search').addEventListener('input', function() {{
    tgSearchVal = this.value.trim();
    TG_SHOW_MAX = 500;
    tgRenderTable();
}});

tgRenderTable();

// ── Section 5: Guideline Chart ─────────────────────────────────────────────
(function() {{
    var canvas = document.getElementById('tg-guideline-canvas');
    if (!canvas || canvas.style.display === 'none') return;
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    var whoCount = 0, niceCount = 0;
    TG_REGISTER.forEach(function(r) {{
        if ((r.who_essential || '').toLowerCase() === 'true') whoCount++;
        if (parseInt(r.nice_guideline_count || 0) > 0) niceCount++;
    }});

    var bars = [
        {{ label: 'WHO Essential Match', value: whoCount, color: '#10b981' }},
        {{ label: 'NICE Guideline Citation', value: niceCount, color: '#3b82f6' }}
    ];
    var maxVal = Math.max.apply(null, bars.map(function(b) {{ return b.value; }})) || 1;

    var padL = 200, padT = 40, padR = 40, padB = 40;
    var barH = 50, gap = 30;
    var chartW = W - padL - padR;

    bars.forEach(function(b, i) {{
        var y = padT + i*(barH + gap);
        var bW = (b.value / maxVal) * chartW;

        ctx.fillStyle = b.color;
        ctx.fillRect(padL, y, Math.max(bW, 2), barH);

        ctx.fillStyle = '#8892b0';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(b.label, padL - 10, y + barH/2 + 4);

        ctx.fillStyle = '#e0e0e0';
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(b.value.toLocaleString(), padL + bW + 8, y + barH/2 + 4);
    }});
}})();

// ── Section 6: Scatter Plot ────────────────────────────────────────────────
(function() {{
    var canvas = document.getElementById('tg-scatter-canvas');
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    var pad = {{t:40, r:120, b:60, l:65}};
    var cW = W - pad.l - pad.r;
    var cH = H - pad.t - pad.b;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (var g = 0; g <= 5; g++) {{
        var gy = pad.t + (g/5)*cH;
        ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(pad.l+cW, gy); ctx.stroke();
        ctx.fillStyle = '#8892b0';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(100*(5-g)/5), pad.l-6, gy+3);
    }}
    for (var g = 0; g <= 5; g++) {{
        var gx = pad.l + (g/5)*cW;
        ctx.beginPath(); ctx.moveTo(gx, pad.t); ctx.lineTo(gx, pad.t+cH); ctx.stroke();
        ctx.fillStyle = '#8892b0';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(100*g/5), gx, pad.t+cH+16);
    }}

    // Quadrant boundary lines (dashed)
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    // x = 60
    var x60 = pad.l + (60/100)*cW;
    ctx.beginPath(); ctx.moveTo(x60, pad.t); ctx.lineTo(x60, pad.t+cH); ctx.stroke();
    // x = 80
    var x80 = pad.l + (80/100)*cW;
    ctx.beginPath(); ctx.moveTo(x80, pad.t); ctx.lineTo(x80, pad.t+cH); ctx.stroke();
    // y = 30
    var y30 = pad.t + cH - (30/100)*cH;
    ctx.beginPath(); ctx.moveTo(pad.l, y30); ctx.lineTo(pad.l+cW, y30); ctx.stroke();
    // y = 70
    var y70 = pad.t + cH - (70/100)*cH;
    ctx.beginPath(); ctx.moveTo(pad.l, y70); ctx.lineTo(pad.l+cW, y70); ctx.stroke();
    ctx.setLineDash([]);

    // Quadrant zone labels
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Low Trust', pad.l + x60/2 - pad.l/2, pad.t + 18);
    ctx.fillText('Hidden Gem zone', pad.l + (x60-pad.l)/2 + (x80-x60)/4, y70 - 8);

    // Points
    for (var i = 0; i < TG_REGISTER.length; i++) {{
        var d = TG_REGISTER[i];
        var sx = parseFloat(d.final_score || 0);
        var sy = parseFloat(d.influence_score || 0);
        if (isNaN(sx) || isNaN(sy)) continue;

        var px = pad.l + (sx/100)*cW;
        var py = pad.t + cH - (sy/100)*cH;

        ctx.beginPath();
        ctx.arc(px, py, 2, 0, 2*Math.PI);
        ctx.fillStyle = tgGradeColor(d.grade) + '99'; // semi-transparent
        ctx.fill();
    }}

    // Axis labels
    ctx.fillStyle = '#8892b0';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('EvidenceScore (Trust)', pad.l + cW/2, H - 5);
    ctx.save();
    ctx.translate(13, pad.t + cH/2);
    ctx.rotate(-Math.PI/2);
    ctx.textAlign = 'center';
    ctx.fillText('Influence Score', 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = '#e0e0e0';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Trust vs Influence — All ' + TG_REGISTER.length + ' MAs', pad.l + cW/2, 20);

    // Grade legend
    var grades = ['A+','A','B','C','D','F'];
    var gColors = {{'A+':'#059669','A':'#10b981','B':'#3b82f6','C':'#f59e0b','D':'#f97316','F':'#ef4444'}};
    var lx = W - pad.r + 12, ly = pad.t;
    ctx.font = '11px sans-serif';
    grades.forEach(function(g, i) {{
        ctx.fillStyle = gColors[g];
        ctx.beginPath(); ctx.arc(lx+6, ly + i*20 + 6, 5, 0, 2*Math.PI); ctx.fill();
        ctx.fillStyle = '#e0e0e0';
        ctx.textAlign = 'left';
        ctx.fillText(g, lx+16, ly + i*20 + 10);
    }});
}})();

</script>
</body>
</html>'''
    return html


def main():
    """Read data, build HTML, write to disk."""
    print("Reading TrustGate results...")
    risk_register, erosion_curve, domain_erosion, summary = read_data()
    print(f"  Risk register: {len(risk_register)} rows")
    print(f"  Erosion curve: {len(erosion_curve)} rows")
    print(f"  Domain erosion: {len(domain_erosion)} rows")

    print("Building HTML dashboard...")
    html = build_html(risk_register, erosion_curve, domain_erosion, summary)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"Dashboard written to: {OUTPUT_HTML}")
    print(f"File size: {size_kb:.1f} KB ({len(html.splitlines())} lines)")


if __name__ == "__main__":
    main()
