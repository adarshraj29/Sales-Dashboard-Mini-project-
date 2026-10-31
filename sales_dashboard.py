"""
SALES ANALYSIS DASHBOARD  (Interactive Edition)
-----------------------------------------------
Author  : Adarsh Raj (UID: 26MCA20214)
Subject : Python Programming  |  Submitted to: Randeep Singh Dhillon

How it works
  1. NumPy   -> builds the sales data (month x category x region) and does all the maths
  2. Matplotlib -> draws line, area, bar and pie charts as PNG images
  3. The NumPy results are saved into an interactive web page (index.html) that
     the user can filter, switch, click and download from. Open it via XAMPP.

Run:  python sales_dashboard.py sales_dashboard
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ settings
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "output"
os.makedirs(OUT_DIR, exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CATEGORIES = ["Electronics", "Clothing", "Groceries", "Furniture", "Sports"]
REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Students", "Young Professionals", "Families", "Seniors"]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
SEG_COLORS = ["#72B7B2", "#EECA3B", "#FF9DA6", "#9D755D"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25})

# ------------------------------------------------------ 1. DATA WITH NUMPY
rng = np.random.default_rng(42)            # fixed seed -> same result each run

unit_price = np.array([18000, 1500, 400, 9000, 2500])        # Rs. per unit
base_units = np.array([120, 300, 650, 80, 150])              # monthly units
season = np.array([0.8, 0.8, 0.9, 0.95, 1.0, 0.95,
                   0.9, 1.0, 1.05, 1.3, 1.4, 1.5])           # festive Oct-Dec peak
region_weight = np.array([1.2, 0.9, 0.8, 1.1])

# units[month, category, region]  (3-D NumPy array using broadcasting)
units = (season[:, None, None] * base_units[None, :, None]
         * region_weight[None, None, :] / 4)
units = (units * rng.normal(1, 0.08, units.shape)).round().astype(int)
revenue = units * unit_price[None, :, None]                  # same shape, in Rs.

# customer demographics: which segment buys which category (rows add up to 1)
segment_share = np.array([[0.22, 0.38, 0.28, 0.12],          # Electronics
                          [0.38, 0.34, 0.20, 0.08],          # Clothing
                          [0.15, 0.25, 0.42, 0.18],          # Groceries
                          [0.08, 0.30, 0.50, 0.12],          # Furniture
                          [0.40, 0.35, 0.17, 0.08]])         # Sports
noisy = segment_share[None, :, :, None] * rng.normal(1, 0.1, (12, 5, 4, 4))
noisy = noisy / noisy.sum(axis=2, keepdims=True)             # normalise per segment axis
seg_rev = revenue[:, :, None, :] * noisy                     # [month, cat, segment, region]
seg_age = [round(float(rng.normal(m, 5, 500).mean()), 1) for m in (21, 29, 41, 63)]


# ------------------------------------------------------ 2. ANALYSIS WITH NUMPY
monthly_revenue = revenue.sum(axis=(1, 2))
category_revenue = revenue.sum(axis=(0, 2))
region_revenue = revenue.sum(axis=(0, 1))
monthly_by_cat = revenue.sum(axis=2)
segment_revenue = seg_rev.sum(axis=(0, 1, 3))
total_rev = revenue.sum()
best_month = MONTHS[monthly_revenue.argmax()]
worst_month = MONTHS[monthly_revenue.argmin()]
best_cat = CATEGORIES[category_revenue.argmax()]
best_region = REGIONS[region_revenue.argmax()]
growth = (monthly_revenue[-1] - monthly_revenue[0]) / monthly_revenue[0] * 100
moving_avg = np.convolve(monthly_revenue, np.ones(3) / 3, mode="valid")
lakh = 1e5

print("=" * 58)
print("         SALES ANALYSIS DASHBOARD - SUMMARY")
print("=" * 58)
print(f"Total revenue        : Rs. {total_rev / 1e7:,.2f} Crore")
print(f"Total units sold     : {units.sum():,}")
print(f"Average monthly rev. : Rs. {monthly_revenue.mean() / lakh:,.1f} Lakh")
print(f"Best / Worst month   : {best_month} / {worst_month}")
print(f"Top category         : {best_cat} "
      f"({category_revenue.max() / total_rev * 100:.1f}% of revenue)")
print(f"Top region           : {best_region}")
print(f"Top segment          : {SEGMENTS[segment_revenue.argmax()]}")
print(f"Jan -> Dec growth    : {growth:+.1f}%")
print("=" * 58)

# ------------------------------------------------------ 3. MATPLOTLIB CHARTS
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130)
    plt.close(fig)

def line_plot(ax):
    ax.plot(MONTHS, monthly_revenue / lakh, marker="o", lw=2.5, color="#4C78A8",
            label="Monthly revenue")
    ax.plot(MONTHS[1:-1], moving_avg / lakh, ls="--", lw=2, color="#E45756",
            label="3-month moving avg")
    ax.annotate(f"Peak: {best_month}", (MONTHS.index(best_month), monthly_revenue.max() / lakh),
                xytext=(-55, -5), textcoords="offset points",
                arrowprops=dict(arrowstyle="->"), fontweight="bold")
    ax.set_title("Monthly Sales Trend (Line Plot)", fontweight="bold")
    ax.set_ylabel("Revenue (Rs. Lakh)")
    ax.legend()

def area_chart(ax):
    ax.stackplot(MONTHS, (monthly_by_cat / lakh).T, labels=CATEGORIES, colors=COLORS, alpha=0.9)
    ax.set_title("Revenue by Category over Time (Stacked Area)", fontweight="bold")
    ax.set_ylabel("Revenue (Rs. Lakh)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

def bar_category(ax):
    bars = ax.bar(CATEGORIES, category_revenue / 1e7, color=COLORS)
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_title("Revenue by Product Category (Bar Plot)", fontweight="bold")
    ax.set_ylabel("Revenue (Rs. Crore)")

def bar_region(ax):
    x = np.arange(len(REGIONS))
    w = 0.16
    reg_cat = revenue.sum(axis=0) / 1e7
    for i, cat in enumerate(CATEGORIES):
        ax.bar(x + (i - 2) * w, reg_cat[i], w, label=cat, color=COLORS[i])
    ax.set_xticks(x, REGIONS)
    ax.set_title("Category Performance across Regions (Grouped Bars)", fontweight="bold")
    ax.set_ylabel("Revenue (Rs. Crore)")
    ax.legend(fontsize=8, ncol=2)

def pie_category(ax):
    ax.pie(category_revenue, labels=CATEGORIES, colors=COLORS, autopct="%1.1f%%",
           startangle=90, wedgeprops=dict(width=0.55, edgecolor="white"))
    ax.set_title("Sales Share by Product Type (Donut)", fontweight="bold")
    ax.grid(False)

def pie_segment(ax):
    ax.pie(segment_revenue, labels=SEGMENTS, autopct="%1.1f%%", startangle=90,
           colors=SEG_COLORS, explode=[0.04] * 4, shadow=True)
    ax.set_title("Sales by Customer Segment (Pie)", fontweight="bold")
    ax.grid(False)

charts = [("1_line_trend.png", "Line plot - monthly sales trend", line_plot),
          ("2_area_chart.png", "Stacked area chart - category revenue over time", area_chart),
          ("3_bar_category.png", "Bar plot - revenue by product category", bar_category),
          ("4_bar_region.png", "Grouped bar plot - categories across regions", bar_region),
          ("5_pie_category.png", "Donut chart - share by product type", pie_category),
          ("6_pie_segment.png", "Pie chart - share by customer segment", pie_segment)]

for fname, _, fn in charts:
    fig, ax = plt.subplots(figsize=(8, 5))
    fn(ax)
    save(fig, fname)

kpis = [("Total Revenue", f"Rs. {total_rev / 1e7:.2f} Cr"), ("Units Sold", f"{units.sum():,}"),
        ("Best Month", best_month), ("Top Category", best_cat), ("Top Region", best_region)]
fig = plt.figure(figsize=(18, 11), facecolor="#F5F7FA")
fig.suptitle("SALES ANALYSIS DASHBOARD", fontsize=24, fontweight="bold", color="#1F3A5F", y=0.985)
for i, (k, v) in enumerate(kpis):
    fig.text(0.1 + i * 0.18, 0.925, v, ha="center", fontsize=17, fontweight="bold", color="#E45756")
    fig.text(0.1 + i * 0.18, 0.90, k, ha="center", fontsize=10, color="#555")
gs = fig.add_gridspec(2, 3, top=0.87, hspace=0.35, wspace=0.28)
for (r, c), fn in zip([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                      [line_plot, area_chart, bar_category, bar_region, pie_category, pie_segment]):
    fn(fig.add_subplot(gs[r, c]))
fig.savefig(os.path.join(OUT_DIR, "dashboard.png"), dpi=110)
plt.close(fig)

# ------------------------------------------------------ 4. INTERACTIVE WEB DASHBOARD
# NumPy computed the data above; here it is exported as JSON and the browser
# (Chart.js) recalculates charts instantly when the user changes a filter.
import json
data = {"months": MONTHS, "cats": CATEGORIES, "regs": REGIONS, "segs": SEGMENTS,
        "price": unit_price.tolist(), "units": units.tolist(),
        "segRev": seg_rev.sum(axis=(1, 3)).T.round().tolist(), "colors": COLORS}

PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sales Analysis Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#eef2f9;--card:#ffffffcc;--tx:#1b2a49;--mut:#6b7a99;--acc:#6c5ce7;--grid:#0001}
[data-theme=dark]{--bg:#0e1424;--card:#1a2340cc;--tx:#e8edff;--mut:#9aa8cc;--grid:#fff2}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--tx);transition:.3s}
header{background:linear-gradient(120deg,#6c5ce7,#00b4d8,#f72585);background-size:200% 200%;animation:bg 12s ease infinite;
 color:#fff;padding:22px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}
@keyframes bg{50%{background-position:100% 50%}}
header h1{margin:0;font-size:26px}header small{opacity:.9}
.wrap{padding:18px 22px;max-width:1500px;margin:auto}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:var(--card);padding:14px;border-radius:14px;
 backdrop-filter:blur(8px);box-shadow:0 4px 18px #0002;margin-bottom:18px}
.bar label{font-size:12px;color:var(--mut);display:block;margin-bottom:3px}
select,button{font:inherit;padding:7px 11px;border-radius:9px;border:1px solid #8884;background:var(--card);color:var(--tx);cursor:pointer}
button:hover{border-color:var(--acc)}.on{background:var(--acc)!important;color:#fff!important;border-color:var(--acc)}
.chip{border-radius:20px;padding:5px 13px;margin:0 3px 3px 0;font-size:13px}
.grp{display:flex;gap:4px;flex-wrap:wrap}.sp{flex:1}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:18px}
.kpi{border-radius:16px;padding:16px 18px;color:#fff;box-shadow:0 6px 18px #0003;transition:.25s;animation:up .6s both}
.kpi:hover{transform:translateY(-5px) scale(1.02)}.kpi b{display:block;font-size:25px;margin-top:4px}.kpi span{font-size:12px;opacity:.9}
@keyframes up{from{opacity:0;transform:translateY(20px)}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.card{background:var(--card);border-radius:16px;padding:16px;box-shadow:0 4px 18px #0002;backdrop-filter:blur(8px);animation:up .7s both;transition:.25s}
.card:hover{box-shadow:0 8px 26px #0004}.card h3{margin:0 0 10px;font-size:15px}.wide{grid-column:1/-1}
.ch{position:relative;height:290px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:8px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}
.ins{line-height:1.8;font-size:14px}.ins i{font-style:normal;color:var(--acc);font-weight:700}
footer{text-align:center;color:var(--mut);padding:18px;font-size:13px}
@media(max-width:520px){.grid{grid-template-columns:1fr}.ch{height:250px}header h1{font-size:20px}}
</style></head><body>
<header><div><h1>&#128200; Sales Analysis Dashboard</h1><small>Python &bull; NumPy &bull; Matplotlib &mdash; Adarsh Raj (26MCA20214)</small></div>
<button id="theme">&#127769; Dark mode</button></header>
<div class="wrap">
<div class="bar">
 <div><label>Region</label><select id="reg"></select></div>
 <div><label>From</label><select id="from"></select></div>
 <div><label>To</label><select id="to"></select></div>
 <div><label>Show</label><div class="grp"><button id="m_rev" class="on">Revenue</button><button id="m_qty">Units</button></div></div>
 <div><label>Trend style</label><div class="grp"><button id="t_line" class="on">Line + forecast</button><button id="t_area">Stacked area</button></div></div>
 <div class="sp"><label>Categories (click to toggle)</label><div id="chips"></div></div>
 <button id="reset">&#8635; Reset</button><button id="csv">&#11015; Export CSV</button>
</div>
<div class="kpis" id="kpis"></div>
<div class="grid">
 <div class="card wide"><h3 id="t1">Sales trend</h3><div class="ch"><canvas id="c1"></canvas></div></div>
 <div class="card"><h3>Performance by category</h3><div class="ch"><canvas id="c2"></canvas></div></div>
 <div class="card"><h3>Performance by region</h3><div class="ch"><canvas id="c3"></canvas></div></div>
 <div class="card"><h3>Share by product type</h3><div class="ch"><canvas id="c4"></canvas></div></div>
 <div class="card"><h3>Share by customer segment</h3><div class="ch"><canvas id="c5"></canvas></div></div>
 <div class="card"><h3>&#127942; Top 5 category-region combinations</h3><table id="tbl"></table></div>
 <div class="card"><h3>&#128161; Auto insights</h3><div class="ins" id="ins"></div></div>
</div></div>
<footer>Data generated with NumPy &bull; Charts drawn with Matplotlib (PNG gallery) &amp; Chart.js (interactive)</footer>
<script>
const D=__DATA__;const $=i=>document.getElementById(i);
const GR=['#6c5ce7,#a29bfe','#00b4d8,#48cae4','#f72585,#ff6b9d','#fb8500,#ffb703','#2a9d8f,#52b788'];
const def={reg:'all',cats:[0,1,2,3,4],from:0,to:11,metric:'rev',mode:'line'};let st=JSON.parse(JSON.stringify(def));
const fm=(v,m)=>m==='rev'?(v>=1e7?'\u20b9'+(v/1e7).toFixed(2)+' Cr':'\u20b9'+(v/1e5).toFixed(1)+' L'):Math.round(v).toLocaleString('en-IN');
const val=(m,c,r,k)=>D.units[m][c][r]*(k==='rev'?D.price[c]:1);
$('reg').innerHTML='<option value="all">All regions</option>'+D.regs.map((r,i)=>`<option value="${i}">${r}</option>`).join('');
['from','to'].forEach(id=>$(id).innerHTML=D.months.map((m,i)=>`<option value="${i}">${m}</option>`).join(''));
$('to').value=11;
$('chips').innerHTML=D.cats.map((c,i)=>`<button class="chip on" data-i="${i}" style="border-color:${D.colors[i]}">${c}</button>`).join('');
Chart.defaults.font.family='Segoe UI,Arial';Chart.defaults.maintainAspectRatio=false;
const mk=(id,type,opt={})=>new Chart($(id),{type,data:{labels:[],datasets:[]},options:Object.assign({animation:{duration:700}},opt)});
const sc={scales:{y:{beginAtZero:true,grid:{color:'#8883'}},x:{grid:{display:false}}}};
const c1=mk('c1','line',sc),c2=mk('c2','bar',{...sc,plugins:{legend:{display:false}}}),c3=mk('c3','bar',sc),
 c4=mk('c4','doughnut',{cutout:'55%'}),c5=mk('c5','pie');
function count(el,to,k){const f=+el.dataset.v||0;el.dataset.v=to;const t0=performance.now();
 (function s(t){const p=Math.min(1,(t-t0)/600);el.textContent=fm(f+(to-f)*p,k);if(p<1)requestAnimationFrame(s)})(t0)}
function update(){
 const R=st.reg==='all'?[0,1,2,3]:[+st.reg],C=st.cats,k=st.metric,ms=[];
 for(let m=st.from;m<=st.to;m++)ms.push(m);
 const S=(M,CC,RR,kk)=>{let t=0;M.forEach(m=>CC.forEach(c=>RR.forEach(r=>t+=val(m,c,r,kk))));return t};
 const monthly=ms.map(m=>S([m],C,R,k)),totR=S(ms,C,R,'rev'),totU=S(ms,C,R,'qty');
 const catT=D.cats.map((_,c)=>C.includes(c)?S(ms,[c],R,k):0),regT=D.regs.map((_,r)=>S(ms,C,[r],k));
 const bi=monthly.indexOf(Math.max(...monthly)),g=monthly.length>1&&monthly[0]?(monthly.at(-1)-monthly[0])/monthly[0]*100:0;
 const tc=catT.indexOf(Math.max(...catT));
 const K=[['Total revenue',totR,'rev'],['Units sold',totU,'qty'],['Avg / month',totR/(ms.length||1),'rev'],
  ['Best month',D.months[ms[bi]]||'-'],['Growth first\u2192last',g.toFixed(1)+'%'],['Top category',C.length?D.cats[tc]:'-']];
 if(!$('kpis').children.length)$('kpis').innerHTML=K.map((x,i)=>`<div class="kpi" style="background:linear-gradient(135deg,${GR[i%5].split(',').join(',')});animation-delay:${i*80}ms"><span>${x[0]}</span><b></b></div>`).join('');
 K.forEach((x,i)=>{const b=$('kpis').children[i].querySelector('b');x[2]?count(b,x[1],x[2]):b.textContent=x[1]});
 // trend (line + linear-regression forecast  OR  stacked area)
 let labels=ms.map(m=>D.months[m]),ds;
 if(st.mode==='line'){const n=monthly.length,xs=[...Array(n).keys()],mx=(n-1)/2,my=monthly.reduce((a,b)=>a+b,0)/(n||1);
  const sl=n>1?xs.reduce((a,x,i)=>a+(x-mx)*(monthly[i]-my),0)/xs.reduce((a,x)=>a+(x-mx)**2,0):0;
  const fc=[1,2,3].map(j=>Math.max(0,my+sl*(n-1+j-mx)));
  labels=labels.concat(['Next+1','Next+2','Next+3']);
  ds=[{label:'Sales',data:monthly,borderColor:'#6c5ce7',backgroundColor:'#6c5ce733',fill:false,tension:.35,borderWidth:3,pointRadius:4},
   {label:'Forecast (linear trend)',data:Array(Math.max(n-1,0)).fill(null).concat([monthly.at(-1)],fc),borderColor:'#f72585',borderDash:[6,5],tension:.35,pointRadius:4}];
  $('t1').textContent='Sales trend with 3-month forecast (NumPy-style linear regression)'}
 else{ds=C.map(c=>({label:D.cats[c],data:ms.map(m=>S([m],[c],R,k)),backgroundColor:D.colors[c]+'cc',borderColor:D.colors[c],fill:true,tension:.35,pointRadius:0}));
  $('t1').textContent='Category contribution over time (stacked area)'}
 c1.data={labels,datasets:ds};c1.options.scales.y.stacked=c1.options.scales.x.stacked=st.mode==='area';c1.update();
 const idx=D.cats.map((_,i)=>i).filter(i=>C.includes(i));
 c2.data={labels:idx.map(i=>D.cats[i]),datasets:[{data:idx.map(i=>catT[i]),backgroundColor:idx.map(i=>D.colors[i]),borderRadius:8}]};c2.update();
 c3.data={labels:D.regs,datasets:C.map(c=>({label:D.cats[c],backgroundColor:D.colors[c],borderRadius:5,data:D.regs.map((_,r)=>S(ms,[c],[r],k))}))};c3.update();
 c4.data={labels:idx.map(i=>D.cats[i]),datasets:[{data:idx.map(i=>catT[i]),backgroundColor:idx.map(i=>D.colors[i]),borderWidth:2}]};c4.update();
 const sg=D.segs.map((_,s)=>ms.reduce((a,m)=>a+D.segRev[s][m],0));
 c5.data={labels:D.segs,datasets:[{data:sg,backgroundColor:['#00b4d8','#ffb703','#f72585','#6c5ce7'],borderWidth:2}]};c5.update();
 // top-5 table
 const rows=[];C.forEach(c=>R.forEach(r=>rows.push([D.cats[c]+' \u2014 '+D.regs[r],S(ms,[c],[r],'qty'),S(ms,[c],[r],'rev')])));
 rows.sort((a,b)=>b[2]-a[2]);
 $('tbl').innerHTML='<tr><th>Category \u2014 Region</th><th>Units</th><th>Revenue</th><th>Share</th></tr>'+
  rows.slice(0,5).map(r=>`<tr><td>${r[0]}</td><td>${r[1].toLocaleString('en-IN')}</td><td>${fm(r[2],'rev')}</td><td>${(r[2]/(totR||1)*100).toFixed(1)}%</td></tr>`).join('');
 const br=regT.indexOf(Math.max(...regT));
 $('ins').innerHTML=C.length&&ms.length?`&#9679; Peak month is <i>${D.months[ms[bi]]}</i> with <i>${fm(monthly[bi],k)}</i>.<br>&#9679; <i>${D.cats[tc]}</i> leads with <i>${(catT[tc]/(catT.reduce((a,b)=>a+b,0)||1)*100).toFixed(1)}%</i> of the selection.<br>&#9679; Strongest region: <i>${D.regs[br]}</i>.<br>&#9679; Sales moved <i>${g>=0?'up':'down'} ${Math.abs(g).toFixed(1)}%</i> from ${D.months[ms[0]]} to ${D.months[ms.at(-1)]}.`:'Select at least one category.';
 window._csv=[['Month','Value']].concat(ms.map((m,i)=>[D.months[m],Math.round(monthly[i])]));
}
const bind=(id,f)=>$(id).onclick=()=>{f();update()};
$('reg').onchange=e=>{st.reg=e.target.value;update()};
$('from').onchange=e=>{st.from=+e.target.value;if(st.from>st.to){st.to=st.from;$('to').value=st.to}update()};
$('to').onchange=e=>{st.to=+e.target.value;if(st.to<st.from){st.from=st.to;$('from').value=st.from}update()};
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{const i=+b.dataset.i;b.classList.toggle('on');
 st.cats=st.cats.includes(i)?st.cats.filter(x=>x!==i):st.cats.concat(i).sort();update()});
const tg=(a,b,key,va,vb)=>{bind(a,()=>{st[key]=va;$(a).classList.add('on');$(b).classList.remove('on')});bind(b,()=>{st[key]=vb;$(b).classList.add('on');$(a).classList.remove('on')})};
tg('m_rev','m_qty','metric','rev','qty');tg('t_line','t_area','mode','line','area');
$('reset').onclick=()=>{st=JSON.parse(JSON.stringify(def));$('reg').value='all';$('from').value=0;$('to').value=11;
 document.querySelectorAll('.chip').forEach(b=>b.classList.add('on'));
 ['m_rev','t_line'].forEach(i=>$(i).classList.add('on'));['m_qty','t_area'].forEach(i=>$(i).classList.remove('on'));update()};
$('csv').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([window._csv.join('\n').replace(/,/g,',')],{type:'text/csv'}));a.download='sales_selection.csv';a.click()};
$('theme').onclick=()=>{const d=document.documentElement.dataset.theme==='dark';document.documentElement.dataset.theme=d?'':'dark';
 $('theme').innerHTML=d?'&#127769; Dark mode':'&#9728; Light mode';Chart.defaults.color=d?'#1b2a49':'#e8edff';[c1,c2,c3,c4,c5].forEach(c=>c.update())};
Chart.defaults.color='#1b2a49';update();
</script></body></html>"""

with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(PAGE.replace("__DATA__", json.dumps(data)))

print(f"\nInteractive dashboard : {os.path.join(os.path.abspath(OUT_DIR), 'index.html')}")
print("Matplotlib charts     : 6 PNG files + dashboard.png in the same folder")
