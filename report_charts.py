"""Offline vector charts with explicit units, coverage and missing-value markers."""
import math
import statistics
from reportlab.graphics.shapes import Drawing, Line, Circle, String, Rect
from reportlab.lib import colors

TEAL = colors.HexColor('#168577')
RED = colors.HexColor('#bc473b')
BLUE = colors.HexColor('#346ab0')
GREY = colors.HexColor('#64748b')
GRID = colors.HexColor('#e2e8f0')


def text(d, x, y, value, size=8, color=GREY):
    d.add(String(x, y, str(value), fontSize=size, fillColor=color))


def fmt(value):
    return f'{value:.4g}'


def per_question(label, values, successful):
    d = Drawing(495, 205)
    text(d, 0, 190, label, 11, colors.black)
    text(d, 0, 175, 'Teal: successful | Red: failed/unknown | X: missing | Blue line: successful median')
    valid = [v for v in values if v is not None]
    if not valid:
        text(d, 80, 95, 'No measurements available', 12)
        return d
    top = max(valid) * 1.1 or 1
    left, bottom, width, height = 65, 42, 420, 112
    for i in range(5):
        y = bottom + height * i / 4
        d.add(Line(left, y, left + width, y, strokeColor=GRID))
        text(d, 0, y - 3, fmt(top * i / 4))
    successful_values = [v for v, ok in zip(values, successful) if ok and v is not None]
    if successful_values:
        median = statistics.median(successful_values)
        y = bottom + height * median / top
        d.add(Line(left, y, left + width, y, strokeColor=BLUE, strokeDashArray=[4, 3]))
        text(d, 65, 160, f'Median: {fmt(median)}', color=BLUE)
    n = len(values)
    for i, v in enumerate(values):
        x = left + width * (i + .5) / n
        if v is None:
            d.add(Line(x - 2, bottom - 3, x + 2, bottom + 3, strokeColor=GREY))
            d.add(Line(x - 2, bottom + 3, x + 2, bottom - 3, strokeColor=GREY))
        else:
            color = TEAL if successful[i] else RED
            y = bottom + height * v / top
            if n <= 60:
                d.add(Rect(x - width / n * .3, bottom, width / n * .6, max(0, y-bottom), fillColor=color, strokeColor=None))
            d.add(Circle(x, y, 1.8, fillColor=color, strokeColor=None))
    for i in sorted({round(j * (n-1) / min(5, max(1,n-1))) for j in range(min(5,max(1,n-1))+1)}):
        text(d, left + width * (i+.5)/n - 3, 27, i+1)
    text(d, 155, 10, f'Question row in CSV | Valid measurements: {len(valid)}/{n}')
    return d


def histogram(label, values):
    d = Drawing(495, 195)
    text(d, 0, 180, label, 11, colors.black)
    if not values:
        text(d, 65, 100, 'No valid successful measurements', 11)
        return d
    lo, hi = min(values), max(values)
    bins = min(7, max(1, math.ceil(math.sqrt(len(values))))) if hi > lo else 1
    counts = [0] * bins
    for v in values:
        counts[min(bins-1, int((v-lo)/(hi-lo)*bins)) if hi > lo else 0] += 1
    peak = max(counts)
    text(d, 0, 161, f'Successful requests: n={len(values)} | Mean {fmt(statistics.mean(values))} | Median {fmt(statistics.median(values))}')
    text(d, 0, 143, 'Count')
    for tick in sorted({0, peak//2, peak}):
        y = 45 + 85 * tick/peak
        text(d, 30, y-3, tick)
        d.add(Line(65, y, 485, y, strokeColor=GRID))
    for i, count in enumerate(counts):
        x = 65 + i * 420/bins
        h = 85*count/peak
        d.add(Rect(x+2,45,420/bins-4,h,fillColor=TEAL,strokeColor=None))
        text(d,x+420/bins/2-3,49+h,count)
    if bins == 1:
        text(d, 200, 29, f'All values: {fmt(lo)}')
    else:
        for i in range(bins+1):
            text(d, 57+i*420/bins,29,fmt(lo+(hi-lo)*i/bins),7)
    text(d, 65, 10, 'Measurement bins (units in title); heights show number of requests')
    return d


def breakdown(title, entries):
    """Entries contain label and per-request observations; never zero-fill missing."""
    d = Drawing(495, 60 + 32*len(entries))
    h = d.height
    text(d,0,h-15,title,11,colors.black)
    text(d,0,h-33,'Observed sums across all requests; partial coverage is labelled.')
    totals = [sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None for _,vals in entries]
    maximum = max((v for v in totals if v is not None),default=0) or 1
    for i, ((label, vals), total) in enumerate(zip(entries,totals)):
        y = h-66-32*i
        valid = sum(v is not None for v in vals)
        text(d,0,y+7,label,9,colors.black)
        if total is None:
            text(d,100,y+7,f'N/A | valid 0/{len(vals)}')
        else:
            d.add(Rect(100,y,230*total/maximum,19,fillColor=TEAL if valid==len(vals) else colors.HexColor('#d18b28'),strokeColor=None))
            text(d,340,y+7,f'{fmt(total)}'+(' (partial)' if valid<len(vals) else ''),9,colors.black)
            text(d,100,y-12,f'Valid {valid}/{len(vals)}')
    return d
