/* Momentum & Reversal Engine — client logic.
 *
 * Everything here is Tier 3: derived in the browser, on every page load, from the
 * static JSON. No fetching of market data, no backend, no keys. If a number moves
 * on screen without the JSON changing, it was computed here.
 */
'use strict';

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = { latest: null, backtest: null, meta: null, tracking: null,
                model: 'all', q: '', tab: 'forward', sort: {} };

/* ───────────────────────── formatting ───────────────────────── */

// Columns that are counts/ratios, not percentages.
const RAW = new Set(['model','horizon_days','signals','trades','profit_factor','avg_hold_days',
                     'avg_R','year','period','regime','symbol','rank','days_to_first_hit',
                     'days_elapsed','first_hit','as_of','entry_date','entry','industry','setup']);

const pct  = (v, d = 1) => v == null ? '—' : (v * 100).toFixed(d) + '%';
const sgn  = (v, d = 1) => v == null ? '—' : (v > 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
const num  = (v, d = 2) => v == null ? '—' : Number(v).toFixed(d);
const inr  = v => v == null ? '—' : '₹' + Number(v).toLocaleString('en-IN',
                  { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const cls  = v => v == null ? '' : v > 0 ? 'pos' : v < 0 ? 'neg' : '';

const LABEL = {
  horizon_days:'Horizon', signals:'Signals', avg_return:'Avg return', median_return:'Median',
  hit_rate_positive:'Hit rate', avg_max_upside_MFE:'Avg MFE', median_max_upside_MFE:'Median MFE',
  'share_reaching_+5pct':'Reached +5%', 'share_reaching_+10pct':'Reached +10%',
  avg_max_drawdown_MAE:'Avg MAE', universe_avg_return:'Universe avg',
  universe_avg_MFE:'Universe MFE', avg_excess_vs_universe:'Excess vs universe',
  nifty_avg_return:'Nifty', win_rate:'Win rate', avg_win:'Avg win', avg_loss:'Avg loss',
  expectancy:'Expectancy', profit_factor:'Profit factor', avg_hold_days:'Avg hold',
  target_hit:'Target', stop_hit:'Stop', time_exit:'Time exit', avg_R:'Avg R',
  return_to_date:'Return to date', first_hit:'First hit', days_elapsed:'Days',
  entry_date:'Entry date', as_of:'List date', days_to_first_hit:'Days to hit'
};
const label = k => LABEL[k] || k.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());

function fmtCell(k, v) {
  if (v == null) return '—';
  if (typeof v !== 'number') return String(v);
  if (k === 'horizon_days') return v + 'd';
  if (k === 'avg_hold_days') return num(v, 1) + 'd';
  if (k === 'entry') return inr(v);
  if (k === 'year') return String(v);          // a year is a label, not a quantity: 2021, not 2,021
  if (RAW.has(k)) return Number.isInteger(v) ? v.toLocaleString('en-IN') : num(v, 3);
  return pct(v, 1);
}

/* ───────────────────────── time / freshness ───────────────────────── */

const IST_OFFSET = 5.5 * 3600e3;
const istNow = () => new Date(Date.now() + IST_OFFSET);

function ageOf(iso) {
  if (!iso) return null;
  const mins = (Date.now() - new Date(iso).getTime()) / 60000;
  if (mins < 60)   return Math.max(0, Math.round(mins)) + ' min ago';
  if (mins < 1440) return Math.round(mins / 60) + ' h ago';
  return Math.round(mins / 1440) + ' d ago';
}

/** NSE cash market: Mon–Fri 09:15–15:30 IST. Holidays are not modelled. */
function marketState() {
  const t = istNow();
  const day = t.getUTCDay(), mins = t.getUTCHours() * 60 + t.getUTCMinutes();
  if (day === 0 || day === 6) return { open: false, text: 'Market closed · weekend' };
  if (mins < 555)  return { open: false, text: 'Pre-open · opens 09:15 IST' };
  if (mins > 930)  return { open: false, text: 'Market closed · 15:30 IST' };
  return { open: true, text: 'Market open' };
}

/* ───────────────────────── header + tiles ───────────────────────── */

function renderStamps() {
  const { meta, latest } = state, ms = marketState();
  const stale = meta.backtest_is_stale;
  $('#stamps').innerHTML = `
    <span class="chip ${ms.open ? 'chip-pass' : 'chip-mute'}"><i class="dot"></i>${ms.text}</span>
    <span class="chip">data ${latest.as_of}</span>
    <span class="chip chip-mute">refreshed ${ageOf(latest.generated_at)}</span>
    <span class="chip ${stale ? 'chip-warn' : 'chip-mute'}">rules ${meta.rules_hash}</span>`;
  $('#foot-stamp').textContent = `generated ${latest.generated_at} · rules ${meta.rules_hash}`;
  $('#asof-inline').textContent = latest.as_of;
}

function renderTiles() {
  const L = state.latest, r = L.regime;
  const nMom = L.picks.momentum.length, nRev = L.picks.reversal.length;
  const tiles = [
    { k: 'Regime', v: r.label.toUpperCase(), n: 'Nifty vs 50/200 DMA', c: r.label },
    { k: 'Breadth', v: pct(r.breadth_50dma, 0), n: 'above their 50 DMA' },
    { k: 'Nifty', v: Number(r.nifty).toLocaleString('en-IN', { maximumFractionDigits: 0 }), n: 'benchmark close' },
    { k: 'Universe', v: L.universe.symbols, n: `${L.universe.eligible_today} passed hygiene` },
    { k: 'Momentum', v: `${nMom}/${L.counts.momentum ?? 0}`, n: 'shown / qualified' },
    { k: 'Reversal', v: `${nRev}/${L.counts.reversal ?? 0}`, n: 'shown / qualified' }
  ];
  $('#tiles').innerHTML = tiles.map(t => `
    <div class="tile ${t.c || ''}">
      <div class="k">${t.k}</div><div class="v">${t.v}</div><div class="n">${t.n}</div>
    </div>`).join('');
}

/* ───────────────────────── verdicts ───────────────────────── */
/* The README fixes these pass-marks in advance. We only evaluate them. */

function verdictFor(model) {
  const B = state.backtest;
  const fw = B.forward.filter(r => r.model === model);
  const tr = B.trades.find(r => r.model === model);
  const ctl = B.trades.find(r => r.model === 'control_naive_low');
  const per = B.by_period.filter(r => r.model === model);
  const out = [];

  const ex = fw.map(r => r.avg_excess_vs_universe).filter(v => v != null);
  out.push({
    ok: ex.length > 0 && ex.every(v => v > 0),
    q: 'Do picks beat <em>any</em> liquid stock?',
    m: fw.map(r => `${r.horizon_days}d ${sgn(r.avg_excess_vs_universe, 2)}`).join('  ·  ') || 'no data',
    pass: 'excess > 0 at 5, 10 and 20 days'
  });

  const f20 = fw.find(r => r.horizon_days === 20);
  out.push({
    ok: !!f20 && f20.avg_max_upside_MFE > f20.universe_avg_MFE,
    q: 'Is there upside to capture?',
    m: f20 ? `MFE ${pct(f20.avg_max_upside_MFE)} vs universe ${pct(f20.universe_avg_MFE)}` : 'no data',
    pass: 'clearly above the universe average'
  });

  out.push({
    ok: !!tr && tr.expectancy > 0 && tr.profit_factor >= 1.2,
    mid: !!tr && tr.expectancy > 0 && tr.profit_factor < 1.2,
    q: 'Do the trades make money after costs?',
    m: tr ? `expectancy ${sgn(tr.expectancy, 2)} · PF ${num(tr.profit_factor)}` : 'no trades',
    pass: 'expectancy > 0 and PF ≥ 1.2'
  });

  const sgns = per.map(p => Math.sign(p.expectancy));
  out.push({
    ok: sgns.length > 1 && sgns.every(s => s === sgns[0] && s > 0),
    q: 'Does it hold out-of-sample?',
    m: per.map(p => `${p.period.replace(/ (in|out)-.*$/, '')} ${sgn(p.expectancy, 2)}`).join('  ·  ') || 'no data',
    pass: 'same sign in both periods'
  });

  if (model === 'reversal') out.push({
    ok: !!tr && !!ctl && tr.profit_factor > ctl.profit_factor,
    q: 'Does the filter beat buying lows blindly?',
    m: tr && ctl ? `PF ${num(tr.profit_factor)} vs control ${num(ctl.profit_factor)}` : 'no data',
    pass: 'clearly better than control_naive_low'
  });
  else out.push({
    ok: !!tr && !!ctl && tr.profit_factor > ctl.profit_factor,
    q: 'Does it beat the naive control?',
    m: tr && ctl ? `PF ${num(tr.profit_factor)} vs control ${num(ctl.profit_factor)}` : 'no data',
    pass: 'better than the baseline it should beat'
  });

  out.push({
    ok: !!tr && tr.trades >= 300,
    q: 'Is there enough evidence?',
    m: tr ? `${tr.trades.toLocaleString('en-IN')} trades` : '0 trades',
    pass: '≥ 300 trades'
  });
  return out;
}

function renderVerdicts() {
  const models = [['momentum', 'tag-mom'], ['reversal', 'tag-rev']];
  $('#verdict-cards').innerHTML = models.map(([m, tag]) => {
    const checks = verdictFor(m);
    const passed = checks.filter(c => c.ok).length;
    const chip = passed === checks.length ? 'chip-pass' : passed >= checks.length - 1 ? 'chip-warn' : 'chip-fail';
    return `
    <div class="vcard">
      <header>
        <h3>${m[0].toUpperCase() + m.slice(1)}<span class="tag ${tag}">model</span></h3>
        <span class="chip ${chip}">${passed}/${checks.length} passed</span>
      </header>
      <ul class="checks">
        ${checks.map(c => `
          <li class="${c.ok ? 'ok' : c.mid ? 'mid' : 'no'}">
            <span class="mark">${c.ok ? '✓' : c.mid ? '~' : '✕'}</span>
            <span class="q">${c.q}<span class="m">${c.m}  —  needs ${c.pass}</span></span>
          </li>`).join('')}
      </ul>
    </div>`;
  }).join('');
}

/* ───────────────────────── picks ───────────────────────── */

const PICK_COLS = [
  ['rank', '#'], ['symbol', 'Symbol'], ['industry', 'Industry'], ['setup', 'Setup'],
  ['score', 'Score'], ['last_close', 'Close'], ['stop_ref', 'Stop'], ['risk_pct', 'Risk'],
  ['target_ref', 'Target'], ['reward_to_risk', 'R:R'], ['median_value_cr', '20d ₹cr']
];

function pickCell(k, p) {
  const v = p[k];
  if (v == null) return '—';
  if (k === 'rank')            return `<span class="rank">${v}</span>`;
  if (k === 'symbol')          return v;
  if (k === 'risk_pct')        return pct(v, 2);
  if (k === 'score')           return num(v, 3);
  if (k === 'reward_to_risk')  return num(v, 1) + 'R';
  if (['last_close', 'stop_ref', 'target_ref'].includes(k)) return inr(v);
  if (k === 'median_value_cr') return num(v, 1);
  return String(v);
}

function renderPicks() {
  const L = state.latest, q = state.q.toLowerCase();
  const wanted = state.model === 'all' ? ['momentum', 'reversal'] : [state.model];
  let shown = 0;

  const html = wanted.map(m => {
    const all = L.picks[m] || [];
    const rows = all.filter(p => !q ||
      [p.symbol, p.industry, p.setup].some(x => String(x).toLowerCase().includes(q)));
    shown += rows.length;
    const qualified = L.counts[m] ?? 0;

    const head = `<h3><span class="pill pill-${m === 'momentum' ? 'mom' : 'rev'}">${m}</span>
        <span class="hint">${rows.length} shown · ${qualified} passed the rules today</span></h3>`;

    if (!rows.length) return `<div class="mblock">${head}
        <div class="empty">${all.length
          ? 'No pick matches that filter.'
          : `No stock satisfied every <strong>${m}</strong> rule at this close. That is a result, not a failure — in this regime the filter is doing its job by returning nothing.`}</div></div>`;

    return `<div class="mblock">${head}
      <div class="tscroll"><table>
        <thead><tr>${PICK_COLS.map(([k, t]) =>
          `<th class="${['symbol','industry','setup','rank'].includes(k) ? 'l' : ''}">${t}</th>`).join('')}</tr></thead>
        <tbody>
        ${rows.map((p, i) => `
          <tr class="clickable" data-m="${m}" data-i="${i}">
            ${PICK_COLS.map(([k]) => `<td class="${k === 'symbol' ? 'sym l'
              : ['industry','setup','rank'].includes(k) ? 'l' : 'num'}">${pickCell(k, p)}</td>`).join('')}
          </tr>
          <tr class="why-row" hidden data-for="${m}-${i}">
            <td colspan="${PICK_COLS.length}">
              <div class="lbl">Why it ranked</div><div class="body">${p.why || '—'}</div>
              <div class="lbl">Risk flags</div>
              <div class="body">${(p.risk_flags || '').split(';').filter(Boolean)
                  .map(f => `<span class="flag">${f.trim()}</span>`).join('') || '—'}</div>
              <div class="plan">
                <span><b>entry ref</b> next open</span>
                <span><b>stop</b> ${inr(p.stop_ref)}</span>
                <span><b>target</b> ${inr(p.target_ref)}</span>
                <span><b>risk</b> ${pct(p.risk_pct, 2)} of price</span>
              </div>
            </td>
          </tr>`).join('')}
        </tbody></table></div></div>`;
  }).join('');

  $('#picks').innerHTML = html;
  $('#pickcount').textContent = `${shown} row${shown === 1 ? '' : 's'}`;

  $$('#picks tr.clickable').forEach(tr => tr.addEventListener('click', () => {
    const row = $(`#picks tr[data-for="${tr.dataset.m}-${tr.dataset.i}"]`);
    if (row) row.hidden = !row.hidden;
  }));
}

/* ───────────────────────── generic table ───────────────────────── */

function table(rows, opts = {}) {
  if (!rows || !rows.length) return `<div class="empty">${opts.empty || 'No data yet.'}</div>`;
  const cols = Object.keys(rows[0]);
  const s = state.sort[opts.key] || {};
  let data = [...rows];
  if (s.col) data.sort((a, b) => {
    const x = a[s.col], y = b[s.col];
    if (x == null) return 1; if (y == null) return -1;
    return (typeof x === 'number' ? x - y : String(x).localeCompare(String(y))) * (s.dir || 1);
  });
  return `<div class="tscroll"><table>
    <thead><tr>${cols.map(c => {
      const a = s.col === c ? (s.dir === 1 ? '▲' : '▼') : '';
      return `<th class="sortable ${typeof rows[0][c] === 'string' ? 'l' : ''}"
                data-col="${c}" data-key="${opts.key}">${label(c)} <span class="arr">${a}</span></th>`;
    }).join('')}</tr></thead>
    <tbody>${data.map(r => `<tr>${cols.map(c => {
      const v = r[c], isNum = typeof v === 'number';
      const colour = ['expectancy','avg_excess_vs_universe','avg_return','return_to_date',
                      'ret_5d','ret_10d','ret_20d','avg_R','median'].includes(c) ? cls(v) : '';
      return `<td class="${isNum ? 'num' : 'l'} ${colour}">${fmtCell(c, v)}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function bindSort(root) {
  $$('th.sortable', root).forEach(th => th.addEventListener('click', () => {
    const key = th.dataset.key, col = th.dataset.col, cur = state.sort[key] || {};
    state.sort[key] = { col, dir: cur.col === col && cur.dir === -1 ? 1 : -1 };
    renderEvidence(); renderTracking();
  }));
}

/* ───────────────────────── evidence ───────────────────────── */

const EV_NOTE = {
  forward: 'Every signal the rules produced, entered at the next open. “Excess vs universe” is the honest test: the pick’s return minus the average of every liquid stock on that same date. MFE is the best price reached, not a return you would have captured.',
  trades: 'Rule-based trades with an ATR stop, an R-multiple target and a 20-session time exit, after 0.25% round-trip costs. Compare each model against control_naive_low — a model that cannot beat blind 52-week-low buying has not earned its complexity.',
  by_period: 'The out-of-sample block is the one that counts. A rule that is strong in-sample and weak out-of-sample has been fitted to the past, not discovered in it.',
  by_regime: 'Where the rules break down. Use this to size down rather than to switch them off after the fact.',
  by_year: 'Year-by-year consistency. A single dominant year carrying the whole edge is a warning, not a track record.'
};

function renderEvidence() {
  const B = state.backtest, t = state.tab;
  $('#evbody').innerHTML = table(B[t], { key: t, empty: 'No rows for this view.' });
  $('#evnote').textContent = EV_NOTE[t] || '';
  $('#bt-window').textContent = `${B.window.start} → ${B.window.end}`;
  $('#bt-scans').textContent = B.scan_dates;
  bindSort($('#evbody'));
}

/* ───────────────────────── tracking ───────────────────────── */

function renderTracking() {
  const T = state.tracking;
  // A list published today has no next open yet, so its rows carry no outcome columns.
  // Rendering those as a table would be a column of symbols pretending to be a result.
  const matured = (T?.rows || []).filter(r => r.return_to_date != null);
  const pending = (T?.rows || []).length - matured.length;

  if (!matured.length) {
    $('#trackbody').innerHTML = `<div class="empty">
      <strong>Nothing has matured yet.</strong>
      ${pending ? `${pending} pick${pending === 1 ? '' : 's'} published for
        ${T.rows[0].as_of} — outcomes appear after the next session's open.` : ''}
      Each scheduled run saves that day's list and follows it forward from there.
      This table is the live, out-of-sample record: it accumulates on its own, and unlike the
      backtest above it cannot be re-run, re-fitted or quietly improved.</div>`;
    return;
  }
  $('#trackbody').innerHTML = table(matured, { key: 'track' }) +
    (pending ? `<p class="note">${pending} more pick${pending === 1 ? '' : 's'} published but
      not yet started — they enter at the next open.</p>` : '');
  bindSort($('#trackbody'));
}

/* ───────────────────────── freshness tiers ───────────────────────── */

function renderTiers() {
  const { latest, meta, backtest } = state;
  const tiers = [
    { c: 'var(--ink-3)', t: 'Tier 0 · immutable', h: 'Method, limits, thresholds',
      p: 'Ships with the page. Changes only when the rules themselves are edited.',
      age: `rules ${meta.rules_hash}` },
    { c: 'var(--warn)', t: 'Tier 1 · deliberately frozen', h: 'Backtest evidence',
      p: 'Not recomputed daily. One extra scan date on ' + backtest.scan_dates +
         ' moves nothing, and a backtest that twitches every morning invites tuning on recent data. Rebuilds when the rules change.',
      age: `computed ${backtest.as_of}` },
    { c: 'var(--pass)', t: 'Tier 2 · self-refreshing', h: 'Lists, regime, tracking',
      p: 'A scheduled job runs the rules after each close and commits fresh JSON. This is where "most current" actually lives.',
      age: `refreshed ${ageOf(latest.generated_at)}` },
    { c: 'var(--accent)', t: 'Tier 3 · every page load', h: 'Verdicts, ages, market state',
      p: 'Derived in your browser from the JSON above — no network call, no backend, nothing to keep running.',
      age: 'computed just now' },
    { c: 'var(--line)', t: 'Tier 4 · manual', h: 'Universe constituents',
      p: `Index membership changes on rebalance. Snapshots are archived dated, never overwritten, so the survivorship bias can be corrected later.`,
      age: `${latest.universe.snapshots} snapshot${latest.universe.snapshots === 1 ? '' : 's'} since ${latest.universe.first_snapshot}` }
  ];
  $('#tiers').innerHTML = tiers.map(t => `
    <div class="tier" style="--tc:${t.c}">
      <div class="t">${t.t}</div><h4>${t.h}</h4><p>${t.p}</p><div class="age">${t.age}</div>
    </div>`).join('');
}

/* ───────────────────────── boot ───────────────────────── */

async function load(name, required = true) {
  try {
    const r = await fetch(`data/${name}.json?v=${Date.now()}`, { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch (e) {
    if (required) throw new Error(`could not load data/${name}.json — ${e.message}`);
    return null;
  }
}

async function init() {
  try {
    const [latest, backtest, meta, tracking] = await Promise.all([
      load('latest'), load('backtest'), load('meta'), load('tracking', false)
    ]);
    Object.assign(state, { latest, backtest, meta, tracking });

    renderStamps(); renderTiles(); renderVerdicts();
    renderPicks(); renderEvidence(); renderTracking(); renderTiers();

    $('#q').addEventListener('input', e => { state.q = e.target.value; renderPicks(); });
    $$('.seg button').forEach(b => b.addEventListener('click', () => {
      $$('.seg button').forEach(x => x.classList.toggle('on', x === b));
      state.model = b.dataset.model; renderPicks();
    }));
    $$('#evtabs button').forEach(b => b.addEventListener('click', () => {
      $$('#evtabs button').forEach(x => x.classList.toggle('on', x === b));
      state.tab = b.dataset.tab; renderEvidence();
    }));

    // Keep relative ages honest on a long-lived tab.
    setInterval(renderStamps, 60000);
  } catch (e) {
    $('#main').insertAdjacentHTML('afterbegin',
      `<div class="empty"><strong>Could not load data.</strong> ${e.message}<br>
       Run <code>python tools/export_site.py</code>, then serve this folder
       (<code>python -m http.server</code>) — opening index.html via file:// blocks fetch.</div>`);
  }
}

init();
