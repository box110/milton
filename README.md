# Milton

Fits the weights that shrub's research pipeline currently guesses, and proposes
better ones for approval.

shrub scores and ranks evidence with hardcoded constants — how many points an
oversold RSI is worth, how fast a news article decays relative to a 10-K, how
much more an SEC filing counts than a Reddit post. Those numbers were guesses.
Milton measures them against what actually happened and searches for better
ones, under a promotion gate strict enough that most candidates don't ship.

## Why this isn't darwin

Darwin evolved agents: open-ended genomes, LLM-in-the-loop fitness, one equity
curve per evaluation, and differences between candidates smaller than the
run-to-run noise. It churned because it was re-selecting on noise.

Milton optimises a small vector of continuous parameters against a
**per-decision** objective. Each evaluation yields one number per *day* rather
than one number per *run*, so candidates are actually distinguishable, and a
t-stat comes for free.

## The objective: mean daily rank IC

Spearman correlation between predicted ranking and realised forward alpha,
computed **within each day** and then averaged. Rank because forward returns
are fat-tailed and one buyout shouldn't set the weights. Per-day because
pooling mixes ordering skill with market beta — on shrub's own data the two
disagree in sign.

Primary horizon is 20 days, where the incumbent weights show a real edge.
5-day is a secondary check against overfitting one horizon.

## Status

Screener weights first, then RAG source-reliability weights.

Verified against live shrub data (7,519 labelled picks):

```
replay fidelity : 7519/7519 exact
incumbent  1d   : IC=-0.0802 IR=-0.33 t=-1.38 over 18 days
incumbent  5d   : IC=-0.0462 IR=-0.14 t=-0.61 over 18 days
incumbent 20d   : IC=+0.1377 IR=+0.54 t=+2.30 over 18 days   <- the bar to beat
```

Replay fidelity is the gate on everything else: milton re-scores every pick
from its logged signals and must reproduce the score shrub actually stored. It
does, exactly, for all 7,519 — including the MACD term, which shrub never
logged and which is recovered as a residual.

| module | does |
|---|---|
| `screener.py` | shrub's scoring function with the constants as parameters |
| `features.py` | logged picks + forward alpha -> labelled rows |
| `objective.py` | mean daily rank IC, IR, t-stat |
| `optimize.py` | coordinate descent from the incumbent, with shrinkage |
| `gate.py` | chronological holdout, paired against the incumbent |
| `proposals.py` | proposal + decision store (SQLite; shrub's DB is read-only) |
| `approval.py` | strict parsing of an emailed decision |
| `notify.py` | renders the ask; shrub sends it |
| `rag.py` | retrieval source weights + top-k evidence scoring |
| `ragfit.py` | source-weight search and its gate |
| `db.py` | read-only access to shrub's MySQL |
| `web.py` + `static/` | read-only UI over the loop |

Latest fit, on live data — **rejected**, which is the gate working:

```
in sample    incumbent +0.0845   candidate +0.3761
out of sample incumbent +0.2335   candidate +0.4694
paired       +0.2359 per day, t = +3.55
verdict      REJECT - only 7 usable test days, need 12
```

The search wants to drop the volume bonus and SMA-support term to zero, halve
the MACD crossover weight, and roughly double both RSI terms. The improvement
survives the holdout, but seven test days cannot tell that apart from luck, so
it does not ship. It becomes decidable as days accumulate.

## Approval

On a PROMOTE, `scripts/propose.py --send` raises a proposal and emails it,
tagged `[milton #N]`. Replying APPROVE or REJECT on the first line records the
decision. Nothing is emailed on a rejection — a notice every run trains you to
ignore the mail, and the UI already shows what the gate has been turning down.

Milton owns no mail credentials and never will. shrub has a working mailbox, so
it sends on milton's behalf and forwards replies back. Those replies are
intercepted by subject tag **before** shrub's operator-to-CEO branch: without
that, a reply saying "approve" would reach the CEO agent and be read as a
directive about trading.

Parsing is deliberately narrow. An approval word must open a line; quoted
history is ignored; "not yet", a question, or a reply containing both words all
decide nothing. Decisions are final and idempotent, and proposals expire after
seven days rather than being honoured late.

## Retrieval source weights

Fitting these is a harder problem than the screener's, and the limits are worth
stating rather than discovering later.

There is no label. Nothing records whether a retrieved chunk was the right one,
so the question is reframed into one the data can answer: *does evidence from
source S, near date D, carry information about ticker T's subsequent return?*
That measures **predictiveness, not truthfulness**. A scrupulous source scores
nothing here if what it reports is already priced; a junk source scores well if
it moves crowds. For a ranking that feeds a return-seeking process that is the
defensible criterion — but a weight fitted this way must never be described as
a measure of whether a source tells the truth.

Scoring takes the **top 3** chunks, matching `_corpus_evidence_block`, not a
sum. Summing was the obvious first attempt and it was wrong: social is 91% of
the ticker-tagged corpus, so a sum made the score a mention count wearing a
reliability weight. What a weight actually decides is which few chunks get
read.

Result on shrub's corpus — **rejected, and informatively so**:

```
incumbent  train +0.0136   test +0.2176
candidate  train +0.0474   test +0.0682
paired     -0.1494, t = -2.63 over 9 test days
```

The search wanted to downweight `sec_filing` from 1.0 to 0.4. It improved
in-sample and halved the held-out IC — textbook overfitting, caught. That is
weak evidence *for* the incumbent ordering, at least for filings outranking
everything else.

Only three sources are fittable at all (`sec_filing`, `news`, `social`);
`newsletter` ingestion began after the labelled window closes. Twenty-two
distinct days is far too few to conclude anything, which the gate says.

Not built yet: writing approved weights into shrub.

## Running

```bash
cp .env.example .env          # fill in the read-only DB password
mysql < scripts/create_ro_user.sql
python scripts/baseline.py    # replay fidelity + incumbent IC
pytest tests/ -q
docker compose up -d --build  # UI on :8200, survives reboots
```

`scripts/run_ui.sh` runs the same app ad hoc against the shrub-app image if you
want it without a build.

The UI binds all interfaces and is reachable across the LAN and VPN. **There is
no authentication in front of it** — that matches how shrub's own debug ports
are exposed, and it is why nothing here should be published beyond the local
network. shrub's Caddy is LAN-only with an internal CA, so no part of this
stack faces the internet.

The UI shows the loop stage by stage, the incumbent's measured IC, and the
per-day dispersion behind it. Stages that aren't built report `built: false`
and render dashed — a scaffold that shows plausible placeholder numbers is
worse than no scaffold, because you start reasoning about results nothing
computed.

Milton holds SELECT-only credentials. Approved weights reach shrub through its
own API, never through this connection.
