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
| `db.py` | read-only access to shrub's MySQL |
| `web.py` + `static/` | read-only UI over the loop |

Not built yet: the optimiser, the out-of-sample promotion gate, the email
approval loop, and the RAG replay harness.

## Running

```bash
cp .env.example .env          # fill in the read-only DB password
mysql < scripts/create_ro_user.sql
python scripts/baseline.py    # replay fidelity + incumbent IC
pytest tests/ -q
./scripts/run_ui.sh           # UI on http://localhost:8200
```

The UI shows the loop stage by stage, the incumbent's measured IC, and the
per-day dispersion behind it. Stages that aren't built report `built: false`
and render dashed — a scaffold that shows plausible placeholder numbers is
worse than no scaffold, because you start reasoning about results nothing
computed.

Milton holds SELECT-only credentials. Approved weights reach shrub through its
own API, never through this connection.
