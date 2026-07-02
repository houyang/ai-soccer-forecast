# FIFA 2026 World Cup — Predictions (after group stage)

Group stage complete. Below: final group standings, all 16 Round-of-32 predictions, and a Monte-Carlo bracket simulation to the champion.

> **Lineup provenance:** when `API_FOOTBALL_KEY` is set, each side's most-recent
> played World Cup lineup is fetched live (source: `live`); otherwise lineups
> are projected from a curated formation table + squad ratings (source: `projected`).
> This committed file was generated with live lineups; regenerate offline for the
> projected variant. R32 pairings are the real fixtures; the R16→Final bracket
> pairing is approximated (sorted by fixture_id).

## Group standings

### Group A

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Mexico | 3 | 3 | 0 | 0 | 6 | 0 | 6 | 9 |
| South Africa | 3 | 1 | 1 | 1 | 2 | 3 | -1 | 4 |
| South Korea | 3 | 1 | 0 | 2 | 2 | 3 | -1 | 3 |
| Czech Republic | 3 | 0 | 1 | 2 | 2 | 6 | -4 | 1 |

### Group B

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Switzerland | 3 | 2 | 1 | 0 | 7 | 3 | 4 | 7 |
| Canada | 3 | 1 | 1 | 1 | 8 | 3 | 5 | 4 |
| Bosnia & Herzegovina | 3 | 1 | 1 | 1 | 5 | 6 | -1 | 4 |
| Qatar | 3 | 0 | 1 | 2 | 2 | 10 | -8 | 1 |

### Group C

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Brazil | 3 | 2 | 1 | 0 | 7 | 1 | 6 | 7 |
| Morocco | 3 | 2 | 1 | 0 | 6 | 3 | 3 | 7 |
| Scotland | 3 | 1 | 0 | 2 | 1 | 4 | -3 | 3 |
| Haiti | 3 | 0 | 0 | 3 | 2 | 8 | -6 | 0 |

### Group D

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| USA | 3 | 2 | 0 | 1 | 8 | 4 | 4 | 6 |
| Australia | 3 | 1 | 1 | 1 | 2 | 2 | 0 | 4 |
| Paraguay | 3 | 1 | 1 | 1 | 2 | 4 | -2 | 4 |
| Türkiye | 3 | 1 | 0 | 2 | 3 | 5 | -2 | 3 |

### Group E

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Germany | 3 | 2 | 0 | 1 | 10 | 4 | 6 | 6 |
| Ivory Coast | 3 | 2 | 0 | 1 | 4 | 2 | 2 | 6 |
| Ecuador | 3 | 1 | 1 | 1 | 2 | 2 | 0 | 4 |
| Curaçao | 3 | 0 | 1 | 2 | 1 | 9 | -8 | 1 |

### Group F

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Netherlands | 3 | 2 | 1 | 0 | 10 | 4 | 6 | 7 |
| Japan | 3 | 1 | 2 | 0 | 7 | 3 | 4 | 5 |
| Sweden | 3 | 1 | 1 | 1 | 7 | 7 | 0 | 4 |
| Tunisia | 3 | 0 | 0 | 3 | 2 | 12 | -10 | 0 |

### Group G

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Belgium | 3 | 1 | 2 | 0 | 6 | 2 | 4 | 5 |
| Egypt | 3 | 1 | 2 | 0 | 5 | 3 | 2 | 5 |
| Iran | 3 | 0 | 3 | 0 | 3 | 3 | 0 | 3 |
| New Zealand | 3 | 0 | 1 | 2 | 4 | 10 | -6 | 1 |

### Group H

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Spain | 3 | 2 | 1 | 0 | 5 | 0 | 5 | 7 |
| Cape Verde Islands | 3 | 0 | 3 | 0 | 2 | 2 | 0 | 3 |
| Uruguay | 3 | 0 | 2 | 1 | 3 | 4 | -1 | 2 |
| Saudi Arabia | 3 | 0 | 2 | 1 | 1 | 5 | -4 | 2 |

### Group I

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| France | 3 | 3 | 0 | 0 | 10 | 2 | 8 | 9 |
| Norway | 3 | 2 | 0 | 1 | 8 | 7 | 1 | 6 |
| Senegal | 3 | 1 | 0 | 2 | 8 | 6 | 2 | 3 |
| Iraq | 3 | 0 | 0 | 3 | 1 | 12 | -11 | 0 |

### Group J

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Argentina | 3 | 3 | 0 | 0 | 8 | 1 | 7 | 9 |
| Austria | 3 | 1 | 1 | 1 | 6 | 6 | 0 | 4 |
| Algeria | 3 | 1 | 1 | 1 | 5 | 7 | -2 | 4 |
| Jordan | 3 | 0 | 0 | 3 | 3 | 8 | -5 | 0 |

### Group K

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| Colombia | 3 | 2 | 1 | 0 | 4 | 1 | 3 | 7 |
| Portugal | 3 | 1 | 2 | 0 | 6 | 1 | 5 | 5 |
| Congo DR | 3 | 1 | 1 | 1 | 4 | 3 | 1 | 4 |
| Uzbekistan | 3 | 0 | 0 | 3 | 2 | 11 | -9 | 0 |

### Group L

| Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|
| England | 3 | 2 | 1 | 0 | 6 | 2 | 4 | 7 |
| Croatia | 3 | 2 | 0 | 1 | 5 | 5 | 0 | 6 |
| Ghana | 3 | 1 | 1 | 1 | 2 | 2 | 0 | 4 |
| Panama | 3 | 0 | 0 | 3 | 0 | 4 | -4 | 0 |

## Round of 32

- `2026-06-28 19:00 UTC`  **South Africa 1-1 Canada**  (W 23% / D 25% / L 52%)  — Eff 41.4 vs 53.1 -> supremacy -0.72; xG 1.03-1.66; adj -1.5/+0.0.
- `2026-06-29 17:00 UTC`  **Brazil 1-1 Japan**  (W 51% / D 24% / L 25%)  — Eff 68.1 vs 54.2 -> supremacy +0.86; xG 1.67-1.09; adj -0.5/-2.0.
- `2026-06-30 01:00 UTC`  **Netherlands 1-1 Morocco**  (W 48% / D 24% / L 28%)  — Eff 66.0 vs 56.8 -> supremacy +0.57; xG 1.68-1.24; adj -1.8/-1.5.
- `2026-07-02 00:00 UTC`  **USA 1-1 Bosnia & Herzegovina**  (W 53% / D 23% / L 23%)  — Eff 60.7 vs 46.6 -> supremacy +0.87; xG 1.81-1.12; adj +4.0/-1.0.
- `2026-06-30 17:00 UTC`  **Ivory Coast 1-1 Norway**  (W 38% / D 25% / L 37%)  — Eff 54.3 vs 54.3 -> supremacy -0.00; xG 1.45-1.41; adj -1.5/-1.0.
- `2026-06-29 20:30 UTC`  **Germany 1-0 Paraguay**  (W 62% / D 22% / L 17%)  — Eff 68.1 vs 47.3 -> supremacy +1.29; xG 1.93-0.89; adj -1.0/-0.5.
- `2026-06-30 21:00 UTC`  **France 1-1 Sweden**  (W 61% / D 21% / L 18%)  — Eff 70.8 vs 50.9 -> supremacy +1.23; xG 2.02-1.01; adj -1.0/-1.0.
- `2026-07-03 18:00 UTC`  **Australia 1-1 Egypt**  (W 28% / D 26% / L 46%)  — Eff 43.9 vs 52.1 -> supremacy -0.51; xG 1.10-1.46; adj -2.0/-1.5.
- `2026-07-03 22:00 UTC`  **Argentina 1-0 Cape Verde Islands**  (W 64% / D 22% / L 14%)  — Eff 65.1 vs 41.2 -> supremacy +1.49; xG 1.86-0.73; adj -0.5/-1.5.
- `2026-07-01 01:00 UTC`  **Mexico 1-0 Ecuador**  (W 49% / D 26% / L 25%)  — Eff 59.8 vs 51.1 -> supremacy +0.54; xG 1.51-0.99; adj +4.0/-0.5.
- `2026-07-01 16:00 UTC`  **England 1-0 Congo DR**  (W 62% / D 22% / L 16%)  — Eff 71.2 vs 46.2 -> supremacy +1.55; xG 1.85-0.81; adj -1.0/-1.5.
- `2026-07-01 20:00 UTC`  **Belgium 1-1 Senegal**  (W 46% / D 24% / L 30%)  — Eff 62.0 vs 54.0 -> supremacy +0.49; xG 1.63-1.27; adj -1.0/-1.5.
- `2026-07-02 23:00 UTC`  **Portugal 1-1 Croatia**  (W 53% / D 24% / L 23%)  — Eff 68.1 vs 54.5 -> supremacy +0.85; xG 1.70-1.03; adj -1.0/-1.0.
- `2026-07-04 01:30 UTC`  **Colombia 1-0 Ghana**  (W 49% / D 26% / L 25%)  — Eff 54.8 vs 43.4 -> supremacy +0.70; xG 1.49-0.97; adj -0.5/-1.5.
- `2026-07-02 19:00 UTC`  **Spain 1-1 Austria**  (W 52% / D 24% / L 24%)  — Eff 64.4 vs 52.3 -> supremacy +0.75; xG 1.67-1.06; adj -1.0/-1.0.
- `2026-07-03 03:00 UTC`  **Switzerland 1-1 Algeria**  (W 46% / D 24% / L 30%)  — Eff 56.1 vs 51.7 -> supremacy +0.27; xG 1.63-1.27; adj -1.0/-1.5.

## Bracket simulation (Monte-Carlo, 10000 iters)

R32 pairings are the real fixtures; R16→Final pairing is approximated (sorted by fixture_id).

**Champion probabilities (top 10):**

- France: 19.9%
- Argentina: 15.2%
- Germany: 11.0%
- England: 9.2%
- Brazil: 7.5%
- Spain: 7.3%
- Portugal: 7.0%
- Netherlands: 6.8%
- Mexico: 2.4%
- Belgium: 2.0%