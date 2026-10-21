# hrsolar-ens

ECMWF IFS ensemble (ENS, 51 members) read at 22 points over Croatia from
[ECMWF open data on AWS](https://registry.opendata.aws/ecmwf-forecasts/) (CC-BY-4.0), for a
day-ahead PV forecasting study (FER, project Solari). Only the surface solar radiation (`ssrd`,
accumulated J/m² since the start of the run) and 2 m temperature (`2t`) messages (the open-data ENS set has no cloud cover) of the 00 UTC run
at steps 18–48 h are fetched, by byte ranges; nothing else is downloaded or kept.

`data/ens_YYYY-MM.parquet`: run, step, param, point, member (0 = control), value, published_at
(when the step file appeared on the bucket, UTC; the study issues its forecast at 10:00 Zagreb time
and only uses what was available by then).

`backfill.yml` fetches a date range in 30-day chunks (workflow_dispatch); `daily.yml` fetches
today's run every morning. Points: `points.yaml` (nearest 0.25° cell).
