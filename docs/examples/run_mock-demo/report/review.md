# Review: run_mock-demo

Stop reason: curator_recommended_stop: no remaining deficit in dimensions the engine input can change (technical_quality mean is 5.0)

Final selection needs a human decision. The system shortlist is advice; you may pick any candidate with an image (overrides are recorded).

System shortlist (weighted score + MMR diversity): cand_mock-demo-i0-s0, cand_mock-demo-i0-s1, cand_mock-demo-i1-s3

| candidate | iter | status | QA | weighted | critic | image |
|---|---|---|---|---|---|---|
| cand_mock-demo-i0-s0 | 0 | succeeded | True | 4.15 | heuristic | images/cand_mock-demo-i0-s0/a1/image_0.png |
| cand_mock-demo-i0-s1 | 0 | succeeded | True | 4.00 | heuristic | images/cand_mock-demo-i0-s1/a1/image_0.png |
| cand_mock-demo-i0-s2 | 0 | succeeded | True | 3.85 | heuristic | images/cand_mock-demo-i0-s2/a1/image_0.png |
| cand_mock-demo-i0-s3 | 0 | succeeded | True | 4.00 | heuristic | images/cand_mock-demo-i0-s3/a1/image_0.png |
| cand_mock-demo-i1-s0 | 1 | succeeded | True | 4.15 | heuristic | images/cand_mock-demo-i1-s0/a1/image_0.png |
| cand_mock-demo-i1-s1 | 1 | succeeded | True | 4.00 | heuristic | images/cand_mock-demo-i1-s1/a1/image_0.png |
| cand_mock-demo-i1-s2 | 1 | succeeded | True | 3.85 | heuristic | images/cand_mock-demo-i1-s2/a1/image_0.png |
| cand_mock-demo-i1-s3 | 1 | succeeded | True | 4.00 | heuristic | images/cand_mock-demo-i1-s3/a1/image_0.png |

All ratings come from the offline HEURISTIC critic: pixel-statistic proxies with low confidence. They are not aesthetic judgements and must not be cited as evidence of image quality.

Contact sheet: report/contact_sheet.png

Approve:  python -m atelier approve docs/examples/run_mock-demo --candidate <id> --reviewer <name> [--note ...]
Reject:   python -m atelier reject docs/examples/run_mock-demo --reviewer <name> [--note ...]
