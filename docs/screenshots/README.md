# Dashboard screenshots

Screenshots are **not committed** to the repo. The Streamlit dashboard
renders client-side, so a static HTTP fetch only returns the JS shell —
automated capture would need a browser driver (Playwright/Selenium),
which is out of scope for the MVP.

To populate this folder for a portfolio write-up, capture them manually:

1. Start the dashboard: `make dashboard` → open `http://localhost:8501`.
2. Capture each of the six pages and save here with these names:

   | File | Page |
   |---|---|
   | `overview.png` | Overview |
   | `absa.png` | ABSA Distribution |
   | `clusters.png` | Issue Clusters |
   | `incidents.png` | Emerging Incidents |
   | `retrieval.png` | Retrieval Search |
   | `data-quality.png` | Data Quality |

3. Reference them from `docs/demo.md` or the main `README.md` as needed.

See `docs/demo.md` for the full demo walkthrough.
