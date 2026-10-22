"""End-to-end check with two real processes talking over HTTP:

    mock C3 service (port 8001)  <--HTTP--  C4 backend (port 8000)  <--HTTP--  this script

Run from the project root:  python scripts/smoke_test.py
(needs the backend dependencies installed; stops both servers at the end)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
C3_PORT, C4_PORT = 8001, 8000


def wait_for(url: str, tries: int = 60) -> None:
    for _ in range(tries):
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise SystemExit(f"timeout waiting for {url}")


def main() -> int:
    env = {**os.environ, "C3_MODE": "api", "C3_API_URL": f"http://127.0.0.1:{C3_PORT}"}
    mock = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(C3_PORT),
                             "--log-level", "warning"], cwd=ROOT / "mock_c3", env=env)
    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(C4_PORT),
                                "--log-level", "warning"], cwd=ROOT / "backend", env=env)
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
        failures += 0 if ok else 1

    try:
        wait_for(f"http://127.0.0.1:{C3_PORT}/v1/health")
        wait_for(f"http://127.0.0.1:{C4_PORT}/api/health")
        c4 = httpx.Client(base_url=f"http://127.0.0.1:{C4_PORT}", timeout=10)

        h = c4.get("/api/health").json()
        check("backend connected to C3 over HTTP", h["status"] == "ok" and h["c3_mode"] == "api")

        cat = c4.get("/api/catalog").json()
        avail = [c["condition_id"] for c in cat["conditions"] if c["available"]]
        check("4 entries with results (2 standard-cycle, 2 local), 3 coming soon",
              avail == ["WLTP_Urban", "WLTP_Expressway", "SLDC_Urban", "SLDC_Expressway"], str(avail))

        for cond in ("SLDC_Urban", "SLDC_Expressway"):
            t0 = time.perf_counter()
            r = c4.get("/api/comparison", params={"ev_id": "EV01", "condition_id": cond})
            ms = (time.perf_counter() - t0) * 1000
            j = r.json()
            check(f"function 1 {cond}: saving {j['savings']['energy_saving_pct']:.2f}% in {ms:.0f} ms",
                  r.status_code == 200 and j["savings"]["energy_saving_wh_per_km"] > 0 and ms < 2000)

        j = c4.get("/api/conditions/compare", params={"ev_id": "EV01"}).json()
        check("function 2 condition comparison", len(j["conditions"]) == 2
              and {r["condition_id"]: r["benchmark_energy_wh_per_km"] for r in j["conditions"]}
              == {"SLDC_Urban": 152.0, "SLDC_Expressway": 171.0})
        j = c4.get("/api/sensitivity", params={"ev_id": "EV01", "condition_id": "SLDC_Urban"}).json()
        check("function 2 sensitivity", j["cd_check"]["within_uncertainty"] is False)
        check("data quality", c4.get("/api/data-quality").json()["status"] == "pass")
        live = {t["id"]: t["status"] for t in c4.get("/api/statistics/live").json()["tests"]}
        check("statistics: t-test ready, ANOVA locked (1 value per condition)",
              live["anova"] == "locked" and live["paired_t"] == "ready", str(live))
        check("statistics sample runs", "anova" in c4.get("/api/statistics/sample").json())

        # C3 goes down: backend must report it clearly, not crash
        mock.terminate()
        mock.wait(timeout=10)
        r = c4.post("/api/refresh")
        check("C3 down -> clear 502 error", r.status_code == 502, r.text[:90])
        check("backend still answers from its last good data",
              c4.get("/api/comparison", params={"ev_id": "EV01", "condition_id": "SLDC_Urban"}).status_code == 200)
    finally:
        for p in (mock, backend):
            if p.poll() is None:
                p.terminate()
        for p in (mock, backend):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())