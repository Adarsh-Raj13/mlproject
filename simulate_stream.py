"""
Real-Time IoT Telemetry Stream Simulator.
Simulates a live ESP32 / Industrial IoT sensor node streaming AC telemetry
into the SQLite database with real-time ML diagnosis and alert logging.

Usage:
    python simulate_stream.py                 # Stream every 3 seconds indefinitely
    python simulate_stream.py --interval 2    # Stream every 2 seconds
    python simulate_stream.py --count 20      # Stream 20 readings and stop
"""

import sys
import time
import argparse
from datetime import datetime

from database.database import DatabaseManager
from src.prediction import InferenceEngine
from app import inject_simulation_reading


def run_simulator(interval: float = 3.0, count: int = 0):
    print("=" * 65)
    print("⚡ SMART LOAD MONITOR - LIVE IoT TELEMETRY STREAM SIMULATOR")
    print("=" * 65)
    print(f"Target DB:  smart_load_monitor.db")
    print(f"Interval:   {interval} seconds")
    print(f"Mode:       {'Continuous (Ctrl+C to stop)' if count <= 0 else f'{count} packets'}")
    print(f"Dashboard:  http://localhost:8501")
    print("-" * 65)

    db = DatabaseManager()
    engine = InferenceEngine()

    if not engine.is_loaded:
        print("Error: ML models not found. Please run 'python train_models.py' first.")
        sys.exit(1)

    packet_num = 0
    try:
        while True:
            packet_num += 1
            ts = datetime.now().strftime("%H:%M:%S")

            # Inject simulation reading into database and run ML inference
            inject_simulation_reading(db, engine)

            # Retrieve the latest reading that was just inserted
            latest_readings = db.get_latest_readings_for_all_appliances()
            history = db.get_readings_history(limit=1)

            if not history.empty:
                r = history.iloc[0]
                app_name = r["appliance_name"]
                v = r["voltage"]
                i = r["current"]
                p = r["active_power"]
                t = r["temperature"]

                # Quick prediction for terminal display
                diag = engine.predict_single({
                    "Timestamp": str(r["timestamp"]),
                    "Appliance_Type": app_name,
                    "Voltage": v,
                    "Current": i,
                    "Active_Power": p,
                    "Power_Factor": r["power_factor"],
                    "Temperature": t,
                    "Frequency": 50.0,
                    "Operating_Duration": r["operating_duration"]
                })

                color_icon = {
                    "Critical": "🔴 CRITICAL",
                    "High Risk": "🟠 HIGH RISK",
                    "Warning": "🟡 WARNING",
                    "Normal": "🟢 NORMAL"
                }.get(diag["severity"], "⚪")

                print(f"[{ts}] Packet #{packet_num:03d} | {app_name:<16} | "
                      f"{v:5.1f}V | {i:5.2f}A | {p:6.0f}W | {t:4.1f}°C | "
                      f"{color_icon:<13} | Fault: {diag['predicted_fault']}")

            if count > 0 and packet_num >= count:
                print(f"\nCompleted {count} packets.")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nSimulator stopped by operator.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Telemetry Simulator")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between packets (default: 3.0)")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to send (0 = infinite)")
    args = parser.parse_args()

    run_simulator(interval=args.interval, count=args.count)
