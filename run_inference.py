"""
Command-Line ML Diagnostic Tool.
Allows quick fault diagnosis from the command line for a single reading or CSV file.

Examples:
    python run_inference.py --appliance "Ceiling Fan" --voltage 226 --current 0.65 --temp 68
    python run_inference.py --csv sample_telemetry.csv --out results.csv
"""

import sys
import argparse
import json
import pandas as pd
from datetime import datetime

from src.prediction import InferenceEngine
from src.feature_engineering import APPLIANCE_SPECS


def main():
    parser = argparse.ArgumentParser(description="Smart Load Monitor CLI Inference Tool")
    parser.add_argument("--appliance", type=str, default="Ceiling Fan", choices=list(APPLIANCE_SPECS.keys()),
                        help="Appliance name")
    parser.add_argument("--voltage", type=float, default=230.0, help="Grid Voltage (V)")
    parser.add_argument("--current", type=float, default=None, help="Current drawn (A)")
    parser.add_argument("--power", type=float, default=None, help="Active Power (W)")
    parser.add_argument("--pf", type=float, default=None, help="Power Factor (0.1 to 1.0)")
    parser.add_argument("--temp", type=float, default=35.0, help="Temperature (°C)")
    parser.add_argument("--duration", type=float, default=60.0, help="Operating Duration (mins)")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file for batch diagnosis")
    parser.add_argument("--out", type=str, default=None, help="Output path for batch results CSV")
    args = parser.parse_args()

    engine = InferenceEngine()
    if not engine.is_loaded:
        print("Error: Models are not trained. Please run 'python train_models.py'.")
        sys.exit(1)

    if args.csv:
        print(f"Reading batch telemetry from {args.csv}...")
        df = pd.read_csv(args.csv)
        results = engine.batch_predict(df)
        print(f"\nEvaluated {len(results)} records:")
        print(results[["Appliance_Type", "Voltage", "Current", "Predicted_Fault", "Alert_Severity", "Model_Confidence"]].head(10))

        if args.out:
            results.to_csv(args.out, index=False)
            print(f"\nSaved diagnostic report to {args.out}")
        return

    # Single inference mode
    spec = APPLIANCE_SPECS[args.appliance]
    cur = args.current if args.current is not None else spec["rated_current"]
    pf = args.pf if args.pf is not None else spec["nominal_pf"]
    p = args.power if args.power is not None else round(args.voltage * cur * pf, 1)

    reading = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Appliance_Type": args.appliance,
        "Voltage": args.voltage,
        "Current": cur,
        "Active_Power": p,
        "Power_Factor": pf,
        "Frequency": 50.0,
        "Temperature": args.temp,
        "Operating_Duration": args.duration
    }

    res = engine.predict_single(reading)

    print("\n" + "=" * 55)
    print("⚡ SMART LOAD MONITOR - DIAGNOSTIC REPORT")
    print("=" * 55)
    print(f"Appliance:           {args.appliance}")
    print(f"Grid Voltage:        {args.voltage:.1f} V")
    print(f"Drawn Current:       {cur:.2f} A (Rated: {spec['rated_current']} A)")
    print(f"Active Power:        {p:.1f} W (Rated: {spec['rated_power']} W)")
    print(f"Temperature:         {args.temp:.1f} °C (Max safe: {spec['max_safe_temp']} °C)")
    print(f"Power Factor:        {pf:.2f}")
    print("-" * 55)
    print(f"Predicted Fault:     {res['predicted_fault']}")
    print(f"Model Confidence:    {res['confidence'] * 100:.1f}%")
    print(f"Alert Severity:      {res['severity']} ({res['severity_color']})")
    print(f"Anomaly Score:       {res['anomaly_score']:.3f} (Is Anomaly: {res['is_anomaly']})")
    print(f"Load Profile:        {res['load_profile_label']} ({res['consumption_category']})")
    print(f"Abnormal Parameters: {', '.join(res['abnormal_parameters']) if res['abnormal_parameters'] else 'None (Nominal)'}")
    print(f"Action:              {res['corrective_action']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
