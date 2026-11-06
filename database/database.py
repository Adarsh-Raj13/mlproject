"""
SQLite Database Connection Manager for Smart Load Profiling & Fault Alert System.
Provides persistent storage, retrieval, and telemetry querying for appliances,
historical sensor readings, ML predictions, and safety alert logs.
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class DatabaseManager:
    """Thread-safe SQLite database manager for electrical telemetry and alerts."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, "smart_load_monitor.db")
        else:
            self.db_path = db_path

        self._lock = threading.RLock()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a SQLite connection configured for row dict access."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initializes tables and seeds default appliances if empty."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()

                # 1. Appliances Table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS appliances (
                    appliance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    appliance_name VARCHAR(100) UNIQUE NOT NULL,
                    appliance_type VARCHAR(100) NOT NULL,
                    rated_voltage FLOAT NOT NULL DEFAULT 230.0,
                    rated_power FLOAT NOT NULL,
                    rated_current FLOAT NOT NULL,
                    nominal_pf FLOAT NOT NULL,
                    max_safe_temp FLOAT NOT NULL,
                    location VARCHAR(150),
                    installation_date VARCHAR(50),
                    operating_status VARCHAR(50) DEFAULT 'Online',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """)

                # 2. Electrical Readings Table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS electrical_readings (
                    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    appliance_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    voltage FLOAT NOT NULL,
                    current FLOAT NOT NULL,
                    active_power FLOAT NOT NULL,
                    reactive_power FLOAT NOT NULL,
                    apparent_power FLOAT NOT NULL,
                    power_factor FLOAT NOT NULL,
                    frequency FLOAT NOT NULL DEFAULT 50.0,
                    temperature FLOAT NOT NULL,
                    operating_duration FLOAT NOT NULL DEFAULT 0.0,
                    energy_consumption FLOAT NOT NULL DEFAULT 0.0,
                    is_simulated BOOLEAN DEFAULT 0,
                    FOREIGN KEY(appliance_id) REFERENCES appliances (appliance_id) ON DELETE CASCADE
                );
                """)

                # 3. Alerts Table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    appliance_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    alert_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(50) NOT NULL,
                    description TEXT NOT NULL,
                    detected_parameter VARCHAR(100) NOT NULL,
                    parameter_value FLOAT NOT NULL,
                    threshold_value FLOAT,
                    recommended_action TEXT NOT NULL,
                    acknowledgement_status VARCHAR(50) NOT NULL DEFAULT 'Active',
                    resolved_at DATETIME,
                    FOREIGN KEY(appliance_id) REFERENCES appliances (appliance_id) ON DELETE CASCADE
                );
                """)

                # 4. ML Predictions Table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    appliance_id INTEGER NOT NULL,
                    reading_id INTEGER,
                    timestamp DATETIME NOT NULL,
                    predicted_load_category VARCHAR(100) NOT NULL,
                    predicted_fault VARCHAR(100) NOT NULL,
                    anomaly_status VARCHAR(50) NOT NULL,
                    anomaly_score FLOAT NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    model_version VARCHAR(50) DEFAULT 'v1.0',
                    abnormal_parameters TEXT,
                    corrective_action TEXT,
                    FOREIGN KEY(appliance_id) REFERENCES appliances (appliance_id) ON DELETE CASCADE,
                    FOREIGN KEY(reading_id) REFERENCES electrical_readings (reading_id) ON DELETE SET NULL
                );
                """)

                # Check if appliances table needs initial seed
                cur.execute("SELECT COUNT(*) FROM appliances;")
                if cur.fetchone()[0] == 0:
                    default_appliances = [
                        ("Ceiling Fan", "Inductive Motor", 230.0, 75.0, 0.40, 0.82, 60.0, "Living Room / Lab Area"),
                        ("Refrigerator", "Cyclic Inductive", 230.0, 160.0, 0.95, 0.76, 65.0, "Kitchen / Utility"),
                        ("Air Conditioner", "Heavy Inductive", 230.0, 1800.0, 8.50, 0.92, 75.0, "Master Suite / Server Room"),
                        ("Washing Machine", "Dynamic Mixed", 230.0, 600.0, 3.20, 0.81, 65.0, "Laundry Room"),
                        ("Electric Iron", "Resistive Heating", 230.0, 1200.0, 5.22, 0.99, 85.0, "Utility Wardrobe"),
                        ("Water Heater", "Resistive Immersion", 230.0, 2000.0, 8.70, 0.99, 78.0, "Bathroom / Boiler Room"),
                    ]
                    cur.executemany("""
                    INSERT INTO appliances (appliance_name, appliance_type, rated_voltage, rated_power,
                                            rated_current, nominal_pf, max_safe_temp, location)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """, default_appliances)

                conn.commit()

    def get_appliances(self) -> List[Dict[str, Any]]:
        """Fetches all registered appliances."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                SELECT appliance_id, appliance_name, appliance_type, rated_voltage,
                       rated_power, rated_current, nominal_pf, max_safe_temp, location, operating_status
                FROM appliances ORDER BY appliance_id ASC;
                """)
                rows = cur.fetchall()
                return [
                    {
                        "id": row["appliance_id"],
                        "name": row["appliance_name"],
                        "type": row["appliance_type"],
                        "category": row["appliance_type"],
                        "rated_voltage": float(row["rated_voltage"]),
                        "rated_power": float(row["rated_power"]),
                        "rated_current": float(row["rated_current"]),
                        "nominal_pf": float(row["nominal_pf"]),
                        "max_safe_temp": float(row["max_safe_temp"]),
                        "location": row["location"] or "Unassigned",
                        "status": row["operating_status"] or "Online"
                    }
                    for row in rows
                ]

    def get_appliance_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves single appliance metadata by name."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                SELECT appliance_id, appliance_name, appliance_type, rated_voltage,
                       rated_power, rated_current, nominal_pf, max_safe_temp, location, operating_status
                FROM appliances WHERE appliance_name = ?;
                """, (name,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row["appliance_id"],
                    "name": row["appliance_name"],
                    "type": row["appliance_type"],
                    "category": row["appliance_type"],
                    "rated_voltage": float(row["rated_voltage"]),
                    "rated_power": float(row["rated_power"]),
                    "rated_current": float(row["rated_current"]),
                    "nominal_pf": float(row["nominal_pf"]),
                    "max_safe_temp": float(row["max_safe_temp"]),
                    "location": row["location"] or "Unassigned",
                    "status": row["operating_status"] or "Online"
                }

    def get_system_summary_stats(self) -> Dict[str, Any]:
        """Calculates system-wide operational KPIs and health score."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()

                cur.execute("SELECT COUNT(*) FROM appliances;")
                total_apps = cur.fetchone()[0]

                cur.execute("""
                SELECT COUNT(*) FROM alerts
                WHERE severity = 'Critical' AND acknowledgement_status IN ('Active', 'Acknowledged');
                """)
                crit_alerts = cur.fetchone()[0]

                cur.execute("""
                SELECT COUNT(*) FROM alerts
                WHERE severity IN ('High Risk', 'Warning') AND acknowledgement_status IN ('Active', 'Acknowledged');
                """)
                warn_alerts = cur.fetchone()[0]

                # Health score calculation
                health = max(10, min(100, int(100 - (crit_alerts * 15 + warn_alerts * 5))))

                # Calculate current active power and avg PF across latest readings
                latest = self.get_latest_readings_for_all_appliances()
                if latest:
                    total_power = sum(r.get("active_power", 0.0) or 0.0 for r in latest)
                    pfs = [r.get("power_factor", 0.85) or 0.85 for r in latest]
                    avg_pf = sum(pfs) / len(pfs) if pfs else 0.88
                else:
                    total_power = 0.0
                    avg_pf = 0.88

                return {
                    "total_appliances": total_apps,
                    "critical_alerts": crit_alerts,
                    "warning_alerts": warn_alerts,
                    "overall_health_score": health,
                    "total_active_power": f"{total_power:,.0f}",
                    "average_power_factor": round(avg_pf, 2)
                }

    def get_latest_readings_for_all_appliances(self) -> List[Dict[str, Any]]:
        """Returns the most recent electrical reading for each monitored appliance."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                SELECT a.appliance_id, a.appliance_name, a.appliance_type, a.rated_power,
                       a.rated_current, a.nominal_pf, a.max_safe_temp, a.location,
                       r.reading_id, r.timestamp, r.voltage, r.current, r.active_power,
                       r.reactive_power, r.apparent_power, r.power_factor, r.frequency,
                       r.temperature, r.operating_duration, r.energy_consumption
                FROM appliances a
                LEFT JOIN electrical_readings r ON r.reading_id = (
                    SELECT r2.reading_id FROM electrical_readings r2
                    WHERE r2.appliance_id = a.appliance_id
                    ORDER BY r2.timestamp DESC, r2.reading_id DESC LIMIT 1
                )
                ORDER BY a.appliance_id ASC;
                """)
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "appliance_id": row["appliance_id"],
                        "appliance_name": row["appliance_name"],
                        "name": row["appliance_name"],
                        "category": row["appliance_type"],
                        "rated_power": float(row["rated_power"]),
                        "rated_current": float(row["rated_current"]),
                        "nominal_pf": float(row["nominal_pf"]),
                        "max_safe_temp": float(row["max_safe_temp"]),
                        "location": row["location"] or "Unassigned",
                        "voltage": float(row["voltage"]) if row["voltage"] is not None else 230.0,
                        "current": float(row["current"]) if row["current"] is not None else float(row["rated_current"]),
                        "active_power": float(row["active_power"]) if row["active_power"] is not None else float(row["rated_power"]),
                        "reactive_power": float(row["reactive_power"]) if row["reactive_power"] is not None else 0.0,
                        "apparent_power": float(row["apparent_power"]) if row["apparent_power"] is not None else float(row["rated_power"]),
                        "power_factor": float(row["power_factor"]) if row["power_factor"] is not None else float(row["nominal_pf"]),
                        "temperature": float(row["temperature"]) if row["temperature"] is not None else 32.0,
                        "timestamp": row["timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                return results

    def get_recent_alerts(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves recent alerts with optional status filtering."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                if status and status != "All Statuses":
                    cur.execute("""
                    SELECT al.alert_id, al.appliance_id, a.appliance_name, al.timestamp,
                           al.alert_type, al.severity, al.description, al.detected_parameter,
                           al.parameter_value, al.threshold_value, al.recommended_action,
                           al.acknowledgement_status
                    FROM alerts al
                    JOIN appliances a ON al.appliance_id = a.appliance_id
                    WHERE al.acknowledgement_status = ?
                    ORDER BY al.timestamp DESC, al.alert_id DESC
                    LIMIT ?;
                    """, (status, limit))
                else:
                    cur.execute("""
                    SELECT al.alert_id, al.appliance_id, a.appliance_name, al.timestamp,
                           al.alert_type, al.severity, al.description, al.detected_parameter,
                           al.parameter_value, al.threshold_value, al.recommended_action,
                           al.acknowledgement_status
                    FROM alerts al
                    JOIN appliances a ON al.appliance_id = a.appliance_id
                    ORDER BY al.timestamp DESC, al.alert_id DESC
                    LIMIT ?;
                    """, (limit,))
                rows = cur.fetchall()
                return [
                    {
                        "id": row["alert_id"],
                        "appliance_id": row["appliance_id"],
                        "appliance_name": row["appliance_name"],
                        "timestamp": str(row["timestamp"]),
                        "alert_type": row["alert_type"],
                        "fault_type": row["alert_type"],
                        "severity": row["severity"],
                        "description": row["description"],
                        "trigger_parameter": row["detected_parameter"],
                        "trigger_value": float(row["parameter_value"]),
                        "threshold_value": float(row["threshold_value"]) if row["threshold_value"] is not None else 0.0,
                        "suggested_action": row["recommended_action"],
                        "status": row["acknowledgement_status"]
                    }
                    for row in rows
                ]

    def get_readings_history(self, appliance_name: Optional[str] = None, limit: int = 250) -> pd.DataFrame:
        """Fetches historical readings as a pandas DataFrame."""
        with self._lock:
            with self._get_connection() as conn:
                if appliance_name:
                    query = """
                    SELECT r.reading_id AS id, a.appliance_name, r.timestamp,
                           r.voltage, r.current, r.active_power, r.reactive_power,
                           r.apparent_power, r.power_factor, r.frequency, r.temperature,
                           r.operating_duration, r.energy_consumption AS energy_kwh
                    FROM electrical_readings r
                    JOIN appliances a ON r.appliance_id = a.appliance_id
                    WHERE a.appliance_name = ?
                    ORDER BY r.timestamp DESC, r.reading_id DESC
                    LIMIT ?;
                    """
                    df = pd.read_sql_query(query, conn, params=(appliance_name, limit))
                else:
                    query = """
                    SELECT r.reading_id AS id, a.appliance_name, r.timestamp,
                           r.voltage, r.current, r.active_power, r.reactive_power,
                           r.apparent_power, r.power_factor, r.frequency, r.temperature,
                           r.operating_duration, r.energy_consumption AS energy_kwh
                    FROM electrical_readings r
                    JOIN appliances a ON r.appliance_id = a.appliance_id
                    ORDER BY r.timestamp DESC, r.reading_id DESC
                    LIMIT ?;
                    """
                    df = pd.read_sql_query(query, conn, params=(limit,))
                return df

    def insert_reading(self, appliance_id: int, timestamp: str, voltage: float,
                       current: float, active_power: float, reactive_power: float,
                       apparent_power: float, power_factor: float, frequency: float,
                       temperature: float, operating_duration: float, energy_kwh: float,
                       is_simulated: bool = False) -> int:
        """Inserts a new electrical telemetry reading record."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO electrical_readings (
                    appliance_id, timestamp, voltage, current, active_power,
                    reactive_power, apparent_power, power_factor, frequency,
                    temperature, operating_duration, energy_consumption, is_simulated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    appliance_id, timestamp, float(voltage), float(current),
                    float(active_power), float(reactive_power), float(apparent_power),
                    float(power_factor), float(frequency), float(temperature),
                    float(operating_duration), float(energy_kwh), int(is_simulated)
                ))
                conn.commit()
                return cur.lastrowid

    def insert_prediction(self, reading_id: Optional[int], appliance_name: str,
                          timestamp: str, load_profile: str, is_anomaly: bool,
                          anomaly_score: float, predicted_fault: str, confidence: float,
                          abnormal_parameters: str, corrective_action: str) -> int:
        """Inserts an ML prediction diagnostic entry."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT appliance_id FROM appliances WHERE appliance_name = ?;", (appliance_name,))
                row = cur.fetchone()
                app_id = row[0] if row else 1

                cur.execute("""
                INSERT INTO ml_predictions (
                    appliance_id, reading_id, timestamp, predicted_load_category,
                    predicted_fault, anomaly_status, anomaly_score, confidence_score,
                    abnormal_parameters, corrective_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    app_id, reading_id, timestamp, load_profile,
                    predicted_fault, "Anomaly" if is_anomaly else "Normal",
                    float(anomaly_score), float(confidence),
                    abnormal_parameters, corrective_action
                ))
                conn.commit()
                return cur.lastrowid

    def insert_alert(self, appliance_name: str, timestamp: str, severity: str,
                     fault_type: str, trigger_parameter: str, trigger_value: float,
                     threshold_value: float, suggested_action: str,
                     description: Optional[str] = None) -> int:
        """Inserts a new safety alert into the alert queue."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT appliance_id FROM appliances WHERE appliance_name = ?;", (appliance_name,))
                row = cur.fetchone()
                app_id = row[0] if row else 1

                if description is None:
                    description = f"{fault_type} identified on {appliance_name}. {trigger_parameter} reached {trigger_value:.2f} (Limit: {threshold_value:.2f})."

                cur.execute("""
                INSERT INTO alerts (
                    appliance_id, timestamp, alert_type, severity, description,
                    detected_parameter, parameter_value, threshold_value,
                    recommended_action, acknowledgement_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Active');
                """, (
                    app_id, timestamp, fault_type, severity, description,
                    trigger_parameter, float(trigger_value), float(threshold_value),
                    suggested_action
                ))
                conn.commit()
                return cur.lastrowid

    def acknowledge_alert(self, alert_id: int):
        """Acknowledges an active alert."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                UPDATE alerts SET acknowledgement_status = 'Acknowledged'
                WHERE alert_id = ?;
                """, (alert_id,))
                conn.commit()

    def resolve_alert(self, alert_id: int):
        """Marks an alert as resolved with timestamp."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                UPDATE alerts
                SET acknowledgement_status = 'Resolved', resolved_at = ?
                WHERE alert_id = ?;
                """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_id))
                conn.commit()
