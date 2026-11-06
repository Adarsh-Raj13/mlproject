"""
SQLAlchemy Database Configuration, Session Management, and Seed Pipeline.
Establishes connection to the SQLite database, declarative base,
session factory, and initialization/historical seeding utilities.
"""

import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings
from backend.core.logging_config import logger

# SQLite connection string
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes tables and seeds default appliances, user, and historical telemetry."""
    from backend.models.appliance import Appliance
    from backend.models.reading import ElectricalReading
    from backend.models.prediction import MLPrediction
    from backend.models.alert import Alert
    from backend.models.user import User

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        # 1. Seed Appliances
        if db.query(Appliance).count() == 0:
            default_appliances = [
                Appliance(
                    appliance_name="Ceiling Fan",
                    appliance_type="Inductive Motor",
                    rated_voltage=230.0,
                    rated_power=75.0,
                    rated_current=0.40,
                    nominal_pf=0.82,
                    max_safe_temp=60.0,
                    installation_date="2024-01-15",
                    operating_status="Normal",
                    location="Living Room / Lab Area"
                ),
                Appliance(
                    appliance_name="Refrigerator",
                    appliance_type="Cyclic Inductive",
                    rated_voltage=230.0,
                    rated_power=160.0,
                    rated_current=0.95,
                    nominal_pf=0.76,
                    max_safe_temp=65.0,
                    installation_date="2023-11-10",
                    operating_status="Normal",
                    location="Kitchen / Utility"
                ),
                Appliance(
                    appliance_name="Air Conditioner",
                    appliance_type="Heavy Inductive",
                    rated_voltage=230.0,
                    rated_power=1800.0,
                    rated_current=8.50,
                    nominal_pf=0.92,
                    max_safe_temp=75.0,
                    installation_date="2024-03-01",
                    operating_status="Normal",
                    location="Master Suite / Server Room"
                ),
                Appliance(
                    appliance_name="Washing Machine",
                    appliance_type="Dynamic Mixed",
                    rated_voltage=230.0,
                    rated_power=600.0,
                    rated_current=3.20,
                    nominal_pf=0.81,
                    max_safe_temp=65.0,
                    installation_date="2024-02-20",
                    operating_status="Normal",
                    location="Laundry Room"
                ),
                Appliance(
                    appliance_name="Electric Iron",
                    appliance_type="Resistive Heating",
                    rated_voltage=230.0,
                    rated_power=1200.0,
                    rated_current=5.22,
                    nominal_pf=0.99,
                    max_safe_temp=85.0,
                    installation_date="2024-04-05",
                    operating_status="Normal",
                    location="Utility Wardrobe"
                ),
                Appliance(
                    appliance_name="Water Heater",
                    appliance_type="Resistive Immersion",
                    rated_voltage=230.0,
                    rated_power=2000.0,
                    rated_current=8.70,
                    nominal_pf=0.99,
                    max_safe_temp=78.0,
                    installation_date="2023-12-01",
                    operating_status="Normal",
                    location="Bathroom / Boiler Room"
                )
            ]
            db.add_all(default_appliances)
            db.commit()
            logger.info("Seeded 6 default appliances into SQLite database.")

        # 2. Seed Default User
        if db.query(User).count() == 0:
            default_user = User(
                username="admin",
                email="admin@smartload.local",
                role="admin",
                is_active=True
            )
            db.add(default_user)
            db.commit()

        # 3. Seed Historical Readings & Alerts from raw dataset if empty
        if db.query(ElectricalReading).count() == 0:
            raw_path = os.path.join(settings.DATASET_DIR, "raw_data.csv")
            if os.path.exists(raw_path):
                df_raw = pd.read_csv(raw_path)
                app_map = {a.appliance_name: a.appliance_id for a in db.query(Appliance).all()}

                # Sample 50 recent records per appliance
                sampled = df_raw.groupby("Appliance_Type").tail(50).sort_values("Timestamp")
                readings_to_add = []
                alerts_to_add = []

                for _, row in sampled.iterrows():
                    app_name = str(row["Appliance_Type"])
                    app_id = app_map.get(app_name, 1)

                    try:
                        ts = datetime.fromisoformat(str(row["Timestamp"]))
                    except Exception:
                        ts = datetime.utcnow()

                    v = float(row["Voltage"])
                    cur = float(row["Current"])
                    p = float(row["Active_Power"])
                    q = float(row["Reactive_Power"])
                    s = float(row["Apparent_Power"])
                    pf = float(row["Power_Factor"])
                    freq = float(row["Frequency"])
                    temp = float(row["Temperature"])
                    dur = float(row["Operating_Duration"])
                    kwh = float(row["Energy_Consumption"])

                    readings_to_add.append(
                        ElectricalReading(
                            appliance_id=app_id,
                            timestamp=ts,
                            voltage=v,
                            current=cur,
                            active_power=p,
                            reactive_power=q,
                            apparent_power=s,
                            power_factor=pf,
                            frequency=freq,
                            temperature=temp,
                            operating_duration=dur,
                            energy_consumption=kwh,
                            is_simulated=False
                        )
                    )

                    cond = str(row.get("Operating_Condition", "Normal"))
                    if cond != "Normal":
                        sev = "Critical" if cond in ["Overcurrent", "Potential Electrical Fault"] else (
                            "High Risk" if cond in ["Overload", "Overheating", "Voltage Abnormality"] else "Warning"
                        )
                        trigger_p = "Current" if cond in ["Overcurrent", "Overload"] else (
                            "Temperature" if cond == "Overheating" else (
                                "Voltage" if cond == "Voltage Abnormality" else "Active Power"
                            )
                        )
                        val = cur if "Current" in trigger_p else (temp if "Temp" in trigger_p else v)
                        alerts_to_add.append(
                            Alert(
                                appliance_id=app_id,
                                timestamp=ts,
                                alert_type=cond,
                                severity=sev,
                                description=f"{cond} identified on {app_name}. {trigger_p} registered at {val:.2f}.",
                                detected_parameter=trigger_p,
                                parameter_value=val,
                                threshold_value=val * 0.85,
                                recommended_action="Inspect appliance operating parameters and electrical connections.",
                                acknowledgement_status="Active"
                            )
                        )

                db.add_all(readings_to_add)
                db.add_all(alerts_to_add)
                db.commit()
                logger.info(f"Seeded {len(readings_to_add)} historical readings and {len(alerts_to_add)} alerts.")
