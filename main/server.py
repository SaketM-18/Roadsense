"""
RoadSense — Central API Server & Web Service
============================================
Provides an automated backend to:
1. Ingest mobile sensor CSVs and dashcam videos via REST endpoints.
2. Run asynchronous edge-fusion processing and YOLO verification in the background.
3. Automatically update the citywide canonical hazard database using DBSCAN.
4. Serve the interactive Web Dashboard directly to phones, tablets, and laptops.
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SURVEY_APP_DIR = os.path.join(BASE_DIR, 'survey_app')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads', 'rides')
AGGREGATED_DIR = os.path.join(BASE_DIR, 'aggregated_hazards')
GEOJSON_FILE = os.path.join(AGGREGATED_DIR, 'canonical_hazards.geojson')
PROFILE_FILE = os.path.join(BASE_DIR, 'vehicle_profile.json')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(AGGREGATED_DIR, exist_ok=True)

# Import pipeline functions
sys.path.append(BASE_DIR)
from process_ride import process_ride_folder, compute_pavement_condition_index, compute_ride_corridors
from calibrate import train_profile
from spatial_cluster import run_spatial_aggregation, verify_repairs_from_ride
from generate_inspection_report import generate_inspection_pdf

# App setup
app = FastAPI(
    title="RoadSense API",
    description="Automated AI Infrastructure Intelligence & Sensor Fusion Service",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory ride registry
RIDE_STATUS = {}


def background_process_ride(ride_id: str, ride_folder: str, do_stabilize: bool = False):
    """Background worker to process ride video + sensor CSV."""
    try:
        RIDE_STATUS[ride_id]["status"] = "processing"
        result = process_ride_folder(
            ride_folder,
            do_stabilize=do_stabilize,
            force_stabilize=False,
            do_retrain=False
        )
        RIDE_STATUS[ride_id].update({
            "status": "completed",
            "summary": result,
            "completed_at": datetime.now().isoformat()
        })
        
        # Save ride metadata to folder
        meta_path = os.path.join(ride_folder, "ride_meta.json")
        with open(meta_path, "w") as f:
            json.dump(RIDE_STATUS[ride_id], f, indent=2)
            
    except Exception as e:
        print(f"Error processing ride {ride_id}: {e}")
        RIDE_STATUS[ride_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


# ============ API ENDPOINTS ============

@app.get("/api/status")
def get_system_status():
    """System health check and profile baseline status."""
    has_profile = os.path.exists(PROFILE_FILE)
    has_hazards = os.path.exists(GEOJSON_FILE)
    return {
        "service": "RoadSense API",
        "version": "2.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vehicle_profile_loaded": has_profile,
        "canonical_hazard_map_active": has_hazards
    }


@app.post("/api/rides/upload")
async def upload_ride(
    background_tasks: BackgroundTasks,
    telemetry: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
    ride_name: Optional[str] = Form(None),
    stabilize: bool = Form(False)
):
    """
    Accepts telemetry CSV and optional video file from mobile device.
    Launches processing asynchronously in the background.
    """
    ride_id = f"ride_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ride_folder = os.path.join(UPLOADS_DIR, ride_id)
    os.makedirs(ride_folder, exist_ok=True)

    # Save telemetry CSV
    csv_path = os.path.join(ride_folder, telemetry.filename or "telemetry.csv")
    with open(csv_path, "wb") as buffer:
        shutil.copyfileobj(telemetry.file, buffer)

    # Save video if provided
    video_path = None
    if video and video.filename:
        video_path = os.path.join(ride_folder, video.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

    # Register ride
    RIDE_STATUS[ride_id] = {
        "ride_id": ride_id,
        "name": ride_name or f"Survey {ride_id}",
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "has_video": video_path is not None,
        "folder": ride_folder
    }

    # Dispatch processing in background
    background_tasks.add_task(background_process_ride, ride_id, ride_folder, do_stabilize=stabilize)

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "ride_id": ride_id,
            "status": "queued",
            "message": "Ride received successfully. Processing running asynchronously."
        }
    )


@app.get("/api/rides")
def list_rides():
    """List all available rides (both uploaded and test runs)."""
    rides = []

    # 1. From uploads directory
    if os.path.exists(UPLOADS_DIR):
        for item in sorted(os.listdir(UPLOADS_DIR), reverse=True):
            folder = os.path.join(UPLOADS_DIR, item)
            if not os.path.isdir(folder):
                continue
            meta_file = os.path.join(folder, "ride_meta.json")
            fused_file = os.path.join(folder, "fused_labeled.csv")
            
            ride_info = {
                "ride_id": item,
                "name": item,
                "folder": folder,
                "status": "completed" if os.path.exists(fused_file) else "queued"
            }
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r") as f:
                        ride_info.update(json.load(f))
                except:
                    pass
            elif os.path.exists(fused_file):
                try:
                    df = pd.read_csv(fused_file)
                    ride_info["summary"] = {
                        "total_rows": len(df),
                        "pothole_count": int((df['Label'] == 1).sum()),
                        "severe_count": int((df.get('Severity') == 'Severe').sum())
                    }
                except:
                    pass
            rides.append(ride_info)

    # 2. From historical test runs
    test_dir = os.path.join(BASE_DIR, 'test')
    if os.path.exists(test_dir):
        for item in os.listdir(test_dir):
            folder = os.path.join(test_dir, item)
            fused_file = os.path.join(folder, "fused_labeled.csv")
            if os.path.isdir(folder) and os.path.exists(fused_file):
                try:
                    df = pd.read_csv(fused_file)
                    potholes = int((df['Label'] == 1).sum())
                    rides.append({
                        "ride_id": f"test_{item}",
                        "name": f"Test Run: {item}",
                        "status": "completed",
                        "folder": folder,
                        "summary": {
                            "total_rows": len(df),
                            "pothole_count": potholes
                        }
                    })
                except:
                    pass

    return {"count": len(rides), "rides": rides}


@app.get("/api/rides/{ride_id}")
def get_ride_detail(ride_id: str):
    """Retrieve details, GPS route, and labeled points for a specific ride."""
    # Find folder
    folder = None
    if ride_id.startswith("test_"):
        folder = os.path.join(BASE_DIR, 'test', ride_id.replace("test_", ""))
    else:
        folder = os.path.join(UPLOADS_DIR, ride_id)

    if not folder or not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Ride not found")

    fused_csv = os.path.join(folder, "fused_labeled.csv")
    if not os.path.exists(fused_csv):
        # Look for any CSV
        csvs = [f for f in os.listdir(folder) if f.endswith('.csv')]
        if csvs:
            fused_csv = os.path.join(folder, csvs[0])
        else:
            raise HTTPException(status_code=404, detail="No telemetry data found for ride")

    try:
        df = pd.read_csv(fused_csv)
        rows = df.to_dict(orient="records")
        return {
            "ride_id": ride_id,
            "total_points": len(rows),
            "rows": rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rides/{ride_id}/download")
def download_ride_csv(ride_id: str):
    """Download the labeled CSV for a specific ride."""
    folder = os.path.join(BASE_DIR, 'test', ride_id.replace("test_", "")) if ride_id.startswith("test_") else os.path.join(UPLOADS_DIR, ride_id)
    fused_csv = os.path.join(folder, "fused_labeled.csv")
    if os.path.exists(fused_csv):
        return FileResponse(fused_csv, media_type="text/csv", filename=f"{ride_id}_fused.csv")
    raise HTTPException(status_code=404, detail="CSV file not found")


@app.get("/api/rides/{ride_id}/report.pdf")
def get_ride_pdf_report(ride_id: str):
    """Generate and return an official ASTM D6433 Municipal PDF Inspection Report."""
    folder = os.path.join(BASE_DIR, 'test', ride_id.replace("test_", "")) if ride_id.startswith("test_") else os.path.join(UPLOADS_DIR, ride_id)
    fused_csv = os.path.join(folder, "fused_labeled.csv")
    if not os.path.exists(fused_csv):
        raise HTTPException(status_code=404, detail="Fused telemetry data not found for ride")
        
    pdf_path = os.path.join(folder, "RoadSense_Inspection_WorkOrder.pdf")
    try:
        generate_inspection_pdf(fused_csv, pdf_path, title=f"Audit {ride_id}")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"RoadSense_Audit_{ride_id}.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


@app.get("/api/rides/{ride_id}/evidence/{filename}")
def get_ride_evidence_image(ride_id: str, filename: str):
    """Serve a visual evidence camera snapshot."""
    folder = os.path.join(BASE_DIR, 'test', ride_id.replace("test_", "")) if ride_id.startswith("test_") else os.path.join(UPLOADS_DIR, ride_id)
    img_path = os.path.join(folder, "evidence", filename)
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Evidence photo not found")


@app.get("/api/hazards/geojson")
def get_canonical_hazards():
    """Returns the citywide canonical clustered hazards GeoJSON."""
    if os.path.exists(GEOJSON_FILE):
        with open(GEOJSON_FILE, "r") as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}


@app.get("/api/rides/{ride_id}/corridors")
def get_ride_corridors(ride_id: str, segment_m: float = 25.0):
    """
    Computes continuous 25-meter micro-segment road quality corridors with
    ASTM D6433 PCI ratings and color hex codes for spatial heatmap visualization.
    """
    folder = os.path.join(BASE_DIR, 'test', ride_id.replace("test_", "")) if ride_id.startswith("test_") else os.path.join(UPLOADS_DIR, ride_id)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Ride folder not found")
        
    fused_csv = os.path.join(folder, "fused_labeled.csv")
    if not os.path.exists(fused_csv):
        csvs = [f for f in os.listdir(folder) if f.endswith('.csv')]
        if csvs:
            fused_csv = os.path.join(folder, csvs[0])
        else:
            raise HTTPException(status_code=404, detail="No telemetry CSV found")
            
    df = pd.read_csv(fused_csv)
    corridors = compute_ride_corridors(df, segment_length_m=segment_m)
    return {
        "ride_id": ride_id,
        "segment_count": len(corridors),
        "corridors": corridors
    }


@app.get("/api/hazards/summary")
def get_hazards_summary():
    """Returns lifecycle metrics, asphalt material tonnage, and repair budget across canonical road hazards."""
    if not os.path.exists(GEOJSON_FILE):
        return {
            "total_hazards": 0,
            "active_hazards": 0,
            "repaired_hazards": 0,
            "deteriorating_hazards": 0,
            "severe_count": 0,
            "moderate_count": 0,
            "repair_rate_pct": 0.0,
            "total_asphalt_kg_needed": 0.0,
            "total_budget_inr": 0,
            "total_budget_usd": 0.0
        }
        
    with open(GEOJSON_FILE, "r") as f:
        data = json.load(f)
        
    features = data.get("features", [])
    total = len(features)
    repaired = sum(1 for f in features if f["properties"].get("repair_status") == "Repaired & Verified")
    deteriorating = sum(1 for f in features if f["properties"].get("lifecycle_stage") == "Deteriorating Defect")
    active = total - repaired
    severe = sum(1 for f in features if f["properties"].get("severity") == "Severe")
    moderate = sum(1 for f in features if f["properties"].get("severity") == "Moderate")
    
    # Material and financial budgeting calculations
    active_props = [f["properties"] for f in features if f["properties"].get("repair_status") != "Repaired & Verified"]
    total_asphalt_kg = round(sum(p.get("asphalt_mass_kg", 0.0) for p in active_props), 1)
    total_budget_inr = int(sum(p.get("repair_cost_inr", 0) for p in active_props))
    total_budget_usd = round(sum(p.get("repair_cost_usd", 0.0) for p in active_props), 2)
    
    return {
        "total_hazards": total,
        "active_hazards": active,
        "repaired_hazards": repaired,
        "deteriorating_hazards": deteriorating,
        "severe_count": severe,
        "moderate_count": moderate,
        "repair_rate_pct": round((repaired / total * 100) if total > 0 else 0, 1),
        "total_asphalt_kg_needed": total_asphalt_kg,
        "total_budget_inr": total_budget_inr,
        "total_budget_usd": total_budget_usd
    }


@app.post("/api/hazards/auto-verify")
def api_auto_verify(
    ride_id: Optional[str] = Form(None),
    radius_m: float = Form(12.0)
):
    """
    Automated Pothole Repair Verification endpoint:
    Evaluates physical sensor vibration from a survey ride against known canonical hazards.
    Marks smooth traversals as 'Repaired & Verified' and persistent shocks as 'Deteriorating Defect'.
    """
    if not ride_id:
        rides_resp = list_rides()
        rides_list = rides_resp.get("rides", [])
        if not rides_list:
            raise HTTPException(status_code=400, detail="No recorded rides available for auto-verification")
        ride_id = rides_list[0]["ride_id"]

    # Locate ride folder
    if ride_id.startswith("test_"):
        folder = os.path.join(BASE_DIR, 'test', ride_id.replace("test_", ""))
    else:
        folder = os.path.join(UPLOADS_DIR, ride_id)

    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"Ride folder not found: {ride_id}")

    fused_csv = os.path.join(folder, "fused_labeled.csv")
    if not os.path.exists(fused_csv):
        csvs = [f for f in os.listdir(folder) if f.endswith('.csv')]
        if csvs:
            fused_csv = os.path.join(folder, csvs[0])
        else:
            raise HTTPException(status_code=404, detail="No telemetry CSV found for ride")

    result = verify_repairs_from_ride(fused_csv, geojson_path=GEOJSON_FILE, radius_m=radius_m, ride_id=ride_id)
    return result


@app.post("/api/hazards/{cluster_id}/status")
def update_hazard_status(cluster_id: int, status: str = Form(...)):
    """Update repair status of a canonical hazard (e.g. 'Repaired & Verified', 'Under Repair')."""
    if not os.path.exists(GEOJSON_FILE):
        raise HTTPException(status_code=404, detail="Canonical hazards not found")
        
    with open(GEOJSON_FILE, "r") as f:
        data = json.load(f)
        
    found = False
    for feat in data.get("features", []):
        if feat["properties"].get("cluster_id") == cluster_id:
            feat["properties"]["repair_status"] = status
            if status == "Repaired & Verified":
                feat["properties"]["lifecycle_stage"] = "Repaired & Closed"
                feat["properties"]["closed_at"] = datetime.now().isoformat()
            found = True
            break
            
    if not found:
        raise HTTPException(status_code=404, detail=f"Hazard cluster {cluster_id} not found")
        
    with open(GEOJSON_FILE, "w") as f:
        json.dump(data, f, indent=2)
        
    # Sync with survey_app
    frontend_json = os.path.join(SURVEY_APP_DIR, "canonical_hazards.json")
    if os.path.exists(SURVEY_APP_DIR):
        with open(frontend_json, "w") as f:
            json.dump(data, f, indent=2)
            
    return {"success": True, "cluster_id": cluster_id, "new_status": status}


@app.get("/api/hazards/export/geojson")
def export_canonical_geojson():
    """Download canonical hazards as standard RFC 7946 GeoJSON for QGIS/ArcGIS."""
    if os.path.exists(GEOJSON_FILE):
        return FileResponse(
            GEOJSON_FILE,
            media_type="application/geo+json",
            filename="roadsense_canonical_hazards_qgis.geojson"
        )
    raise HTTPException(status_code=404, detail="Canonical GeoJSON not found")


@app.get("/api/hazards/export/contractor-dispatch.csv")
def export_contractor_dispatch():
    """Download municipal contractor repair dispatch work-orders as CSV."""
    csv_file = os.path.join(AGGREGATED_DIR, "canonical_hazards.csv")
    if os.path.exists(csv_file):
        return FileResponse(
            csv_file,
            media_type="text/csv",
            filename="roadsense_contractor_dispatch_workorders.csv"
        )
    raise HTTPException(status_code=404, detail="Canonical hazards CSV not found")


@app.post("/api/calibrate/upload")
async def upload_calibration(file: UploadFile = File(...)):
    """Upload a calibration profile CSV and re-train the vehicle baseline."""
    temp_path = os.path.join(BASE_DIR, f"temp_cal_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        train_profile(temp_path)
        with open(PROFILE_FILE, "r") as f:
            updated_profile = json.load(f)
        os.remove(temp_path)
        return {
            "success": True,
            "message": "Vehicle vibration baseline updated successfully.",
            "profile": updated_profile
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {e}")


@app.get("/api/profile")
def get_vehicle_profile():
    """Returns current vehicle profile baseline."""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    return {}


# Mount Static Frontend Web App
if os.path.exists(SURVEY_APP_DIR):
    app.mount("/", StaticFiles(directory=SURVEY_APP_DIR, html=True), name="static_frontend")


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting RoadSense API server on http://0.0.0.0:{port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)

