"""
RoadSense — Spatial Clustering & Multi-Ride Deduplication Engine
================================================================
Merges GPS detections across multiple vehicle rides to:
1. Eliminate duplicate pins caused by GPS drift (within an eps radius).
2. Establish ground-truth confidence based on multi-pass consensus.
3. Compute canonical centroids, peak severity, and hazard type.
4. Export industry-standard GeoJSON for GIS / municipal dashboards and Leaflet.
"""

import os
import sys
import json
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute vector or scalar Haversine distance in meters between two GPS coordinates.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * EARTH_RADIUS_METERS * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def load_anomalies_from_rides(ride_paths):
    """
    Load all hazard detections (Label == 1) from a list of fused CSVs or directories.
    """
    records = []
    
    for path in ride_paths:
        if os.path.isdir(path):
            csv_matches = glob.glob(os.path.join(path, '**', 'fused_labeled.csv'), recursive=True)
            if not csv_matches:
                csv_matches = glob.glob(os.path.join(path, '*.csv'))
        else:
            csv_matches = [path]
            
        for csv_file in csv_matches:
            try:
                df = pd.read_csv(csv_file)
                # Check for label column
                lbl_col = None
                for c in ['Label', 'label', 'LABEL']:
                    if c in df.columns:
                        lbl_col = c
                        break
                        
                if not lbl_col or 'latitude' not in df.columns or 'longitude' not in df.columns:
                    continue
                    
                hazards = df[df[lbl_col] == 1].copy()
                if hazards.empty:
                    continue
                    
                ride_name = os.path.basename(os.path.dirname(os.path.abspath(csv_file)))
                if not ride_name or ride_name == '.':
                    ride_name = os.path.basename(csv_file)
                    
                for _, row in hazards.iterrows():
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    if abs(lat) < 0.001 and abs(lon) < 0.001:
                        continue # Skip unanchored 0,0 points
                        
                    records.append({
                        'ride_id': ride_name,
                        'timestamp': row.get('timestamp', 0),
                        'latitude': lat,
                        'longitude': lon,
                        'speed': float(row.get('speed', 0.0)),
                        'z_score': float(row.get('z_score', 0.0)),
                        'severity': str(row.get('Severity', 'Moderate')),
                        'hazard_type': str(row.get('Hazard_Type', 'Pothole')),
                        'source': str(row.get('Source', 'Fused')),
                        'est_area_cm2': float(row.get('est_area_cm2', 0.0)),
                        'est_depth_cm': float(row.get('est_depth_cm', 0.0)),
                        'asphalt_volume_liters': float(row.get('asphalt_volume_liters', 0.0)),
                        'asphalt_mass_kg': float(row.get('asphalt_mass_kg', 0.0)),
                        'repair_cost_inr': int(row.get('repair_cost_inr', 0)),
                        'repair_cost_usd': float(row.get('repair_cost_usd', 0.0))
                    })
            except Exception as e:
                print(f"  Warning: Could not parse {csv_file}: {e}")
                
    return pd.DataFrame(records)


def cluster_hazards(anomalies_df, eps_meters=8.0):
    """
    Perform Haversine DBSCAN spatial clustering on anomalies.
    Returns clustered DataFrame and GeoJSON summary.
    """
    if anomalies_df.empty or len(anomalies_df) == 0:
        print("No valid anomalies found to cluster.")
        return pd.DataFrame(), {"type": "FeatureCollection", "features": []}

    print(f"Clustering {len(anomalies_df)} raw detections across {anomalies_df['ride_id'].nunique()} ride(s)...")

    # Convert coordinates to radians for Haversine metric
    coords_rad = np.radians(anomalies_df[['latitude', 'longitude']].values)
    eps_rad = eps_meters / EARTH_RADIUS_METERS

    db = DBSCAN(eps=eps_rad, min_samples=1, metric='haversine')
    labels = db.fit_predict(coords_rad)
    anomalies_df['cluster_id'] = labels

    clusters = []
    geojson_features = []

    for cluster_id, group in anomalies_df.groupby('cluster_id'):
        centroid_lat = float(group['latitude'].mean())
        centroid_lon = float(group['longitude'].mean())
        hit_count = len(group)
        unique_rides = int(group['ride_id'].nunique())
        max_z = float(group['z_score'].max())
        avg_z = float(group['z_score'].mean())
        
        # Hazard type consensus
        types = group['hazard_type'].value_counts()
        hazard_type = types.index[0] if len(types) > 0 and types.index[0] != 'None' else 'Pothole'
        
        # Severity consensus
        if (group['severity'] == 'Severe').any() or max_z >= 3.0:
            severity = 'Severe'
        elif (group['severity'] == 'Moderate').any() or max_z >= 2.0:
            severity = 'Moderate'
        else:
            severity = 'Low'

        # Multi-pass ground truth confidence & lifecycle
        if unique_rides >= 2:
            confidence = 'High (Multi-Ride Consensus)'
            confidence_level = 3
            lifecycle_stage = 'Corroborated Defect'
        elif hit_count >= 2 or severity == 'Severe':
            confidence = 'Medium (Strong Single-Ride Impact)'
            confidence_level = 2
            lifecycle_stage = 'Active Defect'
        else:
            confidence = 'Low (Single Pass Anomaly)'
            confidence_level = 1
            lifecycle_stage = 'Monitoring'

        # Municipal Action & Repair Priority
        if severity == 'Severe':
            priority = 'P1 (Urgent Dispatch)'
            repair_status = 'Pending Emergency Repair'
            action = 'Emergency Cold-mix Patching & Compaction' if hazard_type == 'Pothole' else 'Reflective Thermoplastic Warning Stripes'
        elif severity == 'Moderate':
            priority = 'P2 (Scheduled Maintenance)'
            repair_status = 'Queued for Maintenance'
            action = 'Skin Patching & Bitumen Sealing' if hazard_type == 'Pothole' else 'Install Speed Warning Sign'
        else:
            priority = 'P3 (Routine Inspection)'
            repair_status = 'Under Observation'
            action = 'Routine Road Surface Monitoring'

        # Dimensional modeling and asphalt/cost calculations
        area_vals = group['est_area_cm2'].values if 'est_area_cm2' in group.columns else np.zeros(len(group))
        depth_vals = group['est_depth_cm'].values if 'est_depth_cm' in group.columns else np.zeros(len(group))
        
        mean_area = float(np.mean(area_vals[area_vals > 0])) if np.any(area_vals > 0) else round(max(150.0, max_z * 180.0), 1)
        mean_depth = float(np.mean(depth_vals[depth_vals > 0])) if np.any(depth_vals > 0) else round(min(12.0, max(1.5, max_z * 1.65)), 1)
        vol_liters = round(0.5 * mean_area * mean_depth * 0.001, 2)
        mass_kg = round(vol_liters * 2.35, 2)
        cost_inr = int(max(350, round(mass_kg * 45.0 + 300.0)))
        cost_usd = round(max(15.0, mass_kg * 0.65 + 10.0), 2)

        cluster_summary = {
            'cluster_id': int(cluster_id),
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'hazard_type': hazard_type,
            'severity': severity,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'lifecycle_stage': lifecycle_stage,
            'priority': priority,
            'repair_status': repair_status,
            'action_recommendation': action,
            'est_area_cm2': mean_area,
            'est_depth_cm': mean_depth,
            'asphalt_volume_liters': vol_liters,
            'asphalt_mass_kg': mass_kg,
            'repair_cost_inr': cost_inr,
            'repair_cost_usd': cost_usd,
            'total_hits': hit_count,
            'unique_rides': unique_rides,
            'max_z_score': round(max_z, 2),
            'avg_z_score': round(avg_z, 2),
            'contributing_rides': list(group['ride_id'].unique())
        }
        clusters.append(cluster_summary)

        # RFC 7946 GeoJSON Feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [centroid_lon, centroid_lat]
            },
            "properties": cluster_summary
        }
        geojson_features.append(feature)

    clusters_df = pd.DataFrame(clusters)
    geojson_doc = {
        "type": "FeatureCollection",
        "metadata": {
            "generated_by": "RoadSense Spatial Clustering Engine",
            "total_raw_detections": len(anomalies_df),
            "canonical_hazards": len(clusters),
            "eps_meters": eps_meters
        },
        "features": geojson_features
    }

    print(f"Successfully consolidated into {len(clusters)} canonical hazard entity(ies).")
    return clusters_df, geojson_doc


def run_spatial_aggregation(input_paths, output_dir=None, eps_meters=8.0):
    """
    End-to-end multi-ride ingestion, deduplication, and export.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'aggregated_hazards')
        
    os.makedirs(output_dir, exist_ok=True)
    
    anomalies_df = load_anomalies_from_rides(input_paths)
    if anomalies_df.empty:
        print("No hazard rows found in provided inputs.")
        return None, None
        
    clusters_df, geojson_doc = cluster_hazards(anomalies_df, eps_meters=eps_meters)
    
    # Save outputs
    csv_out = os.path.join(output_dir, 'canonical_hazards.csv')
    json_out = os.path.join(output_dir, 'canonical_hazards.geojson')
    
    clusters_df.to_csv(csv_out, index=False)
    with open(json_out, 'w') as f:
        json.dump(geojson_doc, f, indent=2)
        
    print(f"\n[OUTPUT] Canonical Hazard CSV saved: {csv_out}")
    print(f"[OUTPUT] Canonical GeoJSON saved:    {json_out}")
    
    return clusters_df, geojson_doc


def verify_repairs_from_ride(ride_data, geojson_path=None, radius_m=12.0, ride_id=None):
    """
    Automated Pothole Repair Verification:
    Tracks defect trajectories over time. When a subsequent vehicle ride passes over a
    known canonical hazard record:
    - If the physical vibration is smooth (Z_score < 1.0 or dynamic shock is absent),
      the defect is automatically marked as 'Repaired & Verified' and 'Repaired & Closed'.
    - If the vibration remains elevated (Z_score >= 2.5), the defect lifecycle is flagged
      as 'Deteriorating Defect' and its deterioration counter is incremented.
    
    Parameters:
        ride_data: pandas DataFrame or path to telemetry CSV with ['latitude', 'longitude', 'z_score'].
        geojson_path: path to canonical_hazards.geojson (defaults to main/aggregated_hazards/canonical_hazards.geojson).
        radius_m: spatial tolerance distance in meters (default: 12.0m).
        ride_id: string identifier for the verification ride pass.
        
    Returns:
        dict: Summary of traversed hazards, verified repairs, and deteriorating flags.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if geojson_path is None:
        geojson_path = os.path.join(base_dir, 'aggregated_hazards', 'canonical_hazards.geojson')
        
    if not os.path.exists(geojson_path):
        print(f"[AUTO-VERIFY] Canonical GeoJSON not found at {geojson_path}. Skipping verification.")
        return {"success": False, "error": "Canonical GeoJSON not found"}
        
    # Ingest ride data
    if isinstance(ride_data, str):
        if not os.path.exists(ride_data):
            return {"success": False, "error": f"Ride file not found: {ride_data}"}
        ride_df = pd.read_csv(ride_data)
        if not ride_id:
            ride_id = os.path.basename(os.path.dirname(os.path.abspath(ride_data)))
            if not ride_id or ride_id == '.':
                ride_id = os.path.basename(ride_data)
    elif isinstance(ride_data, pd.DataFrame):
        ride_df = ride_data.copy()
        if not ride_id:
            ride_id = f"ride_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        return {"success": False, "error": "Invalid ride_data format. Expected DataFrame or CSV path."}
        
    # Ensure necessary columns
    lat_col = next((c for c in ['latitude', 'lat', 'LATITUDE'] if c in ride_df.columns), None)
    lon_col = next((c for c in ['longitude', 'lon', 'LONGITUDE'] if c in ride_df.columns), None)
    if not lat_col or not lon_col:
        return {"success": False, "error": "GPS coordinates (latitude, longitude) missing from ride data."}
        
    # Filter valid GPS points
    valid_mask = (ride_df[lat_col].abs() > 0.001) & (ride_df[lon_col].abs() > 0.001)
    sub_df = ride_df[valid_mask].copy()
    if sub_df.empty:
        return {"success": True, "message": "No valid GPS trajectory points found in ride.", "repairs_verified_count": 0}

    # Determine vibration column (z_score, or compute dynamic accel proxy)
    z_col = None
    for candidate in ['z_score', 'zscore', 'Z_Score', 'accel_dyn', 'accel_mag']:
        if candidate in sub_df.columns:
            z_col = candidate
            break

    ride_lats = sub_df[lat_col].values
    ride_lons = sub_df[lon_col].values
    if z_col:
        ride_z = sub_df[z_col].values
    else:
        ride_z = np.zeros(len(sub_df))

    with open(geojson_path, 'r') as f:
        geojson_doc = json.load(f)

    features = geojson_doc.get("features", [])
    if not features:
        return {"success": True, "message": "No canonical hazards in database.", "repairs_verified_count": 0}

    verified_repairs = []
    deteriorating_hazards = []
    traversed_clusters = []
    updated = False

    for feat in features:
        props = feat.get("properties", {})
        cluster_id = props.get("cluster_id")
        c_lat = props.get("centroid_lat")
        c_lon = props.get("centroid_lon")
        
        if c_lat is None or c_lon is None:
            continue

        # Compute Haversine distance from all trajectory points
        dists = haversine_distance(c_lat, c_lon, ride_lats, ride_lons)
        within_mask = dists <= radius_m
        
        if np.any(within_mask):
            traversed_clusters.append(cluster_id)
            min_dist = float(np.min(dists))
            pass_z_vals = ride_z[within_mask]
            max_pass_z = float(np.max(pass_z_vals)) if len(pass_z_vals) > 0 else 0.0
            
            # 1. Smooth asphalt verification: z_score < 1.0 (minimal vibration)
            if max_pass_z < 1.0:
                prev_status = props.get("repair_status", "Active")
                props["repair_status"] = "Repaired & Verified"
                props["lifecycle_stage"] = "Repaired & Closed"
                props["repaired_at"] = datetime.now().isoformat()
                props["verification_ride"] = ride_id
                props["verification_max_z"] = round(max_pass_z, 2)
                props["verification_dist_m"] = round(min_dist, 2)
                props["smooth_pass_count"] = props.get("smooth_pass_count", 0) + 1
                updated = True
                verified_repairs.append({
                    "cluster_id": cluster_id,
                    "previous_status": prev_status,
                    "new_status": "Repaired & Verified",
                    "max_pass_z": round(max_pass_z, 2),
                    "min_distance_m": round(min_dist, 2),
                    "ride_id": ride_id
                })
            # 2. Defect deterioration: z_score >= 2.5 (strong or worsening physical shock)
            elif max_pass_z >= 2.5:
                props["lifecycle_stage"] = "Deteriorating Defect"
                props["deterioration_count"] = props.get("deterioration_count", 0) + 1
                props["last_revisit_at"] = datetime.now().isoformat()
                props["last_measured_z"] = round(max_pass_z, 2)
                if max_pass_z >= 3.5:
                    props["severity"] = "Severe"
                    props["priority"] = "P1 (Urgent Dispatch)"
                updated = True
                deteriorating_hazards.append({
                    "cluster_id": cluster_id,
                    "lifecycle_stage": "Deteriorating Defect",
                    "max_pass_z": round(max_pass_z, 2),
                    "min_distance_m": round(min_dist, 2),
                    "ride_id": ride_id
                })
            else:
                # Moderate vibration: update last pass timestamp
                props["last_revisit_at"] = datetime.now().isoformat()
                props["last_measured_z"] = round(max_pass_z, 2)

    if updated:
        with open(geojson_path, 'w') as f:
            json.dump(geojson_doc, f, indent=2)

        # Sync canonical CSV
        csv_path = os.path.join(os.path.dirname(geojson_path), 'canonical_hazards.csv')
        try:
            flat_rows = [feat['properties'] for feat in features]
            pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Warning: could not sync canonical CSV: {e}")

        # Sync to survey_app if present
        app_json = os.path.join(base_dir, 'survey_app', 'canonical_hazards.json')
        if os.path.exists(os.path.dirname(app_json)):
            try:
                with open(app_json, 'w') as f:
                    json.dump(geojson_doc, f, indent=2)
            except Exception as e:
                print(f"Warning: could not sync survey_app JSON: {e}")

    summary = {
        "success": True,
        "ride_id": ride_id,
        "total_canonical_hazards": len(features),
        "traversed_hazard_count": len(traversed_clusters),
        "traversed_clusters": traversed_clusters,
        "repairs_verified_count": len(verified_repairs),
        "repairs_verified": verified_repairs,
        "deteriorating_count": len(deteriorating_hazards),
        "deteriorating_hazards": deteriorating_hazards
    }
    print(f"\n[AUTO-VERIFY] Ride '{ride_id}' passed over {len(traversed_clusters)} hazard(s).")
    print(f"              -> Auto-verified Repairs Closed: {len(verified_repairs)}")
    print(f"              -> Deteriorating Hazards Flagged: {len(deteriorating_hazards)}")
    return summary


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_test_dir = os.path.join(base_dir, 'test')
    
    target_inputs = [sys.argv[1]] if len(sys.argv) > 1 else [default_test_dir]
    run_spatial_aggregation(target_inputs)

