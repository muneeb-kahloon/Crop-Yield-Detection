import os
import json

AGRONOMIC_DB_PATH = os.path.join("database", "crop_agronomic_db.json")

def load_crop_database():
    with open(AGRONOMIC_DB_PATH, "r") as f:
        return json.load(f)

def rank_crop_suitability(soil_ph, soil_type, water_availability):
    crop_db = load_crop_database()
    ranked_results = []
    
    for crop, data in crop_db.items():
        score = 100
        reasons = []
        warnings = []
        
        ph_min = data["ph_min"]
        ph_max = data["ph_max"]
        optimal_ph = data["optimal_ph"]
        
        if ph_min <= soil_ph <= ph_max:
            ph_diff = abs(soil_ph - optimal_ph)
            score -= (ph_diff * 10)
            reasons.append(f"Soil pH {soil_ph:.1f} fits target range ({ph_min} - {ph_max}).")
        else:
            score -= 35
            warnings.append(f"Soil pH {soil_ph:.1f} is outside preferred range ({ph_min} - {ph_max}).")

        suitable_soils = data.get("soil_types", [])
        if soil_type in suitable_soils:
            reasons.append(f"Soil texture '{soil_type}' is optimal.")
        elif soil_type == "Sandy" and "Sandy Loam" in suitable_soils:
            score -= 10
            reasons.append("Sandy soil is acceptable with organic additions.")
        else:
            score -= 20
            warnings.append(f"Soil texture '{soil_type}' is sub-optimal for root aeration.")

        crop_water = data.get("water_need", "Medium")
        if water_availability == "High":
            reasons.append("Water supply is sufficient.")
        elif water_availability == "Medium" and crop_water in ["Medium", "Low to Medium", "Low"]:
            reasons.append("Moderate water supply matches crop requirements.")
        elif water_availability == "Low" and crop_water in ["Low", "Low to Medium"]:
            reasons.append("Crop is relatively drought resilient.")
        else:
            score -= 25
            warnings.append(f"Crop water demand is {crop_water}, but available supply is {water_availability}.")

        final_score = max(10, min(100, round(score)))
        status = "Highly Suitable" if final_score >= 80 else ("Moderately Suitable" if final_score >= 55 else "Low Suitability")
        
        ranked_results.append({
            "crop": crop,
            "score": final_score,
            "status": status,
            "description": data.get("description", ""),
            "reasons": reasons,
            "warnings": warnings,
            "optimal_ph": f"{ph_min} - {ph_max}"
        })
        
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results