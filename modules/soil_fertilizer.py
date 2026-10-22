import os
import json

AGRONOMIC_DB_PATH = os.path.join("database", "crop_agronomic_db.json")
FERTILIZER_DB_PATH = os.path.join("database", "fertilizer_db.json")

def load_databases():
    with open(AGRONOMIC_DB_PATH, "r") as f:
        crop_db = json.load(f)
    with open(FERTILIZER_DB_PATH, "r") as f:
        fertilizer_db = json.load(f)
    return crop_db, fertilizer_db

def evaluate_soil_and_fertilizer(crop_name, soil_ph, soil_type=None, n_val=None, p_val=None, k_val=None, growth_stage="Vegetative"):
    crop_db, fertilizer_db = load_databases()
    
    crop_data = crop_db.get(crop_name)
    if not crop_data:
        return {
            "error": f"Crop '{crop_name}' is not currently in the agronomic database.",
            "supported_crops": list(crop_db.keys())
        }
        
    results = {
        "crop": crop_name,
        "soil_ph": soil_ph,
        "ph_status": "Optimal",
        "ph_analysis": "",
        "ph_remedy": None,
        "deficiencies": [],
        "fertilizer_recommendations": [],
        "general_management": []
    }
    
    ph_min = crop_data["ph_min"]
    ph_max = crop_data["ph_max"]
    
    if soil_ph < ph_min:
        results["ph_status"] = "Too Acidic"
        results["ph_analysis"] = f"Soil pH of {soil_ph:.1f} is below the suitable range ({ph_min} - {ph_max}) for {crop_name}. Acidic soils reduce phosphorus and magnesium availability."
        results["ph_remedy"] = {
            "amendment": "Agricultural Lime (Calcium Carbonate)",
            "guidance": fertilizer_db["Agricultural Lime (Calcium Carbonate)"]["application_tips"]
        }
    elif soil_ph > ph_max:
        results["ph_status"] = "Too Alkaline / Sodic"
        results["ph_analysis"] = f"Soil pH of {soil_ph:.1f} is above the suitable range ({ph_min} - {ph_max}) for {crop_name}. High pH reduces iron, zinc, and manganese uptake."
        results["ph_remedy"] = {
            "amendment": "Agricultural Gypsum or Elemental Sulfur",
            "guidance": fertilizer_db["Agricultural Gypsum (Calcium Sulfate)"]["application_tips"]
        }
    else:
        results["ph_status"] = "Optimal"
        results["ph_analysis"] = f"Soil pH of {soil_ph:.1f} is within the ideal target range ({ph_min} - {ph_max}) for {crop_name}. Nutrient availability is maximized."

    recs = []
    
    if n_val is not None and n_val < 140:
        results["deficiencies"].append("Nitrogen (N) Deficient")
        recs.append({
            "fertilizer": "Urea (46-0-0)",
            "reason": f"Nitrogen level ({n_val} mg/kg) is low. {crop_name} requires strong vegetative growth support.",
            "application": fertilizer_db["Urea"]["application_tips"]
        })

    if p_val is not None and p_val < 15:
        results["deficiencies"].append("Phosphorus (P) Deficient")
        recs.append({
            "fertilizer": "DAP (18-46-0)",
            "reason": f"Phosphorus level ({p_val} mg/kg) is low. Essential for root establishment.",
            "application": fertilizer_db["DAP (Di-Ammonium Phosphate)"]["application_tips"]
        })
            
    if k_val is not None and k_val < 120:
        results["deficiencies"].append("Potassium (K) Deficient")
        fert = "SOP (0-0-50)" if crop_name in ["Potato", "Tomato"] else "MOP (0-0-60)"
        recs.append({
            "fertilizer": fert,
            "reason": f"Potassium level ({k_val} mg/kg) is low. Critical for water regulation and fruit/tuber development.",
            "application": fertilizer_db.get(fert, {}).get("application_tips", "")
        })

    recs.append({
        "fertilizer": "Compost / Well-Rotted Manure",
        "reason": "Enhances soil microbial activity and improves soil moisture retention.",
        "application": fertilizer_db["Compost / Well-Rotted Manure"]["application_tips"]
    })

    results["fertilizer_recommendations"] = recs

    if soil_type:
        if soil_type in crop_data.get("soil_types", []):
            results["general_management"].append(f"Soil texture '{soil_type}' is well-suited for {crop_name}.")
        elif soil_type == "Sandy":
            results["general_management"].append("Sandy soil loses nutrients quickly via leaching. Split fertilizer into smaller doses.")

    return results