import os
from dotenv import load_dotenv

load_dotenv()

def process_assistant_query(user_query):
    query_lower = user_query.lower()
    
    if any(word in query_lower for word in ["spot", "leaf", "leaves", "yellow", "blight", "mold", "fungus", "disease", "rot", "curly"]):
        return {
            "intent": "Crop Disease Inquiry",
            "suggested_tab": "🌿 Crop Disease Diagnosis",
            "response": "It sounds like you are dealing with a visual leaf infection or pest issue.",
            "action_advice": "Head over to the **'Crop Disease Diagnosis'** tab and upload a clear picture of the infected leaf."
        }
        
    elif any(word in query_lower for word in ["ph", "fertilizer", "urea", "dap", "potash", "npk", "nitrogen", "soil", "acidic", "alkaline", "manure"]):
        return {
            "intent": "Soil & Nutrient Management",
            "suggested_tab": "🧪 Soil & Fertilizer Guidance",
            "response": "Your question relates to soil chemistry, pH balancing, or fertilizer requirements.",
            "action_advice": "Navigate to the **'Soil & Fertilizer Guidance'** tab to enter your soil pH and nutrient levels."
        }

    elif any(word in query_lower for word in ["rain", "heat", "drought", "sun", "cold", "frost", "flood", "waterlog", "weather", "temperature"]):
        return {
            "intent": "Weather & Climate Mitigation",
            "suggested_tab": "🌤️ Weather & Climate Advisory",
            "response": "You are describing weather stress or seasonal climate challenges.",
            "action_advice": "Visit the **'Weather & Climate Advisory'** tab to view climate risk recommendations."
        }

    elif any(word in query_lower for word in ["plant", "grow", "suitable", "crop selection", "yield", "season", "what to grow"]):
        return {
            "intent": "Crop Suitability Selection",
            "suggested_tab": "🌾 Crop Suitability Engine",
            "response": "You are asking about choosing the right crop for your land.",
            "action_advice": "Check out the **'Crop Suitability Engine'** tab to input your field conditions."
        }

    else:
        return {
            "intent": "General Agricultural Guidance",
            "suggested_tab": "🏠 Dashboard Overview",
            "response": "I can assist you with Crop Diseases, Soil pH & Fertilizer recommendations, Weather advice, or Crop Selection.",
            "action_advice": "Please select the feature tab that best matches your question or provide more details."
        }