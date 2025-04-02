import json
from pathlib import Path

def test_activities_json():
    """Test that activities.json is properly formatted and accessible."""
    try:
        # Get the path to activities.json
        activities_path = Path("data/activities.json")
        
        # Check if file exists
        if not activities_path.exists():
            print(f"❌ Error: activities.json not found at {activities_path}")
            return False
        
        # Try to load the JSON
        with open(activities_path) as f:
            activities = json.load(f)
        
        # Verify it's a dictionary
        if not isinstance(activities, dict):
            print(f"❌ Error: activities.json is not a dictionary, got {type(activities)}")
            return False
        
        # Check each emotion has a list of recommendations
        for emotion, recommendations in activities.items():
            if not isinstance(recommendations, list):
                print(f"❌ Error: recommendations for {emotion} is not a list, got {type(recommendations)}")
                return False
            
            # Check each recommendation is a string
            for i, rec in enumerate(recommendations):
                if not isinstance(rec, str):
                    print(f"❌ Error: recommendation {i} for {emotion} is not a string, got {type(rec)}")
                    return False
        
        # Check for expected emotions
        expected_emotions = ["happy", "sad", "angry", "neutral", "surprised", "fearful", "disgust", "calm"]
        for emotion in expected_emotions:
            if emotion not in activities:
                print(f"❌ Error: expected emotion '{emotion}' not found in activities.json")
                return False
        
        print("✅ activities.json is properly formatted and accessible")
        print(f"✅ Found {len(activities)} emotions with recommendations")
        for emotion, recommendations in activities.items():
            print(f"  - {emotion}: {len(recommendations)} recommendations")
        
        return True
    
    except json.JSONDecodeError as e:
        print(f"❌ Error: activities.json is not valid JSON: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_activities_json() 