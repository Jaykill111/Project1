"""Test league API endpoints"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000/api"

def test_get_leagues():
    """Test GET /api/leagues"""
    print("\n=== Testing GET /api/leagues ===")
    response = requests.get(f"{BASE_URL}/leagues")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_get_current_league():
    """Test GET /api/league/current"""
    print("\n=== Testing GET /api/league/current ===")
    response = requests.get(f"{BASE_URL}/league/current")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_select_league(league_code):
    """Test POST /api/league/select"""
    print(f"\n=== Testing POST /api/league/select (code: {league_code}) ===")
    response = requests.post(
        f"{BASE_URL}/league/select",
        json={"league_code": league_code}
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        # Test 1: Get all leagues
        test_get_leagues()
        
        # Test 2: Get current league
        test_get_current_league()
        
        # Test 3: Select La Liga
        test_select_league("SP1")
        
        # Test 4: Verify league changed
        test_get_current_league()
        
        # Test 5: Switch back to EPL
        test_select_league("E0")
        test_get_current_league()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
