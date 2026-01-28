#!/usr/bin/env python3
"""
Test production data flow for predictions
Kiểm tra toàn bộ flow dữ liệu từ frontend request → backend processing → predictions
"""

import requests
import json
import time
from datetime import datetime
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for local testing
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

BASE_URL = "https://web-production-a7244.up.railway.app"
# BASE_URL = "http://localhost:5000"  # Uncomment for localhost testing

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def test_health():
    """Test 1: Health check"""
    print_header("TEST 1: Health Check")
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=10, verify=False)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        return res.status_code == 200
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_leagues():
    """Test 2: Get leagues list"""
    print_header("TEST 2: Get Leagues")
    try:
        res = requests.get(f"{BASE_URL}/api/leagues", timeout=10, verify=False)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(f"Current League: {data.get('current')}")
        print(f"Available Leagues: {[l['code'] + ' (' + l['name'] + ')' for l in data.get('leagues', [])]}")
        return res.status_code == 200
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_league_switch(league_code):
    """Test 3: Switch league"""
    print_header(f"TEST 3: Switch to League {league_code}")
    try:
        res = requests.post(
            f"{BASE_URL}/api/league/select",
            json={"league_code": league_code},
            timeout=10,
            verify=False
        )
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        return res.status_code == 200
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_teams(league_code):
    """Test 4: Get teams for current league"""
    print_header(f"TEST 4: Get Teams for {league_code}")
    try:
        # Add cache buster
        res = requests.get(f"{BASE_URL}/api/teams?t={int(time.time() * 1000)}", timeout=10, verify=False)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(f"Backend League: {data.get('league')}")
        print(f"Teams Count: {len(data.get('teams', []))}")
        print(f"Teams: {data.get('teams', [])[:5]}... (showing first 5)")
        
        # Verify league matches
        if data.get('league') != league_code:
            print(f"⚠️  WARNING: Expected {league_code} but got {data.get('league')}")
            return False
        return res.status_code == 200
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_predict(home_team, away_team, league_code):
    """Test 5: Run prediction"""
    print_header(f"TEST 5: Predict {home_team} vs {away_team} ({league_code})")
    try:
        res = requests.post(
            f"{BASE_URL}/api/predict/all",
            json={
                "home_team": home_team,
                "away_team": away_team
            },
            timeout=30,
            verify=False
        )
        print(f"Status: {res.status_code}")
        data = res.json()
        
        if res.status_code != 200:
            print(f"❌ ERROR: {data.get('error')}")
            return False
        
        # Validate response structure
        print(f"\n✅ Prediction successful!")
        print(f"Match: {data['match']['home_team']} vs {data['match']['away_team']}")
        print(f"\nCorners Predictions:")
        for threshold, pred in data['corners']['predictions'].items():
            print(f"  Over {threshold}: {pred['prediction']} (P: {pred['probability']:.2%})")
        
        print(f"\nGoals Predictions:")
        for threshold, pred in data['goals']['predictions'].items():
            print(f"  Over {threshold}: {pred['prediction']} (P: {pred['probability']:.2%})")
        
        print(f"\nStatistics:")
        print(f"  Home Team Stats: {data['statistics']['home_team']['matches']} matches, {data['statistics']['home_team']['avg_corners']:.1f} avg corners")
        print(f"  Away Team Stats: {data['statistics']['away_team']['matches']} matches, {data['statistics']['away_team']['avg_corners']:.1f} avg corners")
        print(f"  Head to Head: {data['statistics']['head_to_head']}")
        
        print(f"\nLLM Assessment: {data.get('llm_assessment', 'N/A')[:200]}...")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_full_flow():
    """Run full flow test"""
    print("\n" + "="*70)
    print("  PRODUCTION DATA FLOW TEST SUITE")
    print(f"  Time: {datetime.now()}")
    print(f"  Base URL: {BASE_URL}")
    print("="*70)
    
    results = {}
    
    # Test each league
    leagues_to_test = [
        {"code": "E0", "teams": ["Arsenal", "Manchester United"]},
        {"code": "SP1", "teams": ["Barcelona", "Real Madrid"]},
        {"code": "I1", "teams": ["Juventus", "Napoli"]},
        {"code": "D1", "teams": ["Bayern Munich", "Borussia Dortmund"]},
        {"code": "F1", "teams": ["Paris Saint-Germain", "Olympique Marseille"]},
    ]
    
    results["health"] = test_health()
    results["leagues"] = test_leagues()
    
    for league in leagues_to_test:
        league_code = league["code"]
        home_team = league["teams"][0]
        away_team = league["teams"][1]
        
        print(f"\n\n{'#'*70}")
        print(f"  TESTING LEAGUE: {league_code}")
        print(f"{'#'*70}")
        
        # Switch league
        results[f"switch_{league_code}"] = test_league_switch(league_code)
        time.sleep(1)
        
        # Get teams
        results[f"teams_{league_code}"] = test_teams(league_code)
        time.sleep(1)
        
        # Make prediction
        results[f"predict_{league_code}"] = test_predict(home_team, away_team, league_code)
        time.sleep(2)
    
    # Print summary
    print_header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print(f"\nDetails:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

if __name__ == "__main__":
    test_full_flow()
