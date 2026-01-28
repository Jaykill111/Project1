# 📋 PRODUCTION DATA FLOW CHECKLIST

## 🎯 Test thành phần chính của dữ liệu dự đoán

### 1. **LOAD TRANG VERCEL**
- [ ] Vào https://myproject-woad-theta.vercel.app
- [ ] Page load thành công (không lỗi, không blank)
- [ ] Thấy "I1 SCOPE" header với league dropdown

### 2. **CHECK LEAGUE PERSISTENCE**
- [ ] Chọn league "SP1 - La Liga" → reload trang
- [ ] Teams dropdown phải show **Spain teams** (Barcelona, Real Madrid, etc.)
- [ ] KHÔNG được show England teams (Arsenal, Manchester, etc.)

**Debug:** Console → xem logs:
```
[DEBUG] Initial load: Switching backend from E0 to SP1
[DEBUG] Backend switch complete: {success: true, ...}
[DEBUG] Fetching teams for league: SP1
[DEBUG] /api/teams data: {league: "SP1", teams: [...]}
```

---

### 3. **LEAGUE SWITCHING TEST**
Test flow: **E0** → **I1** → **SP1** → **D1** → **F1**

#### **E0 - Premier League (England)**
- [ ] Select E0
- [ ] Teams: Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, ...
- [ ] Count: ~20 teams

#### **I1 - Serie A (Italy)**
- [ ] Select I1  
- [ ] Teams: AC Milan, AS Roma, Atalanta, Bologna, Fiorentina, **Juventus**, Lazio, Napoli, ...
- [ ] Count: ~20 teams
- [ ] **CRITICAL**: No England teams visible

#### **SP1 - La Liga (Spain)**
- [ ] Select SP1
- [ ] Teams: Alaves, Athletic Bilbao, **Barcelona**, Betis, Celta, Elche, Espanol, Getafe, Girona, **Real Madrid**, ...
- [ ] Count: ~20 teams

#### **D1 - Bundesliga (Germany)**
- [ ] Select D1
- [ ] Teams: **Augsburg**, **Bayern Munich**, **Borussia Dortmund**, Bochum, ...
- [ ] Count: ~18 teams

#### **F1 - Ligue 1 (France)**
- [ ] Select F1
- [ ] Teams: Ajaccio, Angers, Auxerre, Brest, Clermont, Lille, Lyon, Marseille, **Monaco**, **PSG**, ...
- [ ] Count: ~18 teams

---

### 4. **PREDICTION TEST FOR EACH LEAGUE**

#### **E0: Arsenal vs Manchester United**
- [ ] Select home: Arsenal
- [ ] Select away: Manchester United
- [ ] Click PREDICT
- [ ] Get corners predictions (8.5, 9.5, 10.5, 11.5, 12.5 thresholds)
- [ ] Get goals predictions (1.5, 2.5, 3.5, 4.5 thresholds)
- [ ] Get statistics (avg corners, goals, H2H history)
- [ ] Get LLM assessment

#### **I1: Juventus vs Napoli**
- [ ] Select home: Juventus
- [ ] Select away: Napoli
- [ ] Click PREDICT
- [ ] Verify predictions show **Italy** statistics (not England)

#### **SP1: Barcelona vs Real Madrid**
- [ ] Select home: Barcelona
- [ ] Select away: Real Madrid
- [ ] Click PREDICT
- [ ] Verify predictions show **Spain** data

#### **D1: Bayern Munich vs Borussia Dortmund**
- [ ] Select home: Bayern Munich
- [ ] Select away: Borussia Dortmund
- [ ] Click PREDICT
- [ ] Verify predictions show **Germany** data

#### **F1: PSG vs Olympique Marseille**
- [ ] Select home: Paris Saint-Germain
- [ ] Select away: Olympique Marseille
- [ ] Click PREDICT
- [ ] Verify predictions show **France** data

---

### 5. **MODELS & DATA VERIFICATION**
Open **DevTools** (F12) → **Network** tab

**Check `/api/league/select` response:**
```json
{
  "success": true,
  "league": { "code": "SP1", "name": "La Liga", ... },
  "models_loaded": { "corners": 5, "goals": 4 },
  "message": "Switched to La Liga"
}
```

**Check `/api/teams` response:**
```json
{
  "teams": ["Barcelona", "Real Madrid", ...],
  "league": "SP1"  // ← Must match selected league
}
```

**Check `/api/predict/all` response:**
```json
{
  "match": { "home_team": "Barcelona", "away_team": "Real Madrid" },
  "corners": {
    "predictions": {
      "8.5": { "prediction": "OVER", "probability": 0.75 },
      ...
    }
  },
  "goals": { "predictions": {...} },
  "statistics": {
    "home_team": { "matches": 15, "avg_corners": 5.2, ... },
    "away_team": { "matches": 14, "avg_corners": 4.8, ... }
  }
}
```

---

### 6. **EDGE CASES**

- [ ] **Same team error**: Select "Barcelona" vs "Barcelona" → Should show error "Home and away teams must be different"
- [ ] **Missing team error**: Try typing invalid team name → Should show error
- [ ] **Multiple predictions**: Make 3-4 predictions in sequence → Each should be independent, correct data
- [ ] **Rapid league switching**: Click through 5 leagues in quick succession → Last selected league should be correct
- [ ] **Browser back/forward**: Predict E0 → Go back → Forward → Teams should remain consistent

---

### 7. **DATA FRESHNESS**

- [ ] Check if predictions change over time (match live data from football-data.co.uk)
- [ ] Verify team statistics are from current season (2025-26)
- [ ] Verify H2H history shows recent matches

---

### 8. **CONSOLE ERRORS**
- [ ] No red errors in DevTools Console
- [ ] No 404/500 errors in Network tab
- [ ] No CORS errors
- [ ] No SSL certificate warnings

---

## 📊 Expected Results

| League | Data Source | Teams | Models | Status |
|--------|------------|-------|--------|--------|
| E0     | football-data.co.uk/2526/E0.csv | ~20 | 9 | ✓ |
| SP1    | football-data.co.uk/2526/SP1.csv | ~20 | 9 | ✓ |
| I1     | football-data.co.uk/2526/I1.csv | ~20 | 9 | ✓ |
| D1     | football-data.co.uk/2526/D1.csv | ~18 | 9 | ✓ |
| F1     | football-data.co.uk/2526/F1.csv | ~18 | 9 | ✓ |

**Models per league:** 5 corner (8.5-12.5) + 4 goals (1.5-4.5) = **9 models**

---

## 🔧 Debugging Commands

If issues found, run in browser console:

```javascript
// Check current league
console.log(currentLeague);

// Check loaded teams
console.log(teams);

// Test API call
fetch('https://web-production-a7244.up.railway.app/api/health?t=' + Date.now(), 
  {headers: {'Content-Type': 'application/json'}})
  .then(r => r.json())
  .then(d => console.log(d));

// Test prediction
fetch('https://web-production-a7244.up.railway.app/api/predict/all?t=' + Date.now(), {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({home_team: 'Arsenal', away_team: 'Manchester United'})
})
  .then(r => r.json())
  .then(d => console.log(d));
```

---

## ✅ Success Criteria

- ✅ All 5 leagues load correct teams
- ✅ Predictions run for any league combination
- ✅ Statistics show correct league data
- ✅ No data cross-contamination (Spain data for England league, etc.)
- ✅ Models load correctly for each league (5 corners + 4 goals)
- ✅ Page doesn't crash or show errors
- ✅ Data persists across page reload
- ✅ League switching is instant and accurate
