# AI Trip Planner - Real API Integration Specification

## Overview
This document outlines the implementation plan for replacing placeholder tools with real API integrations to provide actual, up-to-date travel information.

## Current State
All tools currently return deterministic placeholder strings. The LLM generates itineraries based solely on its training data, using tools only as semantic markers for workflow structure.

## Proposed API Integrations

### 1. Weather & Climate Data

#### Tool: `essential_info` → `get_weather_info`
**Current Output:** 
```python
"Key info for {destination}: mild weather, popular sights, local etiquette."
```

**Proposed APIs:**
- **OpenWeatherMap API** (Primary)
  - Endpoint: `api.openweathermap.org/data/2.5/forecast`
  - Features: 5-day forecast, current conditions, UV index
  - Cost: Free tier: 1000 calls/day
  - API Key Required: Yes

- **WeatherAPI** (Alternative)
  - Endpoint: `api.weatherapi.com/v1/forecast.json`
  - Features: 14-day forecast, historical weather, astronomy data
  - Cost: Free tier: 1M calls/month
  - API Key Required: Yes

**Implementation:**
```python
import httpx
from datetime import datetime
from typing import Dict, Any

@tool
async def get_weather_info(destination: str, travel_dates: Optional[str] = None) -> str:
    """Get real weather forecast and conditions for destination."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # Geocode destination first
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": destination, "limit": 1, "appid": api_key}
    
    async with httpx.AsyncClient() as client:
        geo_response = await client.get(geocode_url, params=params)
        if geo_response.status_code == 200:
            location = geo_response.json()[0]
            lat, lon = location['lat'], location['lon']
            
            # Get weather forecast
            weather_url = f"http://api.openweathermap.org/data/2.5/forecast"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "appid": api_key,
                "units": "metric"
            }
            weather_response = await client.get(weather_url, params=weather_params)
            
            if weather_response.status_code == 200:
                data = weather_response.json()
                # Process and format weather data
                temps = [item['main']['temp'] for item in data['list'][:5]]
                avg_temp = sum(temps) / len(temps)
                conditions = data['list'][0]['weather'][0]['description']
                
                return f"""Weather for {destination}:
                - Current conditions: {conditions}
                - Average temperature: {avg_temp:.1f}°C ({avg_temp*9/5+32:.1f}°F)
                - Pack for: {get_packing_suggestion(avg_temp)}
                - Best time to visit outdoor attractions: {get_best_times(data)}"""
    
    return f"Weather data temporarily unavailable for {destination}"
```

### 2. Budget & Pricing Information

#### Tool: `budget_basics` → `get_budget_breakdown`
**Current Output:**
```python
"Budget for {destination} over {duration}: lodging, food, transit, attractions."
```

**Proposed APIs:**
- **Numbeo API** (Cost of Living)
  - Features: Restaurant prices, transport costs, accommodation estimates
  - Cost: Paid API ($600/year for commercial use)
  - Alternative: Web scraping (with permission)

- **Budget Your Trip API**
  - Features: Average daily costs by destination and travel style
  - Cost: Free tier available
  - API Key Required: Yes

**Implementation:**
```python
@tool
async def get_budget_breakdown(
    destination: str, 
    duration: str, 
    budget: Optional[str] = None,
    travel_style: Optional[str] = "mid-range"
) -> str:
    """Get real budget breakdown based on destination costs."""
    
    # Parse duration to days
    days = parse_duration_to_days(duration)
    
    # Get cost data from Numbeo or fallback source
    costs = await fetch_destination_costs(destination)
    
    # Calculate budget breakdown
    daily_costs = {
        "budget": {
            "accommodation": costs.get("hostel_price", 25),
            "food": costs.get("cheap_meal", 10) * 3,
            "transport": costs.get("public_transport_day", 5),
            "activities": 10
        },
        "mid-range": {
            "accommodation": costs.get("hotel_3star", 80),
            "food": costs.get("mid_meal", 25) * 3,
            "transport": costs.get("taxi_day", 20),
            "activities": 30
        },
        "luxury": {
            "accommodation": costs.get("hotel_5star", 250),
            "food": costs.get("fine_dining", 50) * 3,
            "transport": costs.get("car_rental", 80),
            "activities": 100
        }
    }
    
    style_costs = daily_costs.get(travel_style, daily_costs["mid-range"])
    total_per_day = sum(style_costs.values())
    total_trip = total_per_day * days
    
    return f"""Budget breakdown for {destination} ({duration}/{days} days):
    
    Daily costs ({travel_style}):
    - Accommodation: ${style_costs['accommodation']}/night
    - Meals: ${style_costs['food']}/day
    - Local transport: ${style_costs['transport']}/day
    - Activities & entrance fees: ${style_costs['activities']}/day
    
    Total daily: ${total_per_day}
    Total trip estimate: ${total_trip}
    
    Money-saving tips:
    - Book accommodation in advance for better rates
    - Eat where locals eat for authentic and affordable meals
    - Use public transport or walk when possible
    - Look for free walking tours and activities"""
```

### 3. Local Experiences & Attractions

#### Tool: `local_flavor` → `get_local_recommendations`
**Current Output:**
```python
"Local experiences for {destination}: authentic food, culture, and {interests or 'top picks'}."
```

**Proposed APIs:**
- **TripAdvisor API** (via RapidAPI)
  - Features: Top attractions, restaurants, activities by location
  - Cost: Free tier: 500 calls/month
  - API Key Required: Yes

- **Google Places API**
  - Features: Detailed place information, reviews, photos, opening hours
  - Cost: $17 per 1000 requests (after free tier)
  - API Key Required: Yes

- **Foursquare Places API**
  - Features: Venue recommendations, tips, categories
  - Cost: Free tier: 100,000 calls/month
  - API Key Required: Yes

**Implementation:**
```python
@tool
async def get_local_recommendations(
    destination: str,
    interests: Optional[str] = None,
    limit: int = 10
) -> str:
    """Get real local recommendations from multiple sources."""
    
    foursquare_key = os.getenv("FOURSQUARE_API_KEY")
    
    # Parse interests into Foursquare categories
    categories = map_interests_to_categories(interests)
    
    async with httpx.AsyncClient() as client:
        # Search for places
        url = "https://api.foursquare.com/v3/places/search"
        headers = {
            "Authorization": foursquare_key,
            "Accept": "application/json"
        }
        params = {
            "near": destination,
            "categories": ",".join(categories),
            "limit": limit,
            "sort": "RATING"
        }
        
        response = await client.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            places = response.json()['results']
            
            recommendations = []
            for place in places[:5]:
                rec = {
                    "name": place['name'],
                    "category": place['categories'][0]['name'],
                    "address": place['location'].get('formatted_address', 'Address not available'),
                    "rating": place.get('rating', 'Not rated'),
                    "tip": await get_venue_tip(place['fsq_id'], client, headers)
                }
                recommendations.append(rec)
            
            return format_recommendations(destination, interests, recommendations)
    
    return f"Local recommendations temporarily unavailable for {destination}"

def format_recommendations(destination: str, interests: str, recommendations: list) -> str:
    """Format recommendations into readable text."""
    output = f"Top local experiences in {destination}"
    if interests:
        output += f" for {interests}:\n\n"
    else:
        output += ":\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        output += f"{i}. {rec['name']} ({rec['category']})\n"
        output += f"   📍 {rec['address']}\n"
        if rec['rating']:
            output += f"   ⭐ Rating: {rec['rating']}/10\n"
        if rec['tip']:
            output += f"   💡 Tip: {rec['tip']}\n"
        output += "\n"
    
    return output
```

### 4. Attraction Prices & Tickets

#### Tool: `attraction_prices` → `get_attraction_pricing`
**Current Output:**
```python
"Attraction prices in {destination}: Museum: $10-$40, Historic Site: $10-$40, Viewpoint: $10-$40"
```

**Proposed APIs:**
- **GetYourGuide API**
  - Features: Tour prices, attraction tickets, availability
  - Cost: Affiliate commission-based
  - API Key Required: Yes (Partner account)

- **Viator API** (TripAdvisor)
  - Features: Tours, activities, prices, availability
  - Cost: Affiliate commission-based
  - API Key Required: Yes (Partner account)

**Implementation:**
```python
@tool
async def get_attraction_pricing(
    destination: str,
    attractions: Optional[List[str]] = None
) -> str:
    """Get real attraction prices and ticket information."""
    
    viator_key = os.getenv("VIATOR_API_KEY")
    
    if not attractions:
        # Get top attractions for destination
        attractions = await get_top_attractions(destination)
    
    pricing_info = []
    
    async with httpx.AsyncClient() as client:
        for attraction in attractions[:5]:
            # Search for tours/tickets
            search_url = "https://viator.com/api/search"
            params = {
                "destination": destination,
                "query": attraction,
                "currency": "USD"
            }
            headers = {"api-key": viator_key}
            
            response = await client.get(search_url, headers=headers, params=params)
            
            if response.status_code == 200:
                results = response.json()
                if results['products']:
                    product = results['products'][0]
                    pricing_info.append({
                        "name": product['title'],
                        "price": product['price']['amount'],
                        "currency": product['price']['currency'],
                        "duration": product.get('duration', 'Variable'),
                        "includes": product.get('inclusions', [])
                    })
    
    return format_pricing_info(destination, pricing_info)
```

### 5. Visa & Travel Requirements

#### Tool: `visa_brief` → `get_visa_requirements`
**Current Output:**
```python
"Visa guidance for {destination}: check your nationality's embassy site."
```

**Proposed APIs:**
- **Sherpa API** (Visa & Travel Requirements)
  - Features: Visa requirements by nationality, COVID restrictions, travel documents
  - Cost: Enterprise pricing
  - API Key Required: Yes

- **IATA Travel Centre API**
  - Features: Entry requirements, health requirements, customs
  - Cost: Paid service
  - Alternative: Scrape public Timatic data

**Implementation:**
```python
@tool
async def get_visa_requirements(
    destination_country: str,
    nationality: str = "US",
    trip_duration: Optional[int] = 30
) -> str:
    """Get real visa and entry requirements."""
    
    # For demonstration, using a simplified approach
    # In production, would integrate with Sherpa or IATA
    
    visa_db = {
        # Simplified visa database
        ("Thailand", "US"): {
            "visa_required": False,
            "visa_on_arrival": True,
            "max_stay": 30,
            "requirements": ["Valid passport (6+ months)", "Return ticket", "Proof of accommodation"]
        },
        ("Japan", "US"): {
            "visa_required": False,
            "visa_on_arrival": False,
            "max_stay": 90,
            "requirements": ["Valid passport", "Return ticket"]
        },
        # ... more entries
    }
    
    key = (destination_country, nationality)
    if key in visa_db:
        info = visa_db[key]
        return f"""Visa requirements for {nationality} citizens visiting {destination_country}:
        
        Visa required: {'Yes' if info['visa_required'] else 'No'}
        Visa on arrival: {'Available' if info['visa_on_arrival'] else 'Not available'}
        Maximum stay: {info['max_stay']} days
        
        Requirements:
        {format_requirements(info['requirements'])}
        
        ⚠️ Always verify with official sources before travel."""
    
    return f"Please check official embassy website for {destination_country} visa requirements"
```

### 6. Real-time Flight Information

#### New Tool: `get_flight_options`
**Proposed APIs:**
- **Amadeus API**
  - Features: Flight search, pricing, availability
  - Cost: Free tier: 500 calls/month
  - API Key Required: Yes

- **Skyscanner API**
  - Features: Flight search, price comparison
  - Cost: RapidAPI pricing tiers
  - API Key Required: Yes

**Implementation:**
```python
@tool
async def get_flight_options(
    origin: str,
    destination: str,
    departure_date: Optional[str] = None,
    return_date: Optional[str] = None,
    budget: Optional[int] = None
) -> str:
    """Get real flight options and pricing."""
    
    amadeus_key = os.getenv("AMADEUS_API_KEY")
    amadeus_secret = os.getenv("AMADEUS_API_SECRET")
    
    # Get access token
    token = await get_amadeus_token(amadeus_key, amadeus_secret)
    
    # Search for flights
    search_url = "https://api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": get_airport_code(origin),
        "destinationLocationCode": get_airport_code(destination),
        "departureDate": departure_date or get_next_month_date(),
        "adults": 1,
        "max": 5
    }
    
    if return_date:
        params["returnDate"] = return_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(search_url, headers=headers, params=params)
        
        if response.status_code == 200:
            flights = response.json()['data']
            return format_flight_options(flights, budget)
    
    return "Flight information temporarily unavailable"
```

## Implementation Strategy

### Phase 1: Core APIs (Week 1-2)
1. **Weather API** (OpenWeatherMap)
   - Essential for trip planning
   - Free tier sufficient for MVP
   
2. **Local Recommendations** (Foursquare)
   - Rich venue data
   - Generous free tier

3. **Budget Data** (Numbeo scraping or API)
   - Critical for budget planning
   - Consider caching strategies

### Phase 2: Enhanced Features (Week 3-4)
1. **Attraction Pricing** (Viator/GetYourGuide)
   - Affiliate revenue potential
   - Real booking capability

2. **Flight Search** (Amadeus)
   - Complete trip planning
   - Price comparison features

3. **Visa Requirements** (Build database or Sherpa API)
   - Essential for international travel
   - Consider subscription cost

### Phase 3: Premium Features (Week 5-6)
1. **Hotel Recommendations** (Booking.com API)
2. **Restaurant Reservations** (OpenTable API)
3. **Car Rental Options** (RentalCars API)
4. **Travel Insurance** (Partner APIs)

## Technical Considerations

### 1. API Key Management
```python
# .env file structure
OPENWEATHER_API_KEY=xxx
FOURSQUARE_API_KEY=xxx
VIATOR_API_KEY=xxx
AMADEUS_API_KEY=xxx
AMADEUS_API_SECRET=xxx
GOOGLE_PLACES_API_KEY=xxx
```

### 2. Rate Limiting & Caching
```python
from functools import lru_cache
from datetime import datetime, timedelta
import redis

# Redis for distributed caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_api_response(key: str, data: Any, ttl: int = 3600):
    """Cache API responses to reduce API calls and costs."""
    redis_client.setex(key, ttl, json.dumps(data))

def get_cached_response(key: str) -> Optional[Any]:
    """Retrieve cached API response."""
    data = redis_client.get(key)
    return json.loads(data) if data else None

# Rate limiting decorator
def rate_limit(calls_per_minute: int):
    def decorator(func):
        last_called = []
        
        async def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls older than 1 minute
            last_called[:] = [t for t in last_called if now - t < 60]
            
            if len(last_called) >= calls_per_minute:
                wait_time = 60 - (now - last_called[0])
                await asyncio.sleep(wait_time)
            
            last_called.append(now)
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
```

### 3. Error Handling & Fallbacks
```python
class APIError(Exception):
    """Custom exception for API errors."""
    pass

async def fetch_with_fallback(
    primary_func,
    fallback_func,
    *args,
    **kwargs
):
    """Try primary API, fall back to secondary if failed."""
    try:
        return await primary_func(*args, **kwargs)
    except (APIError, httpx.RequestError):
        logger.warning(f"Primary API failed, using fallback")
        return await fallback_func(*args, **kwargs)
```

### 4. Async Implementation
Convert all tools to async for better performance:
```python
# Update agent functions to handle async tools
async def research_agent(state: TripState) -> TripState:
    # ... agent logic with await for async tools
    pass
```

## Cost Analysis

### Monthly API Costs (Estimated)
| API | Free Tier | Paid Tier | Monthly Est. |
|-----|-----------|-----------|--------------|
| OpenWeatherMap | 1000/day | $0 | $0 |
| Foursquare | 100k/month | $299/month | $0 (free tier) |
| Google Places | $200 credit | $17/1000 | ~$50 |
| Amadeus | 500/month | Variable | $0 (free tier) |
| Viator | Commission | Commission | $0 (commission) |
| **Total** | | | **~$50/month** |

### Cost Optimization Strategies
1. **Aggressive Caching**: Cache responses for 24-48 hours
2. **Batch Requests**: Combine multiple queries when possible
3. **Conditional Fetching**: Only call APIs when data is needed
4. **Tiered Service**: Premium users get real-time data, free users get cached
5. **Fallback to LLM**: Use GPT knowledge when API limits reached

## Security Considerations

### 1. API Key Security
- Never expose keys in client-side code
- Use environment variables
- Implement key rotation
- Monitor usage for anomalies

### 2. Rate Limiting
- Implement per-user rate limits
- Use Redis for distributed rate limiting
- Queue system for high load

### 3. Data Privacy
- Don't store personal travel data longer than necessary
- Implement GDPR compliance
- Anonymize analytics data

## Testing Strategy

### 1. Mock APIs for Development
```python
class MockWeatherAPI:
    async def get_weather(self, destination: str):
        return {
            "temp": 22,
            "condition": "sunny",
            "forecast": "clear skies"
        }

# Use mocks in test environment
if os.getenv("ENVIRONMENT") == "test":
    weather_api = MockWeatherAPI()
else:
    weather_api = RealWeatherAPI()
```

### 2. Integration Tests
- Test each API integration separately
- Test fallback mechanisms
- Test rate limiting behavior
- Test caching functionality

### 3. Load Testing
- Simulate concurrent users
- Test API rate limits
- Monitor response times
- Test cache performance

## Monitoring & Analytics

### 1. API Usage Tracking
```python
def track_api_usage(api_name: str, endpoint: str, status: str):
    """Track API usage for monitoring and optimization."""
    metrics = {
        "api": api_name,
        "endpoint": endpoint,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    # Send to monitoring service (e.g., Datadog, CloudWatch)
    send_metrics(metrics)
```

### 2. Cost Monitoring
- Track API calls per user
- Monitor approaching limits
- Alert on unusual usage
- Monthly cost reports

### 3. Performance Metrics
- API response times
- Cache hit rates
- Error rates by API
- User experience metrics

## Migration Plan

### Step 1: Parallel Implementation
- Keep placeholder tools
- Add real API tools alongside
- Feature flag to switch between them

### Step 2: Gradual Rollout
- Test with internal users
- Roll out to 10% of users
- Monitor performance and costs
- Full rollout when stable

### Step 3: Deprecate Placeholders
- Remove placeholder tools
- Update documentation
- Monitor for issues

## Conclusion

This specification provides a roadmap for transforming the AI Trip Planner from a demo system with placeholder data to a production-ready application with real, actionable travel information. The phased approach allows for gradual implementation while managing costs and complexity.

Key benefits of real API integration:
- **Accurate Information**: Real-time weather, prices, and availability
- **Better User Experience**: Actionable recommendations with current data
- **Revenue Potential**: Affiliate commissions from bookings
- **Competitive Advantage**: Superior to generic AI-only solutions

Next steps:
1. Prioritize APIs based on user needs
2. Set up API accounts and keys
3. Implement Phase 1 APIs
4. Test with real users
5. Iterate based on feedback