# @title Define the get_weather Tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
    city_normalized = city.lower().replace(" ", "") # Basic normalization / 소문자로 바꿔주고 공백 제거

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    error_db = {
        "paris": {"status": "error", "error_message": "Sorry, I don't have weather information for Paris LOL."},
        "berlin": {"status": "error", "error_message": "Sorry, I don't have weather information for Berlin."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    
    elif city_normalized not in mock_weather_db:
        return error_db[city_normalized]
    
        # return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}
    # else:
    #     return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

if __name__ == "__main__":
    # Example usage
    print(get_weather("New York"))
    print(get_weather("London"))
    print(get_weather("Tokyo"))
    print(get_weather("Paris"))  # Not in mock data, should return error