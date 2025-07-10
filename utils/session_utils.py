def get_origin_session(datetime_utc):
    hour = datetime_utc.hour
    # Define sessions (UTC)
    if 21 <= hour or hour < 7:
        return 'Asian'
    elif 7 <= hour < 12:
        return 'London'
    elif 12 <= hour < 21:
        return 'New York'
    else:
        return 'Unknown'
