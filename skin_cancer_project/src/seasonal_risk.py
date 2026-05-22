"""
Seasonal & UV Risk Analysis for Skin Cancer
=============================================
Computes UV exposure risk based on season, hemisphere,
skin type, and lesion location on body.
"""

from datetime import datetime


UV_INDEX_BY_MONTH = {
    "northern": {
        1: 1.5, 2: 2.0, 3: 3.5, 4: 5.0,
        5: 7.0, 6: 9.5, 7: 10.0, 8: 9.0,
        9: 6.5, 10: 4.0, 11: 2.5, 12: 1.5
    },
    "southern": {
        1: 10.0, 2: 9.0, 3: 6.5, 4: 4.0,
        5: 2.5, 6: 1.5, 7: 1.5, 8: 2.0,
        9: 3.5, 10: 5.0, 11: 7.0, 12: 9.5
    }
}

SKIN_TYPE_MULTIPLIER = {
    "type_1": 1.5,  # Always burns, never tans
    "type_2": 1.3,  # Usually burns, tans minimally
    "type_3": 1.1,  # Sometimes burns, tans uniformly
    "type_4": 0.9,  # Rarely burns, always tans
    "type_5": 0.7,  # Very rarely burns
    "type_6": 0.6,  # Never burns
}

EXPOSED_LOCATIONS = {
    "face": 1.4, "scalp": 1.3, "ear": 1.3,
    "neck": 1.2, "hand": 1.2, "acral": 1.1,
    "upper extremity": 1.1, "chest": 0.9,
    "back": 0.9, "lower extremity": 0.8,
    "foot": 0.7, "abdomen": 0.6,
    "trunk": 0.6, "genital": 0.3,
}

SEASON_NAMES = {
    "northern": {
        (12, 1, 2): "Winter", (3, 4, 5): "Spring",
        (6, 7, 8): "Summer", (9, 10, 11): "Autumn"
    },
    "southern": {
        (12, 1, 2): "Summer", (3, 4, 5): "Autumn",
        (6, 7, 8): "Winter", (9, 10, 11): "Spring"
    }
}


def get_season(month: int, hemisphere: str = "northern") -> str:
    for months, name in SEASON_NAMES[hemisphere].items():
        if month in months:
            return name
    return "Unknown"


def compute_seasonal_risk(
    month: int = None,
    hemisphere: str = "northern",
    skin_type: str = "type_2",
    localization: str = "face",
    latitude: float = None
) -> dict:
    if month is None:
        month = datetime.now().month

    base_uv = UV_INDEX_BY_MONTH[hemisphere][month]

    # Latitude adjustment (higher latitude = lower UV)
    if latitude is not None:
        lat_factor = max(0.4, 1.0 - (abs(latitude) - 20) * 0.01)
        base_uv *= lat_factor

    skin_mult = SKIN_TYPE_MULTIPLIER.get(skin_type, 1.0)
    loc_mult  = EXPOSED_LOCATIONS.get(localization, 0.8)

    risk_score = round(base_uv * skin_mult * loc_mult, 2)
    season     = get_season(month, hemisphere)

    # Risk level
    if risk_score >= 9:
        risk_level, risk_color = "Extreme", "danger"
    elif risk_score >= 6:
        risk_level, risk_color = "High", "warning"
    elif risk_score >= 3:
        risk_level, risk_color = "Moderate", "caution"
    else:
        risk_level, risk_color = "Low", "safe"

    # Monthly UV trend
    monthly_uv = UV_INDEX_BY_MONTH[hemisphere]

    return {
        "month": month,
        "season": season,
        "hemisphere": hemisphere,
        "base_uv_index": base_uv,
        "skin_type": skin_type,
        "localization": localization,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "monthly_uv_trend": monthly_uv,
        "recommendations": _get_recommendations(risk_level, season),
    }


def _get_recommendations(risk_level: str, season: str) -> list:
    base = [
        "Perform monthly self-skin examinations",
        "Use SPF 30+ sunscreen daily",
        "Wear protective clothing outdoors",
    ]
    if risk_level in ("High", "Extreme"):
        return [
            f"UV risk is {risk_level.upper()} this {season} — limit sun exposure 10am–4pm",
            "Apply SPF 50+ sunscreen every 2 hours outdoors",
            "Wear wide-brimmed hat and UV-blocking sunglasses",
            "Schedule annual dermatologist screening",
        ] + base
    elif risk_level == "Moderate":
        return [
            f"Moderate UV risk in {season} — use SPF 30+ consistently",
            "Seek shade during peak hours (11am–3pm)",
        ] + base
    return base
