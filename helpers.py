# ----------------------------
# Normalization & OSC Helpers
# ----------------------------
class Norm:
    maxPosTheta1 = 30.0
    maxNegTheta1 = -25.0
    maxAbsTheta2  = 30.0

def clamp(v: float, mn: float = -1.0, mx: float = 1.0) -> float:
    return max(mn, min(mx, v))

def scale_and_clamp(v: float, mul: float) -> float:
    return clamp(v * mul)

def scale_offset_and_clamp(v: float, offset: float, mul: float) -> float:
    return clamp((v - offset) * mul)

def calculate_offset_fraction(pitch_offset: float) -> float:
    if pitch_offset == 0:
        return 0.0
    span = Norm.maxPosTheta1 + abs(Norm.maxNegTheta1)
    return (2 * pitch_offset) / span

def normalize_theta1(v: float) -> float:
    return v / Norm.maxPosTheta1 if v >= 0 else v / abs(Norm.maxNegTheta1)

def normalize_theta2(v: float) -> float:
    return v / Norm.maxAbsTheta2

def transform_openness(val: float, cfg: list) -> float:
    h0, h1, h2, h3 = cfg
    if val < h0:
        return 0.0
    elif val < h1:
        return ((val - h0) / (h1 - h0)) * 0.75
    elif val < h2:
        return 0.75
    elif val < h3:
        return 0.75 + ((val - h2) / (h3 - h2)) * 0.25
    else:
        return 1.0