def is_open_palm(hand_landmarks):

    tips = [8, 12, 16, 20]    # index, middle, ring, pinky tips
    bases = [6, 10, 14, 18]   # their base knuckles
    
    fingers_up = sum(
        1 for tip, base in zip(tips, bases)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y
    )
    
    return fingers_up == 4


def are_both_palms_open(hand_landmarks_list):
    
    if len(hand_landmarks_list) != 2:
        return False
    
    return all(is_open_palm(hand) for hand in hand_landmarks_list)