    Wybrano N = 150 użytkowników. Dla każdego użytkownika usunięto losowo 20% książek, z którymi
wcześniej wchodził w interakcję. Następnie, dla każdego z nich wygenerowano 15 rekomendacji (top_n = 15). Na tej
podstawie obliczono wybrane metryki skuteczności rekomendacji.

Zastosowane metryki:
    - Precision@k – odsetek rekomendowanych książek (z top-k), które faktycznie należały do usuniętych
(czyli były trafne).
    - Recall@k – odsetek usuniętych książek, które znalazły się wśród rekomendowanych (czyli zostały
odzyskane).
