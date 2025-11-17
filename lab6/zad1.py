import cv2
import numpy as np
import os
from pathlib import Path

def count_birds(image_path):
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Nie można wczytać obrazu: {image_path}")
        return 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptacyjne progowanie
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15,  # rozmiar bloku - większy dla lepszego wykrywania małych obiektów
        3    # stała odejmowana od średniej
    )
    
    # Operacje morfologiczne do oczyszczenia obrazu
    kernel = np.ones((2, 2), np.uint8)
    
    # Zamknięcie - łączy blisko położone piksele
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Otwarcie - usuwa pojedyncze piksele szumu
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Znajduje kontury
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtruj kontury według rozmiaru
    min_area = 5      # minimalna powierzchnia pikseli
    max_area = 500    # maksymalna powierzchnia pikseli
    
    valid_birds = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_birds += 1
    
    return valid_birds


def process_folder(folder_path):
    
    results = []
    
    if not os.path.exists(folder_path):
        print(f"Folder nie istnieje: {folder_path}")
        return
    
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        bird_count = count_birds(file_path)
        results.append((filename, bird_count))
        print(f"{filename}: {bird_count} ptak(ów)")
    
    print("\n" + "-"*50)
    print("Wynik:")
    print("-"*50)
    for filename, count in results:
        print(f"{filename}: {count}")
    
    total_birds = sum(count for _, count in results)
    print(f"\nŁącznie obrazów: {len(results)}")
    print(f"Łącznie ptaków: {total_birds}")


if __name__ == "__main__":
    folder_path = "bird_miniatures"  
    
    process_folder(folder_path)
    