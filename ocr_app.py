import ollama
import os

# Konfiguracja ścieżek
PHOTOS_DIR = "photos"
OUTPUT_FILE = "result.md"

def process_all_photos():
    # 1. Sprawdzamy czy folder istnieje, jeśli nie - tworzymy go
    if not os.path.exists(PHOTOS_DIR):
        os.makedirs(PHOTOS_DIR)
        print(f"📁 Utworzono folder '{PHOTOS_DIR}'.")
        print(f"Wrzuć do niego swoje obrazki i uruchom skrypt ponownie!")
        return

    # 2. Pobieramy listę obrazków z folderu
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    images =[f for f in os.listdir(PHOTOS_DIR) if f.lower().endswith(valid_extensions)]

    # Sortujemy alfabetycznie, żeby zachować kolejność (opcjonalne)
    images.sort()

    if not images:
        print(f"🤷 Brak obrazków w folderze '{PHOTOS_DIR}'. Dodaj pliki (np. jpg, png) i spróbuj ponownie.")
        return

    print(f"🔎 Znaleziono {len(images)} obrazków. Rozpoczynam przetwarzanie OCR (GLM-OCR) na AMD GPU...\n")

    # 3. Otwieramy plik wynikowy w trybie nadpisywania ('w')
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        out_file.write("# 📄 Zbiorczy wynik OCR\n\n")

        # 4. Przetwarzamy obrazki w pętli
        for img_name in images:
            img_path = os.path.join(PHOTOS_DIR, img_name)
            print(f"⏳ Przetwarzanie: {img_name}...")

            with open(img_path, 'rb') as file:
                image_bytes = file.read()

            try:
                # Wywołanie modelu GLM-OCR z odpowiednim zadaniem
                response = ollama.generate(
                    model='glm-ocr',
                    prompt="Text Recognition:",
                    images=[image_bytes]
                )

                extracted_text = response.get('response', '').strip()

                # 5. Formatowanie zapisu w pliku Markdown
                out_file.write(f"## 🖼️ Plik: `{img_name}`\n\n")
                out_file.write(extracted_text)
                out_file.write("\n\n---\n\n")  # Linia oddzielająca dokumenty

                print(f"✅ Zakończono: {img_name}")

            except Exception as e:
                print(f"❌ Błąd podczas przetwarzania {img_name}: {e}")
                out_file.write(f"## 🖼️ Plik: `{img_name}`\n\n")
                out_file.write(f"*Wystąpił błąd OCR: {e}*\n\n---\n\n")

    print(f"\n🎉 Gotowe! Wszystkie wyniki zostały zapisane w pliku: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_photos()