import ollama
import os
import io
import sys
import shutil
from PIL import Image
from pdf2image import convert_from_path

# Konfiguracja ścieżek
PHOTOS_DIR = "photos"
PROCESSED_DIR = "processed_photos"
OUTPUT_FILE = "result.md"
MODEL_NAME = 'qwen3-vl:8b-instruct'

# Lekko zmodyfikowany prompt ze "słowami-kluczami" ułatwiającymi weryfikację
QWEN_PROMPT = (
    "You are an expert OCR system. Extract all visible text from this image accurately.\n\n"
    "LAYOUT:\n"
    "- Use standard Markdown (paragraphs, headings, lists).\n"
    "- Use tables ONLY if explicit tabular data is present.\n"
    "- Use nested lists for non-linear layouts (mind maps, diagrams).\n\n"
    "UNCERTAIN TEXT RULES:\n"
    "1.IF AND ONLY IF you found uncertain/unreadable text output the best guess in the main body.\n" \
    "2.When the word is uncertain, use [UNCERTAIN] next to it to indicate the uncertainty.\n\n"

    "Output ONLY the extracted Markdown. No conversational text."
)

def ensure_model_exists(model_name):
    print(f"🔄 Sprawdzanie dostępności modelu '{model_name}'...")
    try:
        ollama.show(model_name)
        print(f"✅ Model '{model_name}' jest gotowy do pracy.")
    except Exception as e:
        if 'not found' in str(e).lower() or '404' in str(e):
            print(f"📥 Pobieranie modelu '{model_name}' (to może potrwać kilka minut)...")
            ollama.pull(model_name)
            print(f"✅ Pobieranie zakończone z sukcesem!")
        else:
            print(f"❌ Krytyczny błąd połączenia z Ollamą: {e}")
            sys.exit(1)

def optimize_image_for_ai(image_path, filename, max_dimension=3000):
    base_name = os.path.splitext(filename)[0]
    processed_path = os.path.join(PROCESSED_DIR, f"{base_name}_optimized.jpg")
    
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if max(img.width, img.height) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                print(f"[Optymalizacja] Zmniejszono rozdzielczość do {img.width}x{img.height} px")
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=90)
            img_bytes = img_byte_arr.getvalue()
            
            with open(processed_path, 'wb') as f:
                f.write(img_bytes)
            
            original_size = os.path.getsize(image_path) / (1024 * 1024)
            new_size = len(img_bytes) / (1024 * 1024)
            print(f"[Optymalizacja] Waga: {original_size:.2f} MB -> {new_size:.2f} MB")
            
            return img_bytes
            
    except Exception as e:
        print(f"   ❌ Błąd optymalizacji obrazu: {e}")
        with open(image_path, 'rb') as f:
            original_bytes = f.read()
            
        fallback_path = os.path.join(PROCESSED_DIR, filename)
        with open(fallback_path, 'wb') as f:
            f.write(original_bytes)
            
        return original_bytes

def process_pdf_pages(pdf_path, filename):
    base_name = os.path.splitext(filename)[0]
    
    try:
        pages = convert_from_path(pdf_path, dpi=250)
        images_bytes =[]
        
        for page_num, page in enumerate(pages, 1):
            img_byte_arr = io.BytesIO()
            rgb_page = page.convert('RGB')
            rgb_page.save(img_byte_arr, format='JPEG', quality=85)
            page_bytes = img_byte_arr.getvalue()
            
            processed_path = os.path.join(PROCESSED_DIR, f"{base_name}_page_{page_num}.jpg")
            with open(processed_path, 'wb') as f:
                f.write(page_bytes)
                
            images_bytes.append(page_bytes)
            
        print(f"[Zapisano podgląd PDF] -> {len(pages)} stron(y) w folderze '{PROCESSED_DIR}'")
        return images_bytes
        
    except Exception as e:
        print(f"❌ Błąd konwersji PDF: {e}")
        return[]

def clear_processed_dir(processed_dir):
    """Usuwa całą zawartość katalogu processed_dir, zostawiając sam folder."""
    for entry in os.scandir(processed_dir):
        entry_path = entry.path
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        except Exception as e:
            print(f"⚠️ Nie udało się usunąć '{entry_path}': {e}")

def process_all_photos():
    ensure_model_exists(MODEL_NAME)

    if not os.path.exists(PHOTOS_DIR):
        os.makedirs(PHOTOS_DIR)
        print(f"\n📁 Utworzono folder '{PHOTOS_DIR}'. Wrzuć tam obrazki lub pliki PDF i odpal skrypt ponownie!")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    clear_processed_dir(PROCESSED_DIR)
    print(f"🧹 Wyczyszczono folder '{PROCESSED_DIR}'.")

    valid_image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    valid_pdf_extensions = ('.pdf',)
    
    all_files = os.listdir(PHOTOS_DIR)
    image_files =[f for f in all_files if f.lower().endswith(valid_image_extensions)]
    pdf_files =[f for f in all_files if f.lower().endswith(valid_pdf_extensions)]

    image_files.sort()
    pdf_files.sort()

    if not image_files and not pdf_files:
        print(f"\n🤷 Brak plików w folderze '{PHOTOS_DIR}'.")
        return

    print(f"\n🔎 Znaleziono {len(image_files)} obrazek(ów) i {len(pdf_files)} PDF(ów).")
    print(f"🚀 Uruchamiam procesowanie na AMD GPU...\n")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        out_file.write(f"# 📄 Zbiorczy wynik OCR ({MODEL_NAME})\n\n")

        # --- PRZETWARZANIE OBRAZKÓW ---
        for img_name in image_files:
            img_path = os.path.join(PHOTOS_DIR, img_name)
            print(f"\n⏳ Analiza obrazu: {img_name}...")

            image_bytes = optimize_image_for_ai(img_path, img_name)

            try:
                response = ollama.generate(
                    model=MODEL_NAME,
                    prompt=QWEN_PROMPT,
                    images=[image_bytes]
                )

                extracted_text = response.get('response', '').strip()

                # --- NOWY, INTELIGENTNY BEZPIECZNIK ---
                if "> ⚠️ **Uwagi" in extracted_text:
                    parts = extracted_text.split("> ⚠️ **Uwagi")
                    main_text = parts[0]
                    footer = parts[1]
                    
                    # Jeśli stopka zawiera nasze angielskie tokeny sterujące (skopiowała pusty szablon)
                    # lub jest po prostu pusta, odcinamy ją.
                    is_fake_template = "DESCRIBE_POSITION_IN_POLISH" in footer or "uncertain_word" in footer or len(footer.strip()) < 5
                    
                    if is_fake_template:
                        extracted_text = main_text.strip()
                # --------------------------------------

                out_file.write(f"## 🖼️ Plik: `{img_name}`\n\n")
                out_file.write(extracted_text)
                out_file.write("\n\n---\n\n")

                print(f"✅ OCR Zakończono: {img_name}")

            except Exception as e:
                print(f"❌ Błąd modelu podczas przetwarzania {img_name}: {e}")
                out_file.write(f"## 🖼️ Plik: `{img_name}`\n\n*Błąd OCR: {e}*\n\n---\n\n")
        
        # --- PRZETWARZANIE PLIKÓW PDF ---
        for pdf_name in pdf_files:
            pdf_path = os.path.join(PHOTOS_DIR, pdf_name)
            print(f"\n⏳ Przetwarzanie PDF: {pdf_name}...")
            
            page_images = process_pdf_pages(pdf_path, pdf_name)
            
            if not page_images:
                out_file.write(f"## 📕 PDF: `{pdf_name}`\n\n*Błąd konwersji.*\n\n---\n\n")
                continue
                
            out_file.write(f"## 📕 PDF: `{pdf_name}`\n\n")
            
            for page_num, image_bytes in enumerate(page_images, 1):
                print(f"  ⏳ Analiza OCR strony {page_num}/{len(page_images)}...")
                
                try:
                    response = ollama.generate(
                        model=MODEL_NAME,
                        prompt=QWEN_PROMPT,
                        images=[image_bytes]
                    )
                    
                    extracted_text = response.get('response', '').strip()
                    
                    # --- NOWY, INTELIGENTNY BEZPIECZNIK ---
                    if "> ⚠️ **Uwagi" in extracted_text:
                        parts = extracted_text.split("> ⚠️ **Uwagi")
                        main_text = parts[0]
                        footer = parts[1]
                        
                        is_fake_template = "DESCRIBE_POSITION_IN_POLISH" in footer or "uncertain_word" in footer or len(footer.strip()) < 5
                        
                        if is_fake_template:
                            extracted_text = main_text.strip()
                    # --------------------------------------
                    
                    out_file.write(f"### Strona {page_num}\n\n")
                    out_file.write(extracted_text)
                    out_file.write("\n\n")
                    
                    print(f"  ✅ Strona {page_num} gotowa")
                    
                except Exception as e:
                    print(f"  ❌ Błąd OCR na stronie {page_num}: {e}")
                    out_file.write(f"### Strona {page_num}\n\n*Błąd OCR: {e}*\n\n")
            
            out_file.write("---\n\n")
            print(f"✅ Zakończono cały plik: {pdf_name}")

    print(f"\n🎉 Gotowe! Wszystkie wyniki zostały zapisane w pliku: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_photos()