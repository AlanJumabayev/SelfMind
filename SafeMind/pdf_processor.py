
import os
import re
import cv2
import numpy as np
import pytesseract
import fitz
import random
import traceback
from PIL import Image

class PDFProcessor:
    def __init__(self, anonymizer):
        self.anonymizer = anonymizer
        
    def is_scanned_pdf(self, pdf_document):
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            if len(page.get_text().strip()) < 50 and len(page.get_images()) > 0:
                return True
        return False
    
    def process_scanned_pdf(self, pdf_path, data_types=None, detect_faces=True):
        if data_types is None:
            data_types = self.anonymizer.patterns.keys()
            
        pdf_document = fitz.open(pdf_path)
        self.anonymizer.detected_data = {data_type: [] for data_type in data_types}
        if detect_faces:
            self.anonymizer.detected_data["Лица"] = []
            
        temp_files = []
        new_pdf = fitz.open()
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(alpha=False)
            img_path = f"temp_page_{random.randint(1000, 9999)}.png"
            temp_files.append(img_path)
            pix.save(img_path)
            
            img = cv2.imread(img_path)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang='rus+eng')
            
            ocr_data = pytesseract.image_to_data(gray, lang='rus+eng', output_type=pytesseract.Output.DICT)
            
            for data_type in data_types:
                if data_type in self.anonymizer.patterns:
                    pattern = self.anonymizer.patterns[data_type]
                    for match in re.finditer(pattern, text):
                        found_text = match.group(0)
                        self.anonymizer.detected_data[data_type].append(f"Стр. {page_num+1}: {found_text}")
                        
                        for i, word in enumerate(ocr_data['text']):
                            if word and (word in found_text or found_text in word):
                                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
            
            if detect_faces:
                processed_img, face_count = self.anonymizer.detect_and_blur_faces(img)
                if face_count > 0:
                    self.anonymizer.detected_data["Лица"].append(f"Стр. {page_num+1}: {face_count} лиц")
                    img = processed_img
            
            cv2.imwrite(img_path, img)
            
            new_page = new_pdf.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, filename=img_path)
        
        temp_path = pdf_path.replace('.pdf', '_anonymized.pdf')
        new_pdf.save(temp_path)
        new_pdf.close()
        pdf_document.close()
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
        return temp_path, "PDF обработан с использованием OCR"
    
    def anonymize_pdf(self, pdf_path, data_types=None, detect_faces=True):
        if data_types is None:
            data_types = self.anonymizer.patterns.keys()
            
        try:
            pdf_document = fitz.open(pdf_path)
            
            if self.is_scanned_pdf(pdf_document):
                pdf_document.close()
                return self.process_scanned_pdf(pdf_path, data_types, detect_faces)
            
            self.anonymizer.detected_data = {data_type: [] for data_type in data_types}
            if detect_faces:
                self.anonymizer.detected_data["Лица"] = []
            
            temp_files = []
            found_data = False
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                page_text = page.get_text("text")
                text_words = page.get_text("words")
                
                page_needs_ocr = False
                if len(page_text.strip()) < 50:
                    page_needs_ocr = True
                
                for data_type in data_types:
                    if data_type in self.anonymizer.patterns:
                        pattern = self.anonymizer.patterns[data_type]
                        
                        for match in re.finditer(pattern, page_text):
                            found_text = match.group(0)
                            self.anonymizer.detected_data[data_type].append(f"Стр. {page_num+1}: {found_text}")
                            found_data = True
                            
                            text_instances = page.search_for(found_text)
                            
                            if not text_instances:
                                for word in found_text.split():
                                    if len(word) > 3:
                                        word_instances = page.search_for(word)
                                        for inst in word_instances:
                                            text_instances.append(inst)
                            
                            for rect in text_instances:
                                try:
                                    annot = page.add_redact_annot(rect, fill=(0, 0, 0))
                                    page.apply_redactions()
                                except Exception as e:
                                    print(f"Ошибка при создании аннотации: {str(e)}")
                
                if page_needs_ocr or not found_data:
                    pix = page.get_pixmap(alpha=False)
                    img_path = f"temp_ocr_page_{random.randint(1000, 9999)}.png"
                    temp_files.append(img_path)
                    pix.save(img_path)
                    
                    img = cv2.imread(img_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        ocr_text = pytesseract.image_to_string(gray, lang='rus+eng')
                        
                        data_found_in_ocr = False
                        for data_type in data_types:
                            if data_type in self.anonymizer.patterns:
                                pattern = self.anonymizer.patterns[data_type]
                                for match in re.finditer(pattern, ocr_text):
                                    found_text = match.group(0)
                                    self.anonymizer.detected_data[data_type].append(f"Стр. {page_num+1} (OCR): {found_text}")
                                    data_found_in_ocr = True
                                    found_data = True
                                    
                                    ocr_data = pytesseract.image_to_data(gray, lang='rus+eng', output_type=pytesseract.Output.DICT)
                                    
                                    for i, word in enumerate(ocr_data['text']):
                                        if word and (word in found_text or found_text in word):
                                            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
                        
                        if data_found_in_ocr:
                            cv2.imwrite(img_path, img)
                            if page.get_images():
                                page.delete_image(page.get_images()[0][0])
                            page.insert_image(page.rect, filename=img_path)
                
                if detect_faces:
                    image_list = page.get_images(full=True)
                    
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if img is not None:
                                anonymized_img, face_count = self.anonymizer.detect_and_blur_faces(img)
                                
                                if face_count > 0:
                                    self.anonymizer.detected_data["Лица"].append(f"Стр. {page_num+1}, изобр. {img_index+1}: {face_count} лиц")
                                    found_data = True
                                    
                                    temp_img_path = f"temp_img_{random.randint(1000, 9999)}.png"
                                    temp_files.append(temp_img_path)
                                    
                                    cv2.imwrite(temp_img_path, anonymized_img)
                                    
                                    rect = None
                                    if page.get_image_rects(xref):
                                        rect = page.get_image_rects(xref)[0]
                                    
                                    if rect:
                                        page.delete_image(xref)
                                        page.insert_image(rect, filename=temp_img_path)
                        except Exception as e:
                            continue
            
            if not found_data:
                pdf_document.close()
                return self.process_scanned_pdf(pdf_path, data_types, detect_faces)
            
            for key in list(self.anonymizer.detected_data.keys()):
                if not self.anonymizer.detected_data[key]:
                    del self.anonymizer.detected_data[key]
            
            temp_path = pdf_path.replace('.pdf', '_anonymized.pdf')
            pdf_document.save(temp_path)
            pdf_document.close()
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            
            return temp_path, "PDF обработан"
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Ошибка при обработке PDF: {str(e)}\n{traceback_str}")
            
            return None, f"Ошибка при обработке PDF: {str(e)}"
            
    def extract_text_from_pdf(self, pdf_path):
        try:
            pdf_document = fitz.open(pdf_path)
            text = ""
            
            is_scanned = self.is_scanned_pdf(pdf_document)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                text += f"\n--- Страница {page_num+1} ---\n"
                
                page_text = page.get_text("text").strip()
                
                if len(page_text) < 50 or is_scanned:
                    pix = page.get_pixmap(alpha=False)
                    img_path = f"temp_ocr_page_{random.randint(1000, 9999)}.png"
                    pix.save(img_path)
                    
                    img = cv2.imread(img_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        ocr_text = pytesseract.image_to_string(gray, lang='rus+eng')
                        page_text = ocr_text
                    
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except:
                            pass
                
                text += page_text + "\n"
                
            pdf_document.close()
            return text, "Текст успешно извлечен"
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Ошибка при извлечении текста: {str(e)}\n{traceback_str}")
            return None, f"Ошибка при извлечении текста: {str(e)}"
            
    def count_pages(self, pdf_path):
        try:
            pdf_document = fitz.open(pdf_path)
            count = len(pdf_document)
            pdf_document.close()
            return count
        except Exception:
            return 0
            
    def extract_images(self, pdf_path, output_dir="extracted_images"):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            pdf_document = fitz.open(pdf_path)
            image_count = 0
            extracted_paths = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        image_filename = f"{output_dir}/image_p{page_num+1}_{img_index+1}.{image_ext}"
                        extracted_paths.append(image_filename)
                        
                        with open(image_filename, "wb") as img_file:
                            img_file.write(image_bytes)
                            image_count += 1
                    except Exception as e:
                        print(f"Ошибка при извлечении изображения {img_index} на странице {page_num}: {str(e)}")
                        continue
            
            pdf_document.close()
            return extracted_paths, f"Извлечено {image_count} изображений"
        except Exception as e:
            return None, f"Ошибка при извлечении изображений: {str(e)}"
            
    def add_watermark(self, pdf_path, watermark_text="КОНФИДЕНЦИАЛЬНО"):
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                rect = page.rect
                
                font_size = min(rect.width, rect.height) / 10
                
                page.insert_text(
                    fitz.Point(rect.width/2, rect.height/2),
                    watermark_text,
                    fontsize=font_size,
                    color=(0.8, 0.0, 0.0, 0.5),
                    rotate=45
                )
            
            watermarked_path = pdf_path.replace('.pdf', '_watermarked.pdf')
            pdf_document.save(watermarked_path)
            pdf_document.close()
            
            return watermarked_path, "Водяной знак добавлен"
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Ошибка при добавлении водяного знака: {str(e)}\n{traceback_str}")
            return None, f"Ошибка при добавлении водяного знака: {str(e)}"
            
    def split_pdf(self, pdf_path, start_page, end_page):
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            if start_page < 1 or start_page > total_pages:
                return None, f"Некорректный номер начальной страницы. Допустимый диапазон: 1-{total_pages}"
                
            if end_page < start_page or end_page > total_pages:
                return None, f"Некорректный номер конечной страницы. Допустимый диапазон: {start_page}-{total_pages}"
                
            new_pdf = fitz.open()
            
            for page_num in range(start_page - 1, end_page):
                new_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
            
            split_path = pdf_path.replace('.pdf', f'_p{start_page}-p{end_page}.pdf')
            new_pdf.save(split_path)
            new_pdf.close()
            pdf_document.close()
            
            return split_path, f"PDF разделен, создан файл со страницами {start_page}-{end_page}"
        except Exception as e:
            return None, f"Ошибка при разделении PDF: {str(e)}"
            
    def merge_pdfs(self, pdf_paths, output_path=None):
        try:
            if not pdf_paths:
                return None, "Не указаны файлы для объединения"
                
            merged_pdf = fitz.open()
            
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path):
                    try:
                        pdf_document = fitz.open(pdf_path)
                        merged_pdf.insert_pdf(pdf_document)
                        pdf_document.close()
                    except Exception as e:
                        print(f"Ошибка при добавлении файла {pdf_path}: {str(e)}")
                        continue
                        
            if not output_path:
                if len(pdf_paths) > 0:
                    base_dir = os.path.dirname(pdf_paths[0])
                    output_path = os.path.join(base_dir, "merged_document.pdf")
                else:
                    output_path = "merged_document.pdf"
            
            merged_pdf.save(output_path)
            merged_pdf.close()
            
            return output_path, f"PDF файлы объединены в {output_path}"
        except Exception as e:
            return None, f"Ошибка при объединении PDF: {str(e)}"