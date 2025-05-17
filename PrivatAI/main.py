import sys
import os
import re
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import random
import string
from PIL import Image
import traceback

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog, 
                           QMessageBox, QProgressBar, QComboBox, QCheckBox, QGridLayout,
                           QLineEdit, QListWidget, QListWidgetItem, QSplitter, QFrame,
                           QAbstractItemDelegate)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QTextDocument
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QBuffer, QIODevice, QSize

try:
    from chatbot import ChatBot
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AnonymizerEngine:
    def __init__(self):
        # Расширенные регулярные выражения для лучшего обнаружения данных
        self.patterns = {
            'ИИН': r'\b\d{12}\b|\b\d{3}[\s.-]?\d{3}[\s.-]?\d{6}\b|\b\d{6}[\s.-]?\d{6}\b',
            'Телефон': r'\+?[7]\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}|\b\d{10}\b|\b\d{11}\b|\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}',
            'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'Банковская карта': r'\b(?:\d{4}[\s-]?){4}\b|\b\d{16}\b',
            'ФИО': r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b|\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
            'Банковский счет': r'\b\d{20}\b|\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b',
            'IBAN': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}[0-9]{16}\b|\b[A-Z]{2}[\s-]?\d{2}[\s-]?[A-Z0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b',
            'Дата рождения': r'\b(0[1-9]|[12][0-9]|3[01])[./-](0[1-9]|1[0-2])[./-](19|20)\d\d\b|\b(19|20)\d\d[./-](0[1-9]|1[0-2])[./-](0[1-9]|[12][0-9]|3[01])\b',
            'Возраст': r'\b(возраст|лет|год(?:а)?|годов)[\s:]*\d{1,3}\b|\b\d{1,3}\s*(?:лет|год(?:а)?|годов)\b'
        }
        
        self.detected_data = {}
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.face_detection_available = True
        except Exception:
            self.face_detection_available = False
        
    def anonymize_text(self, text, data_types=None):
        if data_types is None:
            data_types = self.patterns.keys()
            
        self.detected_data = {data_type: [] for data_type in data_types}
        
        anonymized_text = text
        for data_type in data_types:
            if data_type in self.patterns:
                pattern = self.patterns[data_type]
                
                matches = re.finditer(pattern, anonymized_text)
                for match in matches:
                    found_text = match.group(0)
                    start, end = match.span()
                    
                    self.detected_data[data_type].append(found_text)
                    
                    if data_type == 'ИИН':
                        mask = 'XXX-XXX-XXXX'
                    elif data_type == 'Телефон':
                        mask = '+X-XXX-XXX-XXXX'
                    elif data_type == 'Email':
                        if '@' in found_text:
                            username, domain = found_text.split('@')
                            mask = username[0] + 'X' * (len(username) - 2) + username[-1] + '@' + domain
                        else:
                            mask = 'X' * len(found_text)
                    elif data_type == 'Банковская карта':
                        mask = 'XXXX-XXXX-XXXX-' + found_text[-4:]
                    elif data_type == 'Банковский счет':
                        mask = 'XXXX-XXXX-XXXX-XXXX-' + found_text[-4:]
                    elif data_type == 'IBAN':
                        mask = found_text[:2] + 'XX-XXXX-XXXX-XXXX-' + found_text[-4:]
                    elif data_type == 'ФИО':
                        parts = found_text.split()
                        mask = parts[0][0] + 'X' * (len(parts[0]) - 1)
                        for part in parts[1:]:
                            mask += ' ' + part[0] + 'X' * (len(part) - 1)
                    elif data_type == 'Дата рождения':
                        mask = 'XX.XX.XXXX'
                    elif data_type == 'Возраст':
                        mask = 'XX лет'
                    else:
                        mask = 'X' * len(found_text)
                        
                    anonymized_text = anonymized_text[:start] + mask + anonymized_text[end:]
        
        return anonymized_text
    
    def detect_and_blur_faces(self, image, blur_factor=35):
        if not self.face_detection_available:
            return image, 0
            
        try:
            if image is None or len(image.shape) < 2:
                return image, 0
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            face_count = len(faces)
            
            if face_count > 0:
                for (x, y, w, h) in faces:
                    try:
                        x, y = max(0, x), max(0, y)
                        w = min(w, image.shape[1] - x)
                        h = min(h, image.shape[0] - y)
                        
                        if w > 0 and h > 0:
                            face_roi = image[y:y+h, x:x+w]
                            
                            blurred_face = cv2.GaussianBlur(face_roi, (blur_factor, blur_factor), 0)
                            
                            image[y:y+h, x:x+w] = blurred_face
                    except Exception:
                        continue
            
            return image, face_count
        except Exception:
            return image, 0
    
    def anonymize_image(self, image_path, data_types=None, detect_faces=True):
        if data_types is None:
            data_types = self.patterns.keys()
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            if image is None:
                return None, "Не удалось загрузить изображение"
        except Exception as e:
            return None, f"Ошибка при загрузке изображения: {str(e)}"
            
        anonymized_image = image.copy()
        
        self.detected_data = {data_type: [] for data_type in data_types}
        if detect_faces:
            self.detected_data["Лица"] = []
        
        face_count = 0
        if detect_faces:
            try:
                anonymized_image, face_count = self.detect_and_blur_faces(anonymized_image)
                if face_count > 0:
                    self.detected_data["Лица"].append(f"Обнаружено лиц: {face_count}")
            except Exception:
                pass
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text_data = pytesseract.image_to_data(gray, lang='rus+eng', output_type=pytesseract.Output.DICT)
            
            full_text = " ".join(text_data['text'])
            
            for data_type in data_types:
                if data_type in self.patterns:
                    pattern = self.patterns[data_type]
                    
                    for match in re.finditer(pattern, full_text):
                        found_text = match.group(0)
                        self.detected_data[data_type].append(found_text)
                        
                        for i, word in enumerate(text_data['text']):
                            if word and word in found_text:
                                x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
                                cv2.rectangle(anonymized_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
        except Exception:
            pass
        
        for key in list(self.detected_data.keys()):
            if not self.detected_data[key]:
                del self.detected_data[key]
                
        return anonymized_image, "Изображение обработано"
    
    # Новая функция для проверки, является ли PDF сканированным документом
    def is_scanned_pdf(self, pdf_document):
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            if len(page.get_text().strip()) < 50 and len(page.get_images()) > 0:
                return True
        return False
    
    # Новая функция для OCR обработки сканированных PDF
    def process_scanned_pdf(self, pdf_path, data_types=None, detect_faces=True):
        if data_types is None:
            data_types = self.patterns.keys()
            
        pdf_document = fitz.open(pdf_path)
        self.detected_data = {data_type: [] for data_type in data_types}
        if detect_faces:
            self.detected_data["Лица"] = []
            
        temp_files = []
        new_pdf = fitz.open()
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(alpha=False)
            img_path = f"temp_page_{random.randint(1000, 9999)}.png"
            temp_files.append(img_path)
            pix.save(img_path)
            
            # OCR обработка страницы
            img = cv2.imread(img_path)
            
            # Обработка текста через OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang='rus+eng')
            
            # Поиск персональных данных в тексте
            for data_type in data_types:
                if data_type in self.patterns:
                    pattern = self.patterns[data_type]
                    for match in re.finditer(pattern, text):
                        found_text = match.group(0)
                        self.detected_data[data_type].append(f"Стр. {page_num+1}: {found_text}")
            
            # Обработка и размытие лиц
            if detect_faces:
                processed_img, face_count = self.detect_and_blur_faces(img)
                if face_count > 0:
                    self.detected_data["Лица"].append(f"Стр. {page_num+1}: {face_count} лиц")
                    cv2.imwrite(img_path, processed_img)
            
            # Создаем новую страницу
            new_page = new_pdf.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, filename=img_path)
        
        # Сохраняем измененный PDF
        temp_path = pdf_path.replace('.pdf', '_anonymized.pdf')
        new_pdf.save(temp_path)
        new_pdf.close()
        pdf_document.close()
        
        # Удаляем временные файлы
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
        return temp_path, "PDF обработан с использованием OCR"
    
    # Улучшенная функция анонимизации PDF
    def anonymize_pdf(self, pdf_path, data_types=None, detect_faces=True):
        if data_types is None:
            data_types = self.patterns.keys()
            
        try:
            # Открываем PDF-файл
            pdf_document = fitz.open(pdf_path)
            
            # Проверяем, является ли PDF сканированным документом
            if self.is_scanned_pdf(pdf_document):
                pdf_document.close()
                return self.process_scanned_pdf(pdf_path, data_types, detect_faces)
            
            # Инициализируем словарь для отчета
            self.detected_data = {data_type: [] for data_type in data_types}
            if detect_faces:
                self.detected_data["Лица"] = []
            
            # Для хранения путей временных файлов
            temp_files = []
            
            # Обрабатываем каждую страницу
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Различные подходы к извлечению текста
                text_words = page.get_text("words")  # Получаем список слов с координатами
                text_blocks = page.get_text("blocks")  # Получаем блоки текста
                text_whole = page.get_text("text")  # Весь текст страницы
                
                # Ищем персональные данные во всем тексте страницы
                for data_type in data_types:
                    if data_type in self.patterns:
                        pattern = self.patterns[data_type]
                        
                        # Поиск в целом тексте
                        for match in re.finditer(pattern, text_whole):
                            found_text = match.group(0)
                            self.detected_data[data_type].append(f"Стр. {page_num+1}: {found_text}")
                            
                            # Создаем редакцию для каждого совпадения
                            instances = []
                            
                            # Ищем в словах точные совпадения или части
                            for word_info in text_words:
                                word = word_info[4]
                                if word in found_text or found_text in word:
                                    rect = fitz.Rect(word_info[:4])
                                    instances.append(rect)
                            
                            # Если нашли экземпляры слова на странице
                            if instances:
                                # Создаем аннотации редактирования для каждого экземпляра
                                for rect in instances:
                                    annot = page.add_redact_annot(rect, fill=(0, 0, 0))
                                    if data_type == 'ИИН':
                                        annot.set_info(content="XXX-XXX-XXXX")
                                    elif data_type == 'Телефон':
                                        annot.set_info(content="+X-XXX-XXX-XXXX")
                                    elif data_type == 'Email':
                                        parts = found_text.split('@')
                                        if len(parts) > 1:
                                            username, domain = parts
                                            mask = username[0] + "X" * (len(username) - 2) + username[-1] + "@" + domain
                                            annot.set_info(content=mask)
                                    elif data_type == 'ФИО':
                                        parts = found_text.split()
                                        mask = ""
                                        for i, part in enumerate(parts):
                                            if i > 0:
                                                mask += " "
                                            mask += part[0] + "X" * (len(part) - 1)
                                        annot.set_info(content=mask)
                                    else:
                                        annot.set_info(content="X" * len(found_text))
                
                # Применяем редактирование
                page.apply_redactions()
                
                # Обработка изображений на странице, если включена опция обнаружения лиц
                if detect_faces:
                    try:
                        # Получаем список изображений на странице
                        image_list = page.get_images(full=True)
                        
                        # Обрабатываем каждое изображение
                        for img_index, img_info in enumerate(image_list):
                            try:
                                # Извлекаем изображение
                                xref = img_info[0]
                                base_image = pdf_document.extract_image(xref)
                                image_bytes = base_image["image"]
                                
                                # Конвертируем bytes в изображение
                                nparr = np.frombuffer(image_bytes, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                # Если изображение удалось загрузить
                                if img is not None:
                                    # Размываем лица на изображении
                                    anonymized_img, face_count = self.detect_and_blur_faces(img)
                                    
                                    # Если найдены лица
                                    if face_count > 0:
                                        # Добавляем информацию в отчет
                                        self.detected_data["Лица"].append(f"Стр. {page_num+1}, изобр. {img_index+1}: {face_count} лиц")
                                        
                                        # Создаем имя временного файла
                                        temp_img_path = f"temp_img_{random.randint(1000, 9999)}.png"
                                        temp_files.append(temp_img_path)
                                        
                                        # Сохраняем анонимизированное изображение
                                        cv2.imwrite(temp_img_path, anonymized_img)
                                        
                                        # Получаем прямоугольник изображения на странице
                                        rects = page.get_image_rects(xref)
                                        if rects:
                                            rect = rects[0]
                                            
                                            # Удаляем оригинальное изображение
                                            page.delete_image(xref)
                                            
                                            # Вставляем анонимизированное изображение
                                            page.insert_image(rect, filename=temp_img_path)
                            except Exception as e:
                                print(f"Ошибка при обработке изображения {img_index} на странице {page_num}: {str(e)}")
                                continue
                    except Exception as e:
                        print(f"Ошибка при получении списка изображений на странице {page_num}: {str(e)}")
                        continue
            
            # Проверка на пустой отчет - возможно нужен OCR
            if all(not items for items in self.detected_data.values()):
                # Если не нашли данные обычным способом, пробуем через OCR
                pdf_document.close()
                return self.process_scanned_pdf(pdf_path, data_types, detect_faces)
            
            # Если нет обнаруженных лиц, удаляем ключ из отчета
            if "Лица" in self.detected_data and not self.detected_data["Лица"]:
                del self.detected_data["Лица"]
            
            # Удаляем пустые разделы из отчета
            for key in list(self.detected_data.keys()):
                if not self.detected_data[key]:
                    del self.detected_data[key]
            
            # Сохраняем изменения во временный файл
            temp_path = pdf_path.replace('.pdf', '_anonymized.pdf')
            pdf_document.save(temp_path)
            pdf_document.close()
            
            # Удаляем временные файлы
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            
            return temp_path, "PDF обработан"
        except Exception as e:
            # Выводим полную информацию об ошибке
            traceback_str = traceback.format_exc()
            print(f"Ошибка при обработке PDF: {str(e)}\n{traceback_str}")
            
            # Возвращаем информацию об ошибке
            return None, f"Ошибка при обработке PDF: {str(e)}"
    
    def get_report(self):
        report = "Отчет об обнаруженных персональных данных:\n\n"
        
        has_data = False
        for data_type, items in self.detected_data.items():
            if items:
                has_data = True
                report += f"- {data_type}:\n"
                for item in items:
                    report += f"  • {item}\n"
                report += "\n"
                
        if not has_data:
            report += "Персональные данные не обнаружены."
            
        return report

class WorkerThread(QThread):
    finished = pyqtSignal(object, str)
    progress = pyqtSignal(int)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        for i in range(101):
            self.progress.emit(i)
            self.msleep(10)
            
        try:
            result, message = self.func(*self.args, **self.kwargs)
            self.finished.emit(result, message)
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Ошибка в рабочем потоке: {str(e)}\n{traceback_str}")
            self.finished.emit(None, f"Ошибка: {str(e)}")

class ChatItemDelegate(QAbstractItemDelegate):
    def __init__(self):
        super().__init__()
        
    def paint(self, painter, option, index):
        data = index.data(Qt.UserRole)
        if not data:
            return
            
        text = data["text"]
        is_user = data["is_user"]
        
        user_color = QColor(220, 240, 255)
        bot_color = QColor(240, 240, 240)
        text_color = QColor(0, 0, 0)
        
        rect = option.rect
        
        background_color = user_color if is_user else bot_color
        painter.fillRect(rect, background_color)
        
        painter.setPen(QColor(200, 200, 200))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        text_rect = rect.adjusted(10, 5, -10, -5)
        painter.setPen(text_color)
        painter.setFont(QFont("Arial", 9))
        
        header = "Вы:" if is_user else "Бот:"
        header_font = QFont("Arial", 9, QFont.Bold)
        painter.setFont(header_font)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, header)
        
        text_rect = text_rect.adjusted(0, 20, 0, 0)
        painter.setFont(QFont("Arial", 9))
        
        document = QTextDocument()
        document.setDefaultFont(QFont("Arial", 9))
        document.setTextWidth(text_rect.width())
        document.setHtml(text.replace("\n", "<br>"))
        
        if index.model():
            height = document.size().height() + 30
            index.model().setData(index, QSize(int(text_rect.width()), int(height)), Qt.SizeHintRole)
        
        painter.save()
        painter.translate(text_rect.topLeft())
        document.drawContents(painter)
        painter.restore()
    
    def sizeHint(self, option, index):
        size = index.data(Qt.SizeHintRole)
        if size:
            return size
        return QSize(option.rect.width(), 50)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.anonymizer = AnonymizerEngine()
        
        if CHATBOT_AVAILABLE:
            self.chatbot = ChatBot(self.anonymizer)
        else:
            self.chatbot = None
            
        self.current_file_path = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("SafeMind - Анонимизация данных")
        self.setGeometry(100, 100, 1000, 750)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        
        self.text_tab = QWidget()
        self.setup_text_tab()
        self.tabs.addTab(self.text_tab, "Текст")
        
        self.image_tab = QWidget()
        self.setup_image_tab()
        self.tabs.addTab(self.image_tab, "Изображения")
        
        self.pdf_tab = QWidget()
        self.setup_pdf_tab()
        self.tabs.addTab(self.pdf_tab, "PDF-файлы")
        
        self.report_tab = QWidget()
        self.setup_report_tab()
        self.tabs.addTab(self.report_tab, "Отчет")
        
        if self.chatbot:
            self.chat_tab = QWidget()
            self.setup_chat_tab()
            self.tabs.addTab(self.chat_tab, "Чат-бот")
        
        main_layout.addWidget(self.tabs)
        
        self.statusBar().showMessage('Готов к работе')
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
    def setup_text_tab(self):
        layout = QVBoxLayout(self.text_tab)
        
        input_label = QLabel("Введите текст для анонимизации:")
        layout.addWidget(input_label)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Вставьте текст, содержащий персональные данные...")
        layout.addWidget(self.text_input)
        
        options_layout = QHBoxLayout()
        
        self.text_options = {}
        options_widget = QWidget()
        options_grid = QGridLayout(options_widget)
        
        row, col = 0, 0
        for data_type in self.anonymizer.patterns.keys():
            self.text_options[data_type] = QCheckBox(data_type)
            self.text_options[data_type].setChecked(True)
            options_grid.addWidget(self.text_options[data_type], row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
                
        options_layout.addWidget(options_widget)
        layout.addLayout(options_layout)
        
        buttons_layout = QHBoxLayout()
        
        self.anonymize_text_button = QPushButton("Анонимизировать")
        self.anonymize_text_button.clicked.connect(self.on_anonymize_text)
        buttons_layout.addWidget(self.anonymize_text_button)
        
        self.hide_personal_info_button = QPushButton("Скрыть личную информацию")
        self.hide_personal_info_button.clicked.connect(self.on_hide_personal_info_text)
        buttons_layout.addWidget(self.hide_personal_info_button)
        
        self.clear_text_button = QPushButton("Очистить")
        self.clear_text_button.clicked.connect(self.on_clear_text_tab)
        buttons_layout.addWidget(self.clear_text_button)
        
        layout.addLayout(buttons_layout)
        
        result_label = QLabel("Результат:")
        layout.addWidget(result_label)
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        layout.addWidget(self.text_output)
        
        self.save_text_button = QPushButton("Сохранить результат")
        self.save_text_button.clicked.connect(self.on_save_text)
        self.save_text_button.setEnabled(False)
        layout.addWidget(self.save_text_button)
        
    def setup_image_tab(self):
        layout = QVBoxLayout(self.image_tab)
        
        input_layout = QHBoxLayout()
        
        input_panel = QVBoxLayout()
        input_label = QLabel("Исходное изображение:")
        input_panel.addWidget(input_label)
        
        self.image_input_label = QLabel()
        self.image_input_label.setAlignment(Qt.AlignCenter)
        self.image_input_label.setMinimumSize(400, 300)
        self.image_input_label.setStyleSheet("border: 1px solid #cccccc;")
        input_panel.addWidget(self.image_input_label)
        
        self.load_image_button = QPushButton("Загрузить изображение")
        self.load_image_button.clicked.connect(self.on_load_image)
        input_panel.addWidget(self.load_image_button)
        
        input_layout.addLayout(input_panel)
        
        output_panel = QVBoxLayout()
        output_label = QLabel("Анонимизированное изображение:")
        output_panel.addWidget(output_label)
        
        self.image_output_label = QLabel()
        self.image_output_label.setAlignment(Qt.AlignCenter)
        self.image_output_label.setMinimumSize(400, 300)
        self.image_output_label.setStyleSheet("border: 1px solid #cccccc;")
        output_panel.addWidget(self.image_output_label)
        
        self.save_image_button = QPushButton("Сохранить изображение")
        self.save_image_button.clicked.connect(self.on_save_image)
        self.save_image_button.setEnabled(False)
        output_panel.addWidget(self.save_image_button)
        
        input_layout.addLayout(output_panel)
        layout.addLayout(input_layout)
        
        options_layout = QHBoxLayout()
        
        self.image_options = {}
        options_widget = QWidget()
        options_grid = QGridLayout(options_widget)
        
        row, col = 0, 0
        for data_type in self.anonymizer.patterns.keys():
            self.image_options[data_type] = QCheckBox(data_type)
            self.image_options[data_type].setChecked(True)
            options_grid.addWidget(self.image_options[data_type], row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        self.detect_faces_checkbox = QCheckBox("Обнаружение и размытие лиц")
        self.detect_faces_checkbox.setChecked(True)
        options_grid.addWidget(self.detect_faces_checkbox, row + 1, 0, 1, 3)
                
        options_layout.addWidget(options_widget)
        
        buttons_layout = QHBoxLayout()
        
        self.anonymize_image_button = QPushButton("Анонимизировать")
        self.anonymize_image_button.clicked.connect(self.on_anonymize_image)
        self.anonymize_image_button.setEnabled(False)
        buttons_layout.addWidget(self.anonymize_image_button)
        
        self.hide_personal_info_button_image = QPushButton("Скрыть личную информацию")
        self.hide_personal_info_button_image.clicked.connect(self.on_hide_personal_info_image)
        self.hide_personal_info_button_image.setEnabled(False)
        buttons_layout.addWidget(self.hide_personal_info_button_image)
        
        self.clear_image_button = QPushButton("Очистить")
        self.clear_image_button.clicked.connect(self.on_clear_image_tab)
        buttons_layout.addWidget(self.clear_image_button)
        
        options_layout.addLayout(buttons_layout)
        
        layout.addLayout(options_layout)
        
    def setup_pdf_tab(self):
        layout = QVBoxLayout(self.pdf_tab)
        
        self.pdf_info_label = QLabel("Загрузите PDF-файл для анонимизации.")
        layout.addWidget(self.pdf_info_label)
        
        pdf_buttons_layout = QHBoxLayout()
        
        self.load_pdf_button = QPushButton("Загрузить PDF")
        self.load_pdf_button.clicked.connect(self.on_load_pdf)
        pdf_buttons_layout.addWidget(self.load_pdf_button)
        
        self.current_pdf_path = None
        
        layout.addLayout(pdf_buttons_layout)
        
        options_layout = QHBoxLayout()
        
        self.pdf_options = {}
        options_widget = QWidget()
        options_grid = QGridLayout(options_widget)
        
        row, col = 0, 0
        for data_type in self.anonymizer.patterns.keys():
            self.pdf_options[data_type] = QCheckBox(data_type)
            self.pdf_options[data_type].setChecked(True)
            options_grid.addWidget(self.pdf_options[data_type], row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        self.pdf_detect_faces_checkbox = QCheckBox("Обнаружение и размытие лиц")
        self.pdf_detect_faces_checkbox.setChecked(True)
        options_grid.addWidget(self.pdf_detect_faces_checkbox, row + 1, 0, 1, 3)
                
        options_layout.addWidget(options_widget)
        
        buttons_layout = QHBoxLayout()
        
        self.anonymize_pdf_button = QPushButton("Анонимизировать PDF")
        self.anonymize_pdf_button.clicked.connect(self.on_anonymize_pdf)
        self.anonymize_pdf_button.setEnabled(False)
        buttons_layout.addWidget(self.anonymize_pdf_button)
        
        self.hide_personal_info_button_pdf = QPushButton("Скрыть личную информацию")
        self.hide_personal_info_button_pdf.clicked.connect(self.on_hide_personal_info_pdf)
        self.hide_personal_info_button_pdf.setEnabled(False)
        buttons_layout.addWidget(self.hide_personal_info_button_pdf)
        
        self.clear_pdf_button = QPushButton("Очистить")
        self.clear_pdf_button.clicked.connect(self.on_clear_pdf_tab)
        buttons_layout.addWidget(self.clear_pdf_button)
        
        options_layout.addLayout(buttons_layout)
        
        layout.addLayout(options_layout)
        
        preview_label = QLabel("Предпросмотр первой страницы:")
        layout.addWidget(preview_label)
        
        self.pdf_preview_label = QLabel()
        self.pdf_preview_label.setAlignment(Qt.AlignCenter)
        self.pdf_preview_label.setMinimumSize(800, 400)
        self.pdf_preview_label.setStyleSheet("border: 1px solid #cccccc;")
        layout.addWidget(self.pdf_preview_label)
        
        self.save_pdf_button = QPushButton("Сохранить анонимизированный PDF")
        self.save_pdf_button.clicked.connect(self.on_save_pdf)
        self.save_pdf_button.setEnabled(False)
        layout.addWidget(self.save_pdf_button)
        
    def setup_report_tab(self):
        layout = QVBoxLayout(self.report_tab)
        
        report_header = QLabel("Отчет о найденных персональных данных")
        report_header.setFont(QFont("Arial", 14, QFont.Bold))
        report_header.setAlignment(Qt.AlignCenter)
        layout.addWidget(report_header)
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Здесь будет отображен отчет после анонимизации...")
        layout.addWidget(self.report_text)
        
        buttons_layout = QHBoxLayout()
        
        self.save_report_button = QPushButton("Сохранить отчет")
        self.save_report_button.clicked.connect(self.on_save_report)
        buttons_layout.addWidget(self.save_report_button)
        
        self.clear_report_button = QPushButton("Очистить")
        self.clear_report_button.clicked.connect(self.on_clear_report_tab)
        buttons_layout.addWidget(self.clear_report_button)
        
        layout.addLayout(buttons_layout)
    
    def setup_chat_tab(self):
        layout = QVBoxLayout(self.chat_tab)
        
        chat_header = QLabel("SafeMind Чат-бот")
        chat_header.setFont(QFont("Arial", 14, QFont.Bold))
        chat_header.setAlignment(Qt.AlignCenter)
        layout.addWidget(chat_header)
        
        chat_desc = QLabel("Задайте вопрос или введите команду для анонимизации данных")
        chat_desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(chat_desc)
        
        chat_area = QSplitter(Qt.Vertical)
        
        self.chat_history = QListWidget()
        self.chat_history.setWordWrap(True)
        self.chat_history.setStyleSheet("""
            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                background-color: #f9f9f9;
            }
            QListWidget::item {
                padding: 8px;
                margin-bottom: 2px;
            }
        """)
        
        self.chat_history.setItemDelegate(ChatItemDelegate())
        chat_area.addWidget(self.chat_history)
        
        input_layout = QHBoxLayout()
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Введите сообщение здесь...")
        self.chat_input.returnPressed.connect(self.on_send_message)
        input_layout.addWidget(self.chat_input)
        
        self.send_button = QPushButton("Отправить")
        self.send_button.clicked.connect(self.on_send_message)
        input_layout.addWidget(self.send_button)
        
        input_widget = QWidget()
        input_widget.setLayout(input_layout)
        chat_area.addWidget(input_widget)
        
        chat_area.setSizes([400, 50])
        
        layout.addWidget(chat_area)
        
        commands_layout = QHBoxLayout()
        
        quick_commands_label = QLabel("Быстрые команды:")
        commands_layout.addWidget(quick_commands_label)
        
        self.clear_chat_button = QPushButton("Очистить")
        self.clear_chat_button.clicked.connect(self.on_clear_chat_tab)
        commands_layout.addWidget(self.clear_chat_button)
        
        layout.addLayout(commands_layout)
        
        quick_commands_layout = QHBoxLayout()
        
        commands = ["Помощь", "Анонимизировать текст", "Размыть лица", "Функции"]
        for command in commands:
            cmd_button = QPushButton(command)
            cmd_button.clicked.connect(lambda checked, cmd=command: self.add_quick_command(cmd))
            quick_commands_layout.addWidget(cmd_button)
        
        layout.addLayout(quick_commands_layout)
        
        self.add_bot_message(self.chatbot.greet("", None))
    
    # ---- Методы очистки вкладок ----
    
    def on_clear_text_tab(self):
        self.text_input.clear()
        self.text_output.clear()
        self.save_text_button.setEnabled(False)
        self.statusBar().showMessage('Вкладка текста очищена')
    
    def on_clear_image_tab(self):
        self.image_input_label.clear()
        self.image_output_label.clear()
        if hasattr(self, 'image_input_path'):
            delattr(self, 'image_input_path')
        if hasattr(self, 'anonymized_image'):
            delattr(self, 'anonymized_image')
        self.anonymize_image_button.setEnabled(False)
        self.hide_personal_info_button_image.setEnabled(False)
        self.save_image_button.setEnabled(False)
        self.statusBar().showMessage('Вкладка изображений очищена')
    
    def on_clear_pdf_tab(self):
        self.pdf_preview_label.clear()
        self.pdf_info_label.setText("Загрузите PDF-файл для анонимизации.")
        self.current_pdf_path = None
        if hasattr(self, 'anonymized_pdf_path'):
            delattr(self, 'anonymized_pdf_path')
        self.anonymize_pdf_button.setEnabled(False)
        self.hide_personal_info_button_pdf.setEnabled(False)
        self.save_pdf_button.setEnabled(False)
        self.statusBar().showMessage('Вкладка PDF очищена')
    
    def on_clear_report_tab(self):
        self.report_text.clear()
        self.statusBar().showMessage('Отчет очищен')
    
    def on_clear_chat_tab(self):
        if not hasattr(self, 'chat_history') or not hasattr(self, 'chatbot'):
            return
            
        self.chat_history.clear()
        self.chat_input.clear()
        self.add_bot_message(self.chatbot.greet("", None))
        self.statusBar().showMessage('Чат очищен')
        
    # ---- Обработчики чат-бота ----
    
    def on_send_message(self):
        if not hasattr(self, 'chat_input') or not hasattr(self, 'chatbot'):
            return
            
        message = self.chat_input.text().strip()
        if not message:
            return
        
        self.add_user_message(message)
        
        response = self.chatbot.process_message(message, self.current_file_path)
        
        self.add_bot_message(response)
        
        self.chat_input.clear()
        
    def add_user_message(self, message):
        if not hasattr(self, 'chat_history'):
            return
            
        item = QListWidgetItem()
        item.setData(Qt.UserRole, {"text": message, "is_user": True})
        item.setSizeHint(QSize(self.chat_history.width(), 50))
        self.chat_history.addItem(item)
        self.chat_history.scrollToBottom()
        
    def add_bot_message(self, message):
        if not hasattr(self, 'chat_history'):
            return
            
        item = QListWidgetItem()
        item.setData(Qt.UserRole, {"text": message, "is_user": False})
        item.setSizeHint(QSize(self.chat_history.width(), 100))
        self.chat_history.addItem(item)
        self.chat_history.scrollToBottom()
        
    def add_quick_command(self, command):
        if not hasattr(self, 'chat_input'):
            return
            
        self.chat_input.setText(command)
        self.on_send_message()
        
    # ---- Обработчики событий ----
    
    def on_anonymize_text(self):
        input_text = self.text_input.toPlainText()
        if not input_text:
            QMessageBox.warning(self, "Предупреждение", "Введите текст для анонимизации")
            return
            
        selected_types = [data_type for data_type, checkbox in self.text_options.items() if checkbox.isChecked()]
        
        self.current_file_path = "текст"
        
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            lambda: (self.anonymizer.anonymize_text(input_text, selected_types), "Текст обработан")
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_text_processed)
        self.worker.start()
        
    def on_hide_personal_info_text(self):
        input_text = self.text_input.toPlainText()
        if not input_text:
            QMessageBox.warning(self, "Предупреждение", "Введите текст для анонимизации")
            return
            
        # Выбираем только личные данные - все кроме банковских счетов, карт и IBAN
        personal_data_types = ['ИИН', 'Телефон', 'Email', 'ФИО', 'Дата рождения', 'Возраст']
        selected_types = [data_type for data_type in personal_data_types if data_type in self.text_options and self.text_options[data_type].isChecked()]
        
        self.current_file_path = "текст"
        
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            lambda: (self.anonymizer.anonymize_text(input_text, selected_types), "Личная информация скрыта")
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_text_processed)
        self.worker.start()
        
    def on_text_processed(self, result, message):
        self.progress_bar.setVisible(False)
        
        if result:
            self.text_output.setText(result)
            self.save_text_button.setEnabled(True)
            self.statusBar().showMessage(message)
            
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)
        else:
            QMessageBox.warning(self, "Ошибка", message)
    
    def on_save_text(self):
        anonymized_text = self.text_output.toPlainText()
        if not anonymized_text:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить анонимизированный текст", "", "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(anonymized_text)
                self.statusBar().showMessage(f"Текст сохранен в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", str(e))
    
    def on_load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
        )
        
        if file_path:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
                
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_input_label.setPixmap(pixmap)
            self.image_input_path = file_path
            self.anonymize_image_button.setEnabled(True)
            self.hide_personal_info_button_image.setEnabled(True)
            self.statusBar().showMessage(f"Изображение загружено: {file_path}")
            
            self.current_file_path = file_path
    
    def on_anonymize_image(self):
        if not hasattr(self, 'image_input_path'):
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return
            
        selected_types = [data_type for data_type, checkbox in self.image_options.items() if checkbox.isChecked()]
        
        detect_faces = self.detect_faces_checkbox.isChecked()
        
        try:
            import os
            if not os.path.exists(self.image_input_path):
                QMessageBox.warning(self, "Ошибка", f"Файл не найден: {self.image_input_path}")
                return
                
            test_image = cv2.imread(self.image_input_path)
            if test_image is None:
                try:
                    pil_image = Image.open(self.image_input_path)
                    test_image = np.array(pil_image.convert('RGB'))
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
                    
                    temp_path = 'temp_image.png'
                    cv2.imwrite(temp_path, test_image)
                    self.image_input_path = temp_path
                    QMessageBox.information(self, "Информация", "Создана временная копия изображения для обработки")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Невозможно обработать изображение: {str(e)}")
                    return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Невозможно загрузить изображение: {str(e)}")
            return
            
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            self.anonymizer.anonymize_image,
            self.image_input_path,
            selected_types,
            detect_faces
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_image_processed)
        self.worker.start()
        
    def on_hide_personal_info_image(self):
        if not hasattr(self, 'image_input_path'):
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return
            
        # Выбираем только личные данные
        personal_data_types = ['ИИН', 'Телефон', 'Email', 'ФИО', 'Дата рождения', 'Возраст']
        selected_types = [data_type for data_type in personal_data_types if data_type in self.image_options and self.image_options[data_type].isChecked()]
        
        # Всегда включаем распознавание лиц при скрытии личной информации
        detect_faces = True
        
        try:
            import os
            if not os.path.exists(self.image_input_path):
                QMessageBox.warning(self, "Ошибка", f"Файл не найден: {self.image_input_path}")
                return
                
            test_image = cv2.imread(self.image_input_path)
            if test_image is None:
                try:
                    pil_image = Image.open(self.image_input_path)
                    test_image = np.array(pil_image.convert('RGB'))
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
                    
                    temp_path = 'temp_image.png'
                    cv2.imwrite(temp_path, test_image)
                    self.image_input_path = temp_path
                    QMessageBox.information(self, "Информация", "Создана временная копия изображения для обработки")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Невозможно обработать изображение: {str(e)}")
                    return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Невозможно загрузить изображение: {str(e)}")
            return
            
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            self.anonymizer.anonymize_image,
            self.image_input_path,
            selected_types,
            detect_faces
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_image_processed)
        self.worker.start()
        
    def on_image_processed(self, result, message):
        self.progress_bar.setVisible(False)
        
        if result is not None:
            height, width, channel = result.shape
            bytes_per_line = 3 * width
            q_img = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_output_label.setPixmap(pixmap)
            self.anonymized_image = result
            self.save_image_button.setEnabled(True)
            self.statusBar().showMessage(message)
            
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)
        else:
            QMessageBox.warning(self, "Ошибка", message)
    
    def on_save_image(self):
        if not hasattr(self, 'anonymized_image'):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить анонимизированное изображение", "", 
            "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.anonymized_image)
                self.statusBar().showMessage(f"Изображение сохранено в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", str(e))
    
    def on_anonymize_image(self):
        if not hasattr(self, 'image_input_path'):
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return
            
        selected_types = [data_type for data_type, checkbox in self.image_options.items() if checkbox.isChecked()]
        
        detect_faces = self.detect_faces_checkbox.isChecked()
        
        try:
            import os
            if not os.path.exists(self.image_input_path):
                QMessageBox.warning(self, "Ошибка", f"Файл не найден: {self.image_input_path}")
                return
                
            test_image = cv2.imread(self.image_input_path)
            if test_image is None:
                try:
                    pil_image = Image.open(self.image_input_path)
                    test_image = np.array(pil_image.convert('RGB'))
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
                    
                    temp_path = 'temp_image.png'
                    cv2.imwrite(temp_path, test_image)
                    self.image_input_path = temp_path
                    QMessageBox.information(self, "Информация", "Создана временная копия изображения для обработки")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Невозможно обработать изображение: {str(e)}")
                    return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Невозможно загрузить изображение: {str(e)}")
            return
            
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            self.anonymizer.anonymize_image,
            self.image_input_path,
            selected_types,
            detect_faces
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_image_processed)
        self.worker.start()
        
    def on_image_processed(self, result, message):
        self.progress_bar.setVisible(False)
        
        if result is not None:
            height, width, channel = result.shape
            bytes_per_line = 3 * width
            q_img = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_output_label.setPixmap(pixmap)
            self.anonymized_image = result
            self.save_image_button.setEnabled(True)
            self.statusBar().showMessage(message)
            
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)
        else:
            QMessageBox.warning(self, "Ошибка", message)
    
    def on_save_image(self):
        if not hasattr(self, 'anonymized_image'):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить анонимизированное изображение", "", 
            "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.anonymized_image)
                self.statusBar().showMessage(f"Изображение сохранено в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", str(e))
    
    def on_load_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите PDF-файл", "", "PDF-файлы (*.pdf);;Все файлы (*)"
        )
        
        if file_path:
            try:
                pdf_document = fitz.open(file_path)
                if pdf_document.page_count > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    
                    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)
                    
                    self.pdf_preview_label.setPixmap(pixmap.scaled(800, 400, Qt.KeepAspectRatio))
                    
                    self.pdf_info_label.setText(f"PDF-файл: {file_path}\nСтраниц: {pdf_document.page_count}")
                    
                    self.current_pdf_path = file_path
                    self.anonymize_pdf_button.setEnabled(True)
                    self.hide_personal_info_button_pdf.setEnabled(True)
                    self.statusBar().showMessage(f"PDF загружен: {file_path}")
                    
                    self.current_file_path = file_path
                    
                pdf_document.close()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить PDF: {str(e)}")
    
    def on_anonymize_pdf(self):
        if not self.current_pdf_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите PDF-файл")
            return
            
        selected_types = [data_type for data_type, checkbox in self.pdf_options.items() if checkbox.isChecked()]
        
        detect_faces = self.pdf_detect_faces_checkbox.isChecked()
        
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            self.anonymizer.anonymize_pdf,
            self.current_pdf_path,
            selected_types,
            detect_faces
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_pdf_processed)
        self.worker.start()
        
    def on_hide_personal_info_pdf(self):
        if not self.current_pdf_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите PDF-файл")
            return
            
        # Выбираем только личные данные
        personal_data_types = ['ИИН', 'Телефон', 'Email', 'ФИО', 'Дата рождения', 'Возраст']
        selected_types = [data_type for data_type in personal_data_types if data_type in self.pdf_options and self.pdf_options[data_type].isChecked()]
        
        # Всегда включаем распознавание лиц при скрытии личной информации
        detect_faces = True
        
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            self.anonymizer.anonymize_pdf,
            self.current_pdf_path,
            selected_types,
            detect_faces
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_pdf_processed)
        self.worker.start()
        
    def on_pdf_processed(self, result, message):
        self.progress_bar.setVisible(False)
        
        if result:
            self.anonymized_pdf_path = result
            
            try:
                pdf_document = fitz.open(result)
                if pdf_document.page_count > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    
                    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)
                    
                    self.pdf_preview_label.setPixmap(pixmap.scaled(800, 400, Qt.KeepAspectRatio))
                    
                pdf_document.close()
            except Exception as e:
                print(f"Ошибка предпросмотра PDF: {str(e)}")
                QMessageBox.warning(self, "Ошибка предпросмотра", f"Не удалось показать предпросмотр: {str(e)}")
                
            self.save_pdf_button.setEnabled(True)
            self.statusBar().showMessage(message)
            
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)
        else:
            QMessageBox.warning(self, "Ошибка", message)
    
    def on_save_pdf(self):
        if not hasattr(self, 'anonymized_pdf_path'):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить анонимизированный PDF", "", "PDF-файлы (*.pdf);;Все файлы (*)"
        )
        
        if file_path:
            try:
                import shutil
                shutil.copy2(self.anonymized_pdf_path, file_path)
                self.statusBar().showMessage(f"PDF сохранен в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", str(e))
    
    def on_save_report(self):
        report_text = self.report_text.toPlainText()
        if not report_text:
            QMessageBox.warning(self, "Предупреждение", "Отчет пуст")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчет", "", "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(report_text)
                self.statusBar().showMessage(f"Отчет сохранен в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", str(e))
    
    def on_anonymize_pdf(self):
        if not self.current_pdf_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите PDF-файл")
            return
            
        selected_types = [data_type for data_type, checkbox in self.pdf_options.items() if checkbox.isChecked()]
        
        detect_faces = self.pdf_detect_faces_checkbox.isChecked()
        
        self.progress_bar.setVisible(True)
        
        self.worker = WorkerThread(
            self.anonymizer.anonymize_pdf,
            self.current_pdf_path,
            selected_types,
            detect_faces
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_pdf_processed)
        self.worker.start()
        
    def on_pdf_processed(self, result, message):
        self.progress_bar.setVisible(False)
        
        if result:
            self.anonymized_pdf_path = result
            
            try:
                pdf_document = fitz.open(result)
                if pdf_document.page_count > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    
                    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)
                    
                    self.pdf_preview_label.setPixmap(pixmap.scaled(800, 400, Qt.KeepAspectRatio))
                    
                pdf_document.close()
            except Exception as e:
                print(f"Ошибка предпросмотра PDF: {str(e)}")
                QMessageBox.warning(self, "Ошибка предпросмотра", f"Не удалось показать предпросмотр: {str(e)}")
                
            self.save_pdf_button.setEnabled(True)
            self.statusBar().showMessage(message)
            
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)
        else:
            QMessageBox.warning(self, "Ошибка", message)
    
    def on_save_pdf(self):
        if not hasattr(self, 'anonymized_pdf_path'):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить анонимизированный PDF", "", "PDF-файлы (*.pdf);;Все файлы (*)"
        )
        
        if file_path:
            try:
                import shutil
                shutil.copy2(self.anonymized_pdf_path, file_path)
                self.statusBar().showMessage(f"PDF сохранен в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", str(e))
    
    def on_save_report(self):
        report_text = self.report_text.toPlainText()
        if not report_text:
            QMessageBox.warning(self, "Предупреждение", "Отчет пуст")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчет", "", "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(report_text)
                self.statusBar().showMessage(f"Отчет сохранен в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", str(e))
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
