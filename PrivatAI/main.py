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

# Импорты из PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog, 
                           QMessageBox, QProgressBar, QComboBox, QCheckBox, QGridLayout,
                           QLineEdit, QListWidget, QListWidgetItem, QSplitter, QFrame,
                           QAbstractItemDelegate)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QTextDocument
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QBuffer, QIODevice, QSize

# Импорт чат-бота
from chatbot import ChatBot

# Настраиваем путь к tesseract, если он не в PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Для Windows
# Для Linux/Mac обычно не требуется, если установлен через пакетный менеджер

class AnonymizerEngine:
    def __init__(self):
        # Регулярные выражения для распознавания персональных данных
        self.patterns = {
            'ИИН': r'\b\d{12}\b',  # 12 цифр для ИИН Казахстана
            'Телефон': r'\+?[7]\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}',  # Казахстанский формат
            'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'Банковская карта': r'\b(?:\d{4}[\s-]?){4}\b|\b\d{16}\b',
            'ФИО': r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b',  # Простой паттерн для ФИО
            'Банковский счет': r'\b\d{20}\b',  # 20 цифр для банковского счета Казахстана
            'IBAN': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}[0-9]{16}\b',  # Типичный формат IBAN
            'Дата рождения': r'\b(0[1-9]|[12][0-9]|3[01])[./-](0[1-9]|1[0-2])[./-](19|20)\d\d\b'  # ДД.ММ.ГГГГ
        }
        
        # Словарь для хранения распознанных данных (для отчета)
        self.detected_data = {}
        
        # Загрузка каскадов для распознавания лиц
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def anonymize_text(self, text, data_types=None):
        if data_types is None:
            data_types = self.patterns.keys()
            
        # Очищаем предыдущие найденные данные
        self.detected_data = {data_type: [] for data_type in data_types}
        
        anonymized_text = text
        for data_type in data_types:
            if data_type in self.patterns:
                pattern = self.patterns[data_type]
                
                # Находим все совпадения
                matches = re.finditer(pattern, anonymized_text)
                for match in matches:
                    found_text = match.group(0)
                    start, end = match.span()
                    
                    # Добавляем найденные данные для отчета
                    self.detected_data[data_type].append(found_text)
                    
                    # Заменяем персональные данные на маску
                    if data_type == 'ИИН':
                        mask = 'XXX-XXX-XXXX'
                    elif data_type == 'Телефон':
                        mask = '+X-XXX-XXX-XXXX'
                    elif data_type == 'Email':
                        username, domain = found_text.split('@')
                        mask = username[0] + 'X' * (len(username) - 2) + username[-1] + '@' + domain
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
                    else:
                        mask = 'X' * len(found_text)
                        
                    anonymized_text = anonymized_text[:start] + mask + anonymized_text[end:]
        
        return anonymized_text
    
    def detect_and_blur_faces(self, image, blur_factor=35):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            face_count = len(faces)
            
            # Для каждого обнаруженного лица
            for (x, y, w, h) in faces:
                # Вырезаем область лица
                face_roi = image[y:y+h, x:x+w]
                
                # Размываем лицо
                blurred_face = cv2.GaussianBlur(face_roi, (blur_factor, blur_factor), 0)
                
                # Заменяем область на размытую версию
                image[y:y+h, x:x+w] = blurred_face
            
            return image, face_count
        except Exception as e:
            # В случае ошибки возвращаем оригинальное изображение и 0 лиц
            return image, 0
    
    def anonymize_image(self, image_path, data_types=None, detect_faces=True):
        if data_types is None:
            data_types = self.patterns.keys()
            
        # Загружаем изображение
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Альтернативный способ загрузки с использованием PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            if image is None:
                return None, "Не удалось загрузить изображение"
        except Exception as e:
            return None, f"Ошибка при загрузке изображения: {str(e)}"
            
        # Создаем копию для анонимизированной версии
        anonymized_image = image.copy()
        
        # Детектируем и маскируем лица, если включена опция
        if detect_faces:
            anonymized_image, face_count = self.detect_and_blur_faces(anonymized_image)
            if face_count > 0:
                self.detected_data['Лица'] = [f"Обнаружено лиц: {face_count}"]
        
        # Используем pytesseract для извлечения текста и его расположения
        try:
            # Конвертируем в градации серого для лучшего распознавания
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text_data = pytesseract.image_to_data(gray, lang='rus+eng', output_type=pytesseract.Output.DICT)
            
            # Очищаем предыдущие найденные данные (если не было обнаружено лиц)
            if 'Лица' not in self.detected_data:
                self.detected_data = {data_type: [] for data_type in data_types}
            else:
                # Иначе инициализируем только отсутствующие ключи
                for data_type in data_types:
                    if data_type not in self.detected_data:
                        self.detected_data[data_type] = []
            
            # Объединяем слова в строки для более точного сопоставления с шаблонами
            full_text = " ".join(text_data['text'])
            
            # Анонимизируем найденный текст
            for data_type in data_types:
                if data_type in self.patterns:
                    pattern = self.patterns[data_type]
                    
                    # Ищем соответствия в полном тексте
                    for match in re.finditer(pattern, full_text):
                        found_text = match.group(0)
                        self.detected_data[data_type].append(found_text)
                        
                        # Ищем соответствующие слова на изображении
                        for i, word in enumerate(text_data['text']):
                            if word and word in found_text:
                                # Закрашиваем область с персональными данными
                                x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
                                cv2.rectangle(anonymized_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
        except Exception as e:
            # Если произошла ошибка при распознавании текста, просто продолжаем
            # и возвращаем изображение с замаскированными лицами
            pass
        
        return anonymized_image, "Изображение обработано"
    
    def anonymize_pdf(self, pdf_path, data_types=None, detect_faces=True):
        if data_types is None:
            data_types = self.patterns.keys()
            
        try:
            # Открываем PDF-файл
            pdf_document = fitz.open(pdf_path)
            
            # Очищаем предыдущие найденные данные
            self.detected_data = {data_type: [] for data_type in data_types}
            if detect_faces:
                self.detected_data['Лица'] = []
            
            # Для хранения анонимизированного текста страниц
            anonymized_pages = []
            
            # Обрабатываем каждую страницу
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Извлекаем текст со страницы
                text = page.get_text()
                
                # Анонимизируем текст
                anonymized_text = self.anonymize_text(text, data_types)
                anonymized_pages.append(anonymized_text)
                
                # Обработка изображений на странице
                try:
                    image_list = page.get_images(full=True)
                    
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Конвертируем bytes в изображение
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if img is not None and detect_faces:
                                # Обнаруживаем и размываем лица
                                anonymized_img, face_count = self.detect_and_blur_faces(img)
                                
                                if face_count > 0:
                                    # Добавляем информацию о найденных лицах
                                    self.detected_data['Лица'].append(f"Стр. {page_num+1}, изобр. {img_index+1}: {face_count} лиц")
                                    
                                    # Сохраняем анонимизированное изображение во временный файл
                                    temp_img_path = f"temp_img_{random.randint(1000, 9999)}.png"
                                    cv2.imwrite(temp_img_path, anonymized_img)
                                    
                                    # Заменяем изображение в PDF
                                    rect = page.get_image_rects(xref)[0]  # Получаем положение изображения
                                    page.delete_image(xref)  # Удаляем оригинальное изображение
                                    page.insert_image(rect, filename=temp_img_path)  # Вставляем анонимизированное
                                    
                                    # Удаляем временный файл
                                    if os.path.exists(temp_img_path):
                                        os.remove(temp_img_path)
                        except Exception as e:
                            # Пропускаем изображение при ошибке
                            continue
                except Exception as e:
                    # Пропускаем обработку изображений, если возникла ошибка
                    pass
                
                # Создаем новую страницу с анонимизированным текстом
                try:
                    page.apply_redactions()  # Очищаем страницу
                    page.insert_text((50, 50), anonymized_text)  # Вставляем анонимизированный текст
                except Exception as e:
                    # Пропускаем обновление текста, если возникла ошибка
                    pass
            
            # Если лица не были обнаружены, и ключ 'Лица' существует, удаляем его из отчета
            if 'Лица' in self.detected_data and not self.detected_data['Лица']:
                del self.detected_data['Лица']
            
            # Сохраняем изменения во временный файл
            temp_path = pdf_path.replace('.pdf', '_anonymized.pdf')
            pdf_document.save(temp_path)
            pdf_document.close()
            
            return temp_path, "PDF обработан"
            
        except Exception as e:
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
        # Эмуляция прогресса (в реальном приложении будет основана на реальном прогрессе)
        for i in range(101):
            self.progress.emit(i)
            self.msleep(10)  # Небольшая задержка
            
        # Выполняем функцию
        try:
            result, message = self.func(*self.args, **self.kwargs)
            self.finished.emit(result, message)
        except Exception as e:
            self.finished.emit(None, f"Ошибка: {str(e)}")

class ChatItemDelegate(QAbstractItemDelegate):
    def __init__(self):
        super().__init__()
        
    def paint(self, painter, option, index):
        # Получаем данные элемента
        data = index.data(Qt.UserRole)
        if not data:
            return
            
        text = data["text"]
        is_user = data["is_user"]
        
        # Настройка цветов и стилей
        user_color = QColor(220, 240, 255)
        bot_color = QColor(240, 240, 240)
        text_color = QColor(0, 0, 0)
        
        # Получаем прямоугольник для отрисовки
        rect = option.rect
        
        # Рисуем фон
        background_color = user_color if is_user else bot_color
        painter.fillRect(rect, background_color)
        
        # Рисуем границу
        painter.setPen(QColor(200, 200, 200))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        # Рисуем текст
        text_rect = rect.adjusted(10, 5, -10, -5)
        painter.setPen(text_color)
        painter.setFont(QFont("Arial", 9))
        
        # Добавляем заголовок
        header = "Вы:" if is_user else "Бот:"
        header_font = QFont("Arial", 9, QFont.Bold)
        painter.setFont(header_font)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, header)
        
        # Рисуем основной текст
        text_rect = text_rect.adjusted(0, 20, 0, 0)
        painter.setFont(QFont("Arial", 9))
        
        # Разбиваем текст на строки для корректного отображения
        document = QTextDocument()
        document.setDefaultFont(QFont("Arial", 9))
        document.setTextWidth(text_rect.width())
        document.setHtml(text.replace("\n", "<br>"))
        
        # Адаптируем размер элемента к содержимому
        if index.model():
            height = document.size().height() + 30  # +30 для заголовка и отступов
            index.model().setData(index, QSize(int(text_rect.width()), int(height)), Qt.SizeHintRole)
        
        # Отрисовываем текст
        painter.save()
        painter.translate(text_rect.topLeft())
        document.drawContents(painter)
        painter.restore()
    
    def sizeHint(self, option, index):
        # Возвращаем размер, сохраненный в данных
        size = index.data(Qt.SizeHintRole)
        if size:
            return size
        return QSize(option.rect.width(), 50)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.anonymizer = AnonymizerEngine()
        self.chatbot = ChatBot(self.anonymizer)
        self.current_file_path = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("SafeMind - Анонимизация данных")
        self.setGeometry(100, 100, 1000, 750)
        
        # Создаем центральный виджет и основной layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем вкладки для разных типов данных
        self.tabs = QTabWidget()
        
        # Вкладка для текста
        self.text_tab = QWidget()
        self.setup_text_tab()
        self.tabs.addTab(self.text_tab, "Текст")
        
        # Вкладка для изображений
        self.image_tab = QWidget()
        self.setup_image_tab()
        self.tabs.addTab(self.image_tab, "Изображения")
        
        # Вкладка для PDF
        self.pdf_tab = QWidget()
        self.setup_pdf_tab()
        self.tabs.addTab(self.pdf_tab, "PDF-файлы")
        
        # Вкладка отчета
        self.report_tab = QWidget()
        self.setup_report_tab()
        self.tabs.addTab(self.report_tab, "Отчет")
        
        # Вкладка чат-бота
        self.chat_tab = QWidget()
        self.setup_chat_tab()
        self.tabs.addTab(self.chat_tab, "Чат-бот")
        
        main_layout.addWidget(self.tabs)
        
        # Создаем статусбар
        self.statusBar().showMessage('Готов к работе')
        
        # Индикатор прогресса
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
    def setup_text_tab(self):
        layout = QVBoxLayout(self.text_tab)
        
        # Область для ввода текста
        input_label = QLabel("Введите текст для анонимизации:")
        layout.addWidget(input_label)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Вставьте текст, содержащий персональные данные...")
        layout.addWidget(self.text_input)
        
        # Настройки анонимизации
        options_layout = QHBoxLayout()
        
        # Чекбоксы для выбора типов данных
        self.text_options = {}
        options_widget = QWidget()
        options_grid = QGridLayout(options_widget)
        
        row, col = 0, 0
        for data_type in self.anonymizer.patterns.keys():
            self.text_options[data_type] = QCheckBox(data_type)
            self.text_options[data_type].setChecked(True)
            options_grid.addWidget(self.text_options[data_type], row, col)
            col += 1
            if col > 2:  # 3 чекбокса в строке
                col = 0
                row += 1
                
        options_layout.addWidget(options_widget)
        layout.addLayout(options_layout)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        self.anonymize_text_button = QPushButton("Анонимизировать")
        self.anonymize_text_button.clicked.connect(self.on_anonymize_text)
        buttons_layout.addWidget(self.anonymize_text_button)
        
        self.clear_text_button = QPushButton("Очистить")
        self.clear_text_button.clicked.connect(self.on_clear_text_tab)
        buttons_layout.addWidget(self.clear_text_button)
        
        layout.addLayout(buttons_layout)
        
        # Область для вывода результата
        result_label = QLabel("Результат:")
        layout.addWidget(result_label)
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        layout.addWidget(self.text_output)
        
        # Кнопка сохранения
        self.save_text_button = QPushButton("Сохранить результат")
        self.save_text_button.clicked.connect(self.on_save_text)
        self.save_text_button.setEnabled(False)
        layout.addWidget(self.save_text_button)
        
    def setup_image_tab(self):
        layout = QVBoxLayout(self.image_tab)
        
        # Область для загрузки и отображения изображения
        input_layout = QHBoxLayout()
        
        # Левая панель с исходным изображением
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
        
        # Правая панель с обработанным изображением
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
        
        # Настройки анонимизации
        options_layout = QHBoxLayout()
        
        # Чекбоксы для выбора типов данных
        self.image_options = {}
        options_widget = QWidget()
        options_grid = QGridLayout(options_widget)
        
        row, col = 0, 0
        for data_type in self.anonymizer.patterns.keys():
            self.image_options[data_type] = QCheckBox(data_type)
            self.image_options[data_type].setChecked(True)
            options_grid.addWidget(self.image_options[data_type], row, col)
            col += 1
            if col > 2:  # 3 чекбокса в строке
                col = 0
                row += 1
        
        # Добавляем чекбокс для распознавания лиц
        self.detect_faces_checkbox = QCheckBox("Обнаружение и размытие лиц")
        self.detect_faces_checkbox.setChecked(True)
        options_grid.addWidget(self.detect_faces_checkbox, row + 1, 0, 1, 3)
                
        options_layout.addWidget(options_widget)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        self.anonymize_image_button = QPushButton("Анонимизировать")
        self.anonymize_image_button.clicked.connect(self.on_anonymize_image)
        self.anonymize_image_button.setEnabled(False)
        buttons_layout.addWidget(self.anonymize_image_button)
        
        self.clear_image_button = QPushButton("Очистить")
        self.clear_image_button.clicked.connect(self.on_clear_image_tab)
        buttons_layout.addWidget(self.clear_image_button)
        
        options_layout.addLayout(buttons_layout)
        
        layout.addLayout(options_layout)
        
    def setup_pdf_tab(self):
        layout = QVBoxLayout(self.pdf_tab)
        
        # Информация о PDF
        self.pdf_info_label = QLabel("Загрузите PDF-файл для анонимизации.")
        layout.addWidget(self.pdf_info_label)
        
        # Кнопки для загрузки PDF
        pdf_buttons_layout = QHBoxLayout()
        
        self.load_pdf_button = QPushButton("Загрузить PDF")
        self.load_pdf_button.clicked.connect(self.on_load_pdf)
        pdf_buttons_layout.addWidget(self.load_pdf_button)
        
        self.current_pdf_path = None
        
        layout.addLayout(pdf_buttons_layout)
        
        # Настройки анонимизации
        options_layout = QHBoxLayout()
        
        # Чекбоксы для выбора типов данных
        self.pdf_options = {}
        options_widget = QWidget()
        options_grid = QGridLayout(options_widget)
        
        row, col = 0, 0
        for data_type in self.anonymizer.patterns.keys():
            self.pdf_options[data_type] = QCheckBox(data_type)
            self.pdf_options[data_type].setChecked(True)
            options_grid.addWidget(self.pdf_options[data_type], row, col)
            col += 1
            if col > 2:  # 3 чекбокса в строке
                col = 0
                row += 1
        
        # Добавляем чекбокс для распознавания лиц
        self.pdf_detect_faces_checkbox = QCheckBox("Обнаружение и размытие лиц")
        self.pdf_detect_faces_checkbox.setChecked(True)
        options_grid.addWidget(self.pdf_detect_faces_checkbox, row + 1, 0, 1, 3)
                
        options_layout.addWidget(options_widget)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        self.anonymize_pdf_button = QPushButton("Анонимизировать PDF")
        self.anonymize_pdf_button.clicked.connect(self.on_anonymize_pdf)
        self.anonymize_pdf_button.setEnabled(False)
        buttons_layout.addWidget(self.anonymize_pdf_button)
        
        self.clear_pdf_button = QPushButton("Очистить")
        self.clear_pdf_button.clicked.connect(self.on_clear_pdf_tab)
        buttons_layout.addWidget(self.clear_pdf_button)
        
        options_layout.addLayout(buttons_layout)
        
        layout.addLayout(options_layout)
        
        # Область для предпросмотра PDF
        preview_label = QLabel("Предпросмотр первой страницы:")
        layout.addWidget(preview_label)
        
        self.pdf_preview_label = QLabel()
        self.pdf_preview_label.setAlignment(Qt.AlignCenter)
        self.pdf_preview_label.setMinimumSize(800, 400)
        self.pdf_preview_label.setStyleSheet("border: 1px solid #cccccc;")
        layout.addWidget(self.pdf_preview_label)
        
        # Кнопка сохранения
        self.save_pdf_button = QPushButton("Сохранить анонимизированный PDF")
        self.save_pdf_button.clicked.connect(self.on_save_pdf)
        self.save_pdf_button.setEnabled(False)
        layout.addWidget(self.save_pdf_button)
        
    def setup_report_tab(self):
        layout = QVBoxLayout(self.report_tab)
        
        # Заголовок
        report_header = QLabel("Отчет о найденных персональных данных")
        report_header.setFont(QFont("Arial", 14, QFont.Bold))
        report_header.setAlignment(Qt.AlignCenter)
        layout.addWidget(report_header)
        
        # Текстовая область для отчета
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Здесь будет отображен отчет после анонимизации...")
        layout.addWidget(self.report_text)
        
        # Кнопки действий
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
        
        # Заголовок
        chat_header = QLabel("SafeMind Чат-бот")
        chat_header.setFont(QFont("Arial", 14, QFont.Bold))
        chat_header.setAlignment(Qt.AlignCenter)
        layout.addWidget(chat_header)
        
        # Описание
        chat_desc = QLabel("Задайте вопрос или введите команду для анонимизации данных")
        chat_desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(chat_desc)
        
        # Область чата
        chat_area = QSplitter(Qt.Vertical)
        
        # История сообщений
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
        
        # Настраиваем делегат для отображения сообщений
        self.chat_history.setItemDelegate(ChatItemDelegate())
        chat_area.addWidget(self.chat_history)
        
        # Поле ввода сообщений
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
        
        # Устанавливаем соотношение размеров
        chat_area.setSizes([400, 50])
        
        layout.addWidget(chat_area)
        
        # Кнопки быстрых команд
        commands_layout = QHBoxLayout()
        
        quick_commands_label = QLabel("Быстрые команды:")
        commands_layout.addWidget(quick_commands_label)
        
        self.clear_chat_button = QPushButton("Очистить")
        self.clear_chat_button.clicked.connect(self.on_clear_chat_tab)
        commands_layout.addWidget(self.clear_chat_button)
        
        layout.addLayout(commands_layout)
        
        # Кнопки быстрых команд
        quick_commands_layout = QHBoxLayout()
        
        commands = ["Помощь", "Анонимизировать текст", "Размыть лица", "Функции"]
        for command in commands:
            cmd_button = QPushButton(command)
            cmd_button.clicked.connect(lambda checked, cmd=command: self.add_quick_command(cmd))
            quick_commands_layout.addWidget(cmd_button)
        
        layout.addLayout(quick_commands_layout)
        
        # Добавляем приветственное сообщение
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
        self.save_image_button.setEnabled(False)
        self.statusBar().showMessage('Вкладка изображений очищена')
    
    def on_clear_pdf_tab(self):
        self.pdf_preview_label.clear()
        self.pdf_info_label.setText("Загрузите PDF-файл для анонимизации.")
        self.current_pdf_path = None
        if hasattr(self, 'anonymized_pdf_path'):
            delattr(self, 'anonymized_pdf_path')
        self.anonymize_pdf_button.setEnabled(False)
        self.save_pdf_button.setEnabled(False)
        self.statusBar().showMessage('Вкладка PDF очищена')
    
    def on_clear_report_tab(self):
        self.report_text.clear()
        self.statusBar().showMessage('Отчет очищен')
    
    def on_clear_chat_tab(self):
        self.chat_history.clear()
        self.chat_input.clear()
        # Добавляем снова приветственное сообщение
        self.add_bot_message(self.chatbot.greet("", None))
        self.statusBar().showMessage('Чат очищен')
        
    # ---- Обработчики чат-бота ----
    
    def on_send_message(self):
        message = self.chat_input.text().strip()
        if not message:
            return
        
        # Добавляем сообщение пользователя в историю
        self.add_user_message(message)
        
        # Обрабатываем сообщение с помощью чат-бота
        response = self.chatbot.process_message(message, self.current_file_path)
        
        # Добавляем ответ бота в историю
        self.add_bot_message(response)
        
        # Очищаем поле ввода
        self.chat_input.clear()
        
    def add_user_message(self, message):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, {"text": message, "is_user": True})
        item.setSizeHint(QSize(self.chat_history.width(), 50))  # Начальный размер
        self.chat_history.addItem(item)
        self.chat_history.scrollToBottom()
        
    def add_bot_message(self, message):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, {"text": message, "is_user": False})
        item.setSizeHint(QSize(self.chat_history.width(), 100))  # Начальный размер
        self.chat_history.addItem(item)
        self.chat_history.scrollToBottom()
        
    def add_quick_command(self, command):
        self.chat_input.setText(command)
        self.on_send_message()
        
    # ---- Обработчики событий ----
    
    def on_anonymize_text(self):
        input_text = self.text_input.toPlainText()
        if not input_text:
            QMessageBox.warning(self, "Предупреждение", "Введите текст для анонимизации")
            return
            
        # Получаем выбранные типы данных
        selected_types = [data_type for data_type, checkbox in self.text_options.items() if checkbox.isChecked()]
        
        # Обновляем текущий путь к файлу для чат-бота (в данном случае это не файл, а текст)
        self.current_file_path = "текст"
        
        # Показываем прогресс
        self.progress_bar.setVisible(True)
        
        # Запускаем анонимизацию в отдельном потоке
        self.worker = WorkerThread(
            lambda: (self.anonymizer.anonymize_text(input_text, selected_types), "Текст обработан")
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
            
            # Обновляем отчет
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)  # Переключаемся на вкладку отчета
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
                
            # Масштабируем изображение для отображения
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_input_label.setPixmap(pixmap)
            self.image_input_path = file_path
            self.anonymize_image_button.setEnabled(True)
            self.statusBar().showMessage(f"Изображение загружено: {file_path}")
            
            # Обновляем текущий путь к файлу для чат-бота
            self.current_file_path = file_path
    
    def on_anonymize_image(self):
        if not hasattr(self, 'image_input_path'):
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return
            
        # Получаем выбранные типы данных
        selected_types = [data_type for data_type, checkbox in self.image_options.items() if checkbox.isChecked()]
        
        # Проверяем статус чекбокса обнаружения лиц
        detect_faces = self.detect_faces_checkbox.isChecked()
        
        # Проверка существования файла
        try:
            import os
            if not os.path.exists(self.image_input_path):
                QMessageBox.warning(self, "Ошибка", f"Файл не найден: {self.image_input_path}")
                return
                
            # Пробуем прочитать изображение напрямую
            test_image = cv2.imread(self.image_input_path)
            if test_image is None:
                # Альтернативный способ загрузки
                try:
                    pil_image = Image.open(self.image_input_path)
                    test_image = np.array(pil_image.convert('RGB'))
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
                    
                    # Создаем временную копию изображения с простым именем
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
            
        # Показываем прогресс
        self.progress_bar.setVisible(True)
        
        # Запускаем анонимизацию в отдельном потоке
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
            # Конвертируем OpenCV изображение в QPixmap
            height, width, channel = result.shape
            bytes_per_line = 3 * width
            q_img = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            # Масштабируем изображение для отображения
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_output_label.setPixmap(pixmap)
            self.anonymized_image = result
            self.save_image_button.setEnabled(True)
            self.statusBar().showMessage(message)
            
            # Обновляем отчет
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)  # Переключаемся на вкладку отчета
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
                # Загружаем PDF-файл и показываем предпросмотр первой страницы
                pdf_document = fitz.open(file_path)
                if pdf_document.page_count > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    
                    # Конвертируем в QImage и затем в QPixmap
                    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)
                    
                    # Устанавливаем предпросмотр
                    self.pdf_preview_label.setPixmap(pixmap.scaled(800, 400, Qt.KeepAspectRatio))
                    
                    # Обновляем информацию
                    self.pdf_info_label.setText(f"PDF-файл: {file_path}\nСтраниц: {pdf_document.page_count}")
                    
                    self.current_pdf_path = file_path
                    self.anonymize_pdf_button.setEnabled(True)
                    self.statusBar().showMessage(f"PDF загружен: {file_path}")
                    
                    # Обновляем текущий путь к файлу для чат-бота
                    self.current_file_path = file_path
                    
                pdf_document.close()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить PDF: {str(e)}")
    
    def on_anonymize_pdf(self):
        if not self.current_pdf_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите PDF-файл")
            return
            
        # Получаем выбранные типы данных
        selected_types = [data_type for data_type, checkbox in self.pdf_options.items() if checkbox.isChecked()]
        
        # Проверяем статус чекбокса обнаружения лиц
        detect_faces = self.pdf_detect_faces_checkbox.isChecked()
        
        # Показываем прогресс
        self.progress_bar.setVisible(True)
        
        # Запускаем анонимизацию в отдельном потоке
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
                # Показываем предпросмотр первой страницы анонимизированного PDF
                pdf_document = fitz.open(result)
                if pdf_document.page_count > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    
                    # Конвертируем в QImage и затем в QPixmap
                    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)
                    
                    # Устанавливаем предпросмотр
                    self.pdf_preview_label.setPixmap(pixmap.scaled(800, 400, Qt.KeepAspectRatio))
                    
                pdf_document.close()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка предпросмотра", str(e))
                
            self.save_pdf_button.setEnabled(True)
            self.statusBar().showMessage(message)
            
            # Обновляем отчет
            self.report_text.setText(self.anonymizer.get_report())
            self.tabs.setCurrentIndex(3)  # Переключаемся на вкладку отчета
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
                # Копируем временный файл в выбранное пользователем место
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