"""
SafeMind Chatbot - модуль диалогового помощника для анонимизации данных
С возможностью интеграции с Gemini AI
"""

try:
    from gemini_api import GeminiAPI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class ChatBot:
    """Простой чат-бот для взаимодействия с пользователем и анонимизации данных"""
    
    def __init__(self, anonymizer):
        self.anonymizer = anonymizer
        self.commands = {
            'привет': self.greet,
            'помощь': self.help_command,
            'анонимизировать текст': self.anonymize_text,
            'анонимизировать изображение': self.anonymize_image,
            'анонимизировать pdf': self.anonymize_pdf,
            'очистить от фио': self.anonymize_fio,
            'очистить от иин': self.anonymize_iin,
            'очистить от карт': self.anonymize_card,
            'проверить на приватную информацию': self.check_private_info,
            'очистить лица': self.blur_faces,
            'размыть лица': self.blur_faces,
            'функции': self.list_functions,
            'возможности': self.list_functions,
            'что ты умеешь': self.list_functions,
            'включить gemini': self.enable_gemini,
            'выключить gemini': self.disable_gemini
        }
        
        # Инициализация Gemini API, если доступно
        if GEMINI_AVAILABLE:
            self.gemini_api = GeminiAPI()
            self.use_gemini = self.gemini_api.is_available()
        else:
            self.gemini_api = None
            self.use_gemini = False
        
    def process_message(self, message, current_file_path=None):
        """Обрабатывает сообщение пользователя и возвращает ответ"""
        message = message.strip().lower()
        
        # Проверяем прямые совпадения с командами
        for command, handler in self.commands.items():
            if message == command or message.startswith(command + ' '):
                args = message[len(command):].strip()
                return handler(args, current_file_path)
                
        # Проверяем совпадения по ключевым словам
        if 'привет' in message:
            return self.greet('', current_file_path)
        elif any(word in message for word in ['помощь', 'помоги', 'как пользоваться']):
            return self.help_command('', current_file_path)
        elif all(word in message for word in ['анонимизировать', 'текст']):
            return self.anonymize_text(message, current_file_path)
        elif all(word in message for word in ['анонимизировать', 'изображение']) or all(word in message for word in ['анонимизировать', 'фото']):
            return self.anonymize_image(message, current_file_path)
        elif all(word in message for word in ['анонимизировать', 'pdf']):
            return self.anonymize_pdf(message, current_file_path)
        elif any(word in message for word in ['очистить', 'удалить', 'скрыть']) and any(word in message for word in ['фио', 'имя', 'фамилия', 'имена']):
            return self.anonymize_fio(message, current_file_path)
        elif any(word in message for word in ['очистить', 'удалить', 'скрыть']) and any(word in message for word in ['иин', 'ин', 'индивидуальный номер']):
            return self.anonymize_iin(message, current_file_path)
        elif any(word in message for word in ['очистить', 'удалить', 'скрыть']) and any(word in message for word in ['карта', 'карты', 'банковская', 'кредитная']):
            return self.anonymize_card(message, current_file_path)
        elif any(word in message for word in ['проверить', 'проверка', 'найти']) and any(word in message for word in ['приватная', 'личная', 'персональная', 'информация', 'данные']):
            return self.check_private_info(message, current_file_path)
        elif any(word in message for word in ['скрыть', 'размыть', 'блюрить', 'очистить']) and any(word in message for word in ['лица', 'лицо', 'люди']):
            return self.blur_faces(message, current_file_path)
        elif any(word in message for word in ['функции', 'возможности', 'команды', 'что ты умеешь']):
            return self.list_functions('', current_file_path)
        elif any(word in message for word in ['включи', 'активируй', 'подключи']) and any(word in message for word in ['gemini', 'гемини', 'api']):
            return self.enable_gemini('', current_file_path)
        elif any(word in message for word in ['выключи', 'деактивируй', 'отключи']) and any(word in message for word in ['gemini', 'гемини', 'api']):
            return self.disable_gemini('', current_file_path)
        else:
            # Если нет совпадений с командами и включен Gemini API, пробуем получить ответ от API
            if self.use_gemini and self.gemini_api:
                response = self.gemini_api.get_response(message)
                return response
            else:
                return "Извините, я не понимаю эту команду. Напишите 'помощь', чтобы узнать, что я умею."
    
    def greet(self, args, current_file_path=None):
        """Приветствие"""
        return """Здравствуйте! Я бот-помощник для анонимизации данных SafeMind.
        
Я могу помочь вам:
• Анонимизировать текст, изображения и PDF-файлы
• Обнаруживать и маскировать ФИО, ИИН, банковские карты и другие данные
• Размывать лица на фотографиях

Напишите 'помощь', чтобы узнать больше о моих возможностях."""
    
    def help_command(self, args, current_file_path=None):
        """Помощь по командам"""
        base_help = """Список доступных команд:

• анонимизировать текст - маскирует персональные данные в тексте
• анонимизировать изображение - обрабатывает изображение
• анонимизировать pdf - обрабатывает PDF-файл
• очистить от фио - скрывает имена людей
• очистить от иин - скрывает индивидуальные идентификационные номера
• очистить от карт - скрывает номера банковских карт
• проверить на приватную информацию - анализирует данные на наличие конфиденциальной информации
• размыть лица - распознает и маскирует лица на изображениях
• функции - показывает список всех возможностей

Вы также можете использовать основной интерфейс программы для более точной настройки анонимизации."""

        # Добавляем информацию о Gemini API, если доступно
        if GEMINI_AVAILABLE:
            gemini_status = "включен" if self.use_gemini else "выключен"
            gemini_help = f"""
Интеграция с Gemini AI ({gemini_status}):
• включить gemini - активирует ответы от Gemini AI
• выключить gemini - отключает использование Gemini AI
"""
            return base_help + gemini_help
        else:
            return base_help
    
    def anonymize_text(self, args, current_file_path=None):
        """Команда для анонимизации текста"""
        if not current_file_path:
            return "Пожалуйста, сначала введите текст или загрузите файл для анонимизации через интерфейс программы."
        return "Функция анонимизации текста активирована. Пожалуйста, используйте интерфейс программы для продолжения."
    
    def anonymize_image(self, args, current_file_path=None):
        """Команда для анонимизации изображения"""
        if not current_file_path:
            return "Пожалуйста, сначала загрузите изображение через интерфейс программы."
        return "Функция анонимизации изображения активирована. Пожалуйста, используйте интерфейс программы для продолжения."
    
    def anonymize_pdf(self, args, current_file_path=None):
        """Команда для анонимизации PDF"""
        if not current_file_path:
            return "Пожалуйста, сначала загрузите PDF-файл через интерфейс программы."
        return "Функция анонимизации PDF активирована. Пожалуйста, используйте интерфейс программы для продолжения."
    
    def anonymize_fio(self, args, current_file_path=None):
        """Команда для очистки ФИО"""
        if not current_file_path:
            return "Пожалуйста, сначала введите текст или загрузите файл через интерфейс программы."
        return "Активирована очистка ФИО. Запустите анонимизацию через интерфейс программы, убедившись, что отмечен параметр 'ФИО'."
    
    def anonymize_iin(self, args, current_file_path=None):
        """Команда для очистки ИИН"""
        if not current_file_path:
            return "Пожалуйста, сначала введите текст или загрузите файл через интерфейс программы."
        return "Активирована очистка ИИН. Запустите анонимизацию через интерфейс программы, убедившись, что отмечен параметр 'ИИН'."
    
    def anonymize_card(self, args, current_file_path=None):
        """Команда для очистки банковских карт"""
        if not current_file_path:
            return "Пожалуйста, сначала введите текст или загрузите файл через интерфейс программы."
        return "Активирована очистка банковских карт. Запустите анонимизацию через интерфейс программы, убедившись, что отмечен параметр 'Банковская карта'."
    
    def check_private_info(self, args, current_file_path=None):
        """Команда для проверки на приватную информацию"""
        if not current_file_path:
            return "Пожалуйста, сначала введите текст или загрузите файл через интерфейс программы."
        return "Активирована проверка на приватную информацию. Запустите анонимизацию через интерфейс программы, после чего перейдите на вкладку 'Отчет'."
    
    def blur_faces(self, args, current_file_path=None):
        """Команда для размытия лиц"""
        if not current_file_path:
            return "Пожалуйста, сначала загрузите изображение через интерфейс программы."
        return "Активировано размытие лиц. Запустите анонимизацию через интерфейс программы, убедившись, что отмечен параметр 'Обнаружение и размытие лиц'."
    
    def enable_gemini(self, args, current_file_path=None):
        """Включает интеграцию с Gemini API"""
        if not GEMINI_AVAILABLE:
            return """Gemini API недоступен. Для использования необходимо:
1. Установить библиотеку: pip install google-generativeai
2. Создать файл конфигурации с API-ключом

Подробнее: введите 'настройка gemini'"""
        
        if not self.gemini_api.is_available():
            return "Gemini API не инициализирован. Проверьте API-ключ в файле конфигурации."
        
        self.use_gemini = True
        return "Интеграция с Gemini AI активирована! Теперь я могу отвечать на общие вопросы и давать более подробные объяснения."
    
    def disable_gemini(self, args, current_file_path=None):
        """Выключает интеграцию с Gemini API"""
        self.use_gemini = False
        return "Интеграция с Gemini AI отключена. Я буду отвечать только на предопределенные команды."
    
    def list_functions(self, args, current_file_path=None):
        """Перечисляет все функции бота"""
        functions_list = """Я умею:

✓ Анонимизировать текст с персональными данными
✓ Распознавать и маскировать текст на изображениях
✓ Обнаруживать и размывать лица на фотографиях
✓ Обрабатывать PDF-документы
✓ Распознавать и маскировать:
  • ИИН (индивидуальные идентификационные номера)
  • ФИО (имена, фамилии, отчества)
  • Телефонные номера
  • Email адреса
  • Банковские карты и счета
  • Даты рождения
  • IBAN"""

        # Добавляем информацию о Gemini, если доступно
        if GEMINI_AVAILABLE and self.gemini_api.is_available():
            gemini_status = "включена" if self.use_gemini else "отключена"
            gemini_info = f"""
✓ Интеграция с Gemini AI ({gemini_status})"""
            functions_list += gemini_info
            
        functions_list += "\n\nНапишите 'помощь' для получения инструкций по использованию этих функций."
        return functions_list