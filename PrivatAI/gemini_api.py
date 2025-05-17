"""
Модуль для интеграции с Google Gemini API
Позволяет улучшить ответы чат-бота с помощью искусственного интеллекта
"""

import os
import json

# Проверка наличия установленной библиотеки google.generativeai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiAPI:
    """Класс для работы с Gemini API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        self.initialized = False
        
        # Пытаемся загрузить API-ключ из файла конфигурации
        if not api_key:
            try:
                config_path = os.path.join(os.path.dirname(__file__), 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'gemini_api_key' in config:
                            self.api_key = config['gemini_api_key']
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {str(e)}")
        
        # Инициализация API, если ключ доступен и библиотека установлена
        if self.api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                
                # Получаем доступ к модели
                self.model = genai.GenerativeModel('gemini-pro')
                self.initialized = True
            except Exception as e:
                print(f"Ошибка инициализации Gemini API: {str(e)}")
    
    def is_available(self):
        """Проверяет, доступен ли API"""
        return self.initialized
    
    def get_response(self, prompt, context=None):
        """Получает ответ от Gemini API"""
        if not self.initialized:
            return "Gemini API не инициализирован. Проверьте API-ключ."
        
        try:
            # Формируем промпт для API с контекстом, если он предоставлен
            full_prompt = prompt
            if context:
                full_prompt = f"Контекст: {context}\n\nЗапрос: {prompt}"
            
            # Получаем ответ от модели
            response = self.model.generate_content(full_prompt)
            
            # Возвращаем текст ответа
            return response.text
        except Exception as e:
            return f"Ошибка при обращении к Gemini API: {str(e)}"
    
    @staticmethod
    def get_installation_instructions():
        """Возвращает инструкции по установке и настройке Gemini API"""
        return """
Для использования Google Gemini API необходимо:

1. Установить библиотеку:
   pip install google-generativeai

2. Получить API-ключ:
   - Создайте аккаунт на Google AI Studio (https://makersuite.google.com/)
   - Получите API-ключ в разделе API Keys

3. Создайте файл config.json в папке с программой:
   {
     "gemini_api_key": "AIzaSyBV4RtmbkzZzfWTw7XPUYmgf0-86Ua5xgM"
   }

После выполнения этих шагов, чат-бот сможет использовать Gemini API для улучшения ответов.
"""


# Пример использования:
if __name__ == "__main__":
    # Для тестирования модуля
    api = GeminiAPI()
    if api.is_available():
        response = api.get_response("Что такое анонимизация данных?")
        print(response)
    else:
        print("Gemini API недоступен.")
        print(GeminiAPI.get_installation_instructions())