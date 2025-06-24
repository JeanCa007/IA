from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Función para extraer texto de una URL
def extraer_texto_de_pagina(url, ruta_driver):
    # Configurar el servicio del WebDriver (en este caso, Chrome)
    servicio = Service(ruta_driver)
    driver = webdriver.Chrome(service=servicio)
    
    try:
        # Navegar a la URL proporcionada
        driver.get(url)
        
        # Esperar a que la página se cargue completamente
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        
        # Obtener el HTML de la página
        html = driver.page_source
        
        # Parsear el HTML con BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Eliminar elementos no visibles como <script> y <style>
        for elemento in soup(["script", "style"]):
            elemento.extract()
        
        # Extraer todo el texto visible
        texto = soup.get_text(separator=' ')
        
        # Limpiar el texto eliminando espacios y saltos de línea extra
        texto_limpio = ' '.join(texto.split())
        
        return texto_limpio
    
    finally:
        # Cerrar el navegador
        driver.quit()

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta al ChromeDriver 
    ruta_chromedriver = 'D:/chromedriver.exe'  # Ejemplo: 'C:/chromedriver.exe'
    
    # URL de la página que quieres extraer
    url = 'https://cnet.co.cr/'  # Cambia esto por la URL que desees
    
    # Extraer el texto
    texto_extraido = extraer_texto_de_pagina(url, ruta_chromedriver)
    
    # Imprimir el texto
    print("Texto extraído de la página:")
    print(texto_extraido)