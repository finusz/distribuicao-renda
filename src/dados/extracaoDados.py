import io
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.util.demais_funcoes import configurar_log

#------------ LOGS ------------

PASTA_LOGS = Path(__file__).parent / "logs"
PASTA_LOGS.mkdir(parents=True, exist_ok=True)

logger = configurar_log('extracao',
                        PASTA_LOGS,
                        'extracao.log')

#------------ CONFIGURAÇÕES ------------

URL = "https://dados.gov.br/dados/conjuntos-dados/distribuio-de-renda"

TITULO_RECURSO = "Distribuição da renda tributável, dos sócios, dividendos e tributada exclusivamente na fonte"

CAMINHO_DADOS = Path(__file__).parent / "distribuicao_renda.csv"    


def salvar_screenshot(driver, nome="debug"):
    """Salva um screenshot com data/hora na pasta de logs e registra o caminho."""
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = PASTA_LOGS / f"{nome}_{carimbo}.png"
    if driver.save_screenshot(str(caminho)):
        logger.error(f"Screenshot salvo em: {caminho}")
    else:
        logger.error("Não foi possível salvar o screenshot.")
    return caminho


def criar_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    logger.info('Driver criado')
    return webdriver.Chrome(options=options)


def expandir_aba_recursos(driver, wait):
    aba = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.botao-collapse-Recursos"))
    )
    if aba.get_attribute("aria-expanded") == "false":
        aba.click()
    wait.until(EC.visibility_of_element_located((By.ID, "collapse-recursos")))
    logger.info("Aba 'Recursos' expandida.")


def obter_url_do_recurso(driver, wait, titulo):
    xpath = (
        f"//div[contains(@class,'col-10')][normalize-space(h4)='{titulo}']"
        "//button[@id='btnDownloadUrl']"
    )
    botao = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))

    driver.execute_script(
        """
        window.__urls = [];
        window.open = function (u) { window.__urls.push(u); return null; };
        """
    )
    abas_antes = len(driver.window_handles)
    url_antes = driver.current_url

    driver.execute_script("arguments[0].click();", botao)

    try:
        wait.until(
            lambda d: d.execute_script("return window.__urls.length") > 0
            or len(d.window_handles) > abas_antes
            or d.current_url != url_antes
        )
    except TimeoutException:
        logger.error(
            "O clique não gerou URL (nem window.open, nem nova aba, nem navegação). "
            "Verifique a aba Network do DevTools para ver como o download é disparado."
        )
        salvar_screenshot(driver)
        raise

    urls = driver.execute_script("return window.__urls")
    if urls:
        url_download = urls[0]
        logger.info("URL capturada via window.open.")
    elif len(driver.window_handles) > abas_antes:
        driver.switch_to.window(driver.window_handles[-1])
        url_download = driver.current_url
        logger.info("URL capturada em nova aba.")
    else:
        url_download = driver.current_url
        logger.info("URL capturada via navegação na mesma aba.")

    logger.info(f"URL do recurso: {url_download}")
    return url_download


def baixar_csv(driver, url_download):
    session = requests.Session()
    for cookie in driver.get_cookies():
        session.cookies.set(cookie["name"], cookie["value"])

    user_agent = driver.execute_script("return navigator.userAgent;")
    session.headers.update({"User-Agent": user_agent})

    resposta = session.get(url_download, timeout=60)

    if resposta.status_code != 200:
        logger.error(f"Erro ao capturar dados. Status: {resposta.status_code}")
        return None

    # Lê tudo como texto: a conversão para número é feita em limpezaDados (_para_float).
    # Não usar thousands=".": o Centil tem subdivisões como "100.1" e "100.10.1",
    # que seriam lidas como 1001 e 1001010, e as subdivisões passariam a contar como centis normais.
    df = pd.read_csv(io.BytesIO(resposta.content), sep=";", encoding="utf-8-sig", dtype=str)
    logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    logger.info("\n" + df.head().to_string())
    return df


def main():
    """Faz a coleta no portal e devolve o DataFrame bruto."""
    driver = criar_driver()
    try:
        driver.get(URL)
        wait = WebDriverWait(driver, 15)

        expandir_aba_recursos(driver, wait)
        url_download = obter_url_do_recurso(driver, wait, TITULO_RECURSO)
        return baixar_csv(driver, url_download)
    except Exception:
        logger.exception("Falha na extração")
        raise
    finally:
        driver.quit()


def extrair_e_salvar():
    """Roda a coleta e grava o resultado em data/distribuicao_renda.csv."""
    df = main()
    if df is None:
        raise RuntimeError("A extração não retornou dados; arquivo não foi salvo.")

    CAMINHO_DADOS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_DADOS, sep=";", index=False, encoding="utf-8-sig")
    logger.info(f"Dados salvos em: {CAMINHO_DADOS}")
    return df


if __name__ == "__main__":
    extrair_e_salvar()