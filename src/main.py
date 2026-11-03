from preprocessing import (
    carregar_dados_completos,
    ajustar_escolaridade_pais,
    ajustar_renda_familia,
    aplicar_label_encoder,
    aplicar_ordinal_encoder  # <--- Importando a nova função
)

def main():
    print("--- INICIANDO PIPELINE DE DADOS DO ENEM 2020 ---\n")
    
    # 1. Carregar e limpar colunas desnecessárias
    df = carregar_dados_completos()
    
    # 2. Ajustar Q001 e Q002 
    df = ajustar_escolaridade_pais(df)
    
    # 3. Ajustar Q006 
    df = ajustar_renda_familia(df)
    
    # 4. Aplicar o Label Encoder nas nominais (Sexo, Estados, etc.)
    df = aplicar_label_encoder(df)
    
    # 5. Aplicar o Ordinal Encoder nas demais variáveis do questionário
    df = aplicar_ordinal_encoder(df)
    
    print("\n--- Pipeline concluído com sucesso! ---")
    print(f"Dimensões finais do DataFrame processado: {df.shape}")

if __name__ == "__main__":
    main()