from preprocessing import (
    carregar_dados_completos,
    ajustar_escolaridade_pais,
    ajustar_renda_familia,
    aplicar_label_encoder,
    aplicar_ordinal_encoder,
    processar_media_e_remover_notas,
    criar_coluna_classe,
    separar_features_e_target
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

    # 6. Criar a coluna 'Nota' (média) e excluir as notas individuais
    df = processar_media_e_remover_notas(df)
    
    # 7. Criar a coluna 'Classe' baseada no quartil superior
    df = criar_coluna_classe(df)
    
    # 8. Passos finais: Excluir Nota, isolar 'y' e excluir Classe de X
    X, y = separar_features_e_target(df)
    
    print("\n--- Pipeline concluído com sucesso! ---")
    print(f"Dimensões de X (features/características): {X.shape}")
    print(f"Dimensões de y (target/alvo): {y.shape}")

    
if __name__ == "__main__":
    main()