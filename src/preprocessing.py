import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

def carregar_dados_completos(caminho_relativo="DADOS/MICRODADOS_ENEM_2020.csv"):
    
    # Função 1: Carrega o arquivo CSV completo do ENEM e remove colunas desnecessárias.

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / caminho_relativo

    print(f"Lendo o arquivo em: {DATA_PATH}")
    print("Isso pode levar alguns segundos devido ao tamanho do arquivo do ENEM...")

    # Lendo o arquivo CSV completo
    df = pd.read_csv(
        DATA_PATH,
        sep=';',
        encoding='latin-1',
        low_memory=False
    )

    print("Dados carregados com sucesso!")

    # Removendo colunas que não serão utilizadas
    labels_to_remove = [
        'NU_ANO', 'CO_MUNICIPIO_ESC', 'NO_MUNICIPIO_ESC', 'CO_UF_ESC', 
        'CO_MUNICIPIO_PROVA', 'NO_MUNICIPIO_PROVA', 'CO_UF_PROVA',
        'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 
        'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT', 
        'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT', 
        'TX_GABARITO_CN', 'TX_GABARITO_CL', 'TX_GABARITO_LC', 'TX_GABARITO_MT', 
        'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5'
    ]

    df = df.drop(columns=labels_to_remove, errors='ignore')
    return df


def ajustar_escolaridade_pais(df):
    
    # Função 2: Ajusta e agrupa as colunas Q001 (Pai) e Q002 (Mãe).
    
    # --- Ajustando Q001 (Escolaridade do Pai) ---
    if 'Q001' in df.columns:
        conditions_Q001 = [
            df['Q001'].isin(['A', 'B', 'C']),
            df['Q001'] == 'D',
            df['Q001'] == 'E',
            df['Q001'].isin(['F', 'G']),
            df['Q001'] == 'H'
        ]
        results_Q001 = ['FundamentalIncompleto', 'FundamentalCompleto', 'Médio', 'Superior', 'NaoSabe']
        df['Q001'] = np.select(conditions_Q001, results_Q001, default='Desconhecido')

    # --- Ajustando Q002 (Escolaridade da Mãe) ---
    if 'Q002' in df.columns:
        conditions_Q002 = [
            df['Q002'].isin(['A', 'B', 'C']),
            df['Q002'] == 'D',
            df['Q002'] == 'E',
            df['Q002'].isin(['F', 'G']),
            df['Q002'] == 'H'
        ]
        results_Q002 = ['FundamentalIncompleto', 'FundamentalCompleto', 'Médio', 'Superior', 'NaoSabe']
        df['Q002'] = np.select(conditions_Q002, results_Q002, default='Desconhecido')

    print("Colunas Q001 e Q002 ajustadas com sucesso!")
    return df


def ajustar_renda_familia(df):
   
    # Função 3: Ajusta e agrupa a coluna Q006 (Rendimento da família).
   
    if 'Q006' in df.columns:
        conditions_Q006 = [
            df['Q006'].isin(['A', 'B', 'C', 'D']),
            df['Q006'].isin(['E', 'F', 'G', 'H']),
            df['Q006'].isin(['H', 'I', 'J', 'K', 'L', 'M']),
            df['Q006'].isin(['N', 'O', 'P']),
            df['Q006'] == 'Q'
        ]
        results_Q006 = ['E', 'D', 'C', 'B', 'A']
        df['Q006'] = np.select(conditions_Q006, results_Q006, default='Desconhecido')
        print("Coluna Q006 ajustada com sucesso!")
        
    return df


def aplicar_label_encoder(df):
    
    # Função 4: Aplica o LabelEncoder nas colunas categóricas especificadas.

    colunas_para_codificar = [
        'TP_SEXO', 
        'SG_UF_ESC', 
        'SG_UF_PROVA', 
        'Q018', 
        'Q020', 
        'Q021', 
        'Q023', 
        'Q025'
    ]
    
    le = LabelEncoder()
    
    for coluna in colunas_para_codificar:
        if coluna in df.columns:
            df[coluna] = df[coluna].fillna('NA')
            df[coluna] = le.fit_transform(df[coluna].astype(str))
            print(f"Coluna codificada com sucesso: {coluna}")
        else:
            print(f"Aviso: A coluna {coluna} não foi encontrada no DataFrame.")
            
    return df

from sklearn.preprocessing import OrdinalEncoder

def aplicar_ordinal_encoder(df):
    
    # Função 5: Aplica o OrdinalEncoder nas variáveis socioeconômicas ordenadas do ENEM.
    
    # Lista exata das colunas ordenadas solicitadas
    colunas_ordinais = [
        'Q001', 'Q002', 'Q003', 'Q004', 
        'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 
        'Q019', 
        'Q022', 
        'Q024'
    ]
    
    # Filtramos apenas as colunas que realmente existem no DataFrame para evitar erros
    colunas_presentes = [col for col in colunas_ordinais if col in df.columns]
    
    if colunas_presentes:
        # Preenche valores nulos antes de codificar
        df[colunas_presentes] = df[colunas_presentes].fillna('NA')
        
        # Instancia o OrdinalEncoder
        oe = OrdinalEncoder()
        
        # Aplica a codificação ordinal em bloco nas colunas
        df[colunas_presentes] = oe.fit_transform(df[colunas_presentes].astype(str))
        print("OrdinalEncoder aplicado com sucesso nas variáveis!")
    else:
        print("Aviso: Nenhuma das colunas ordinais especificada foi encontrada no DataFrame.")
        
    return df