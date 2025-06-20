import mysql.connector

conexao = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='tcc'
)
cursor = conexao.cursor()

# Função para inserir uma execução (exemplo do tempo real)
def registrar_execucao(dados):
    query = """
        INSERT INTO execucoes (
            usuario, angulo_direito, y_punho_direito, y_ombro_direito,
            angulo_esquerdo, y_punho_esquerdo, y_ombro_esquerdo,
            classe_prevista, confianca
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, dados)
    conexao.commit()

# Função para inserir dados de treino (substituindo csv)
def salvar_dado_treinamento(dados):
    query = """
        INSERT INTO dados_treinamento (
            angulo_direito, y_punho_direito, y_ombro_direito,
            angulo_esquerdo, y_punho_esquerdo, y_ombro_esquerdo, rotulo
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, dados)
    conexao.commit()