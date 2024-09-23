import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Carregar o DataFrame
df = pd.read_excel("result.xlsx")

# Remover o índice
df.reset_index(drop=True, inplace=True)
print(df)

# Adicionar coluna 'ID' substituindo 'IP Address'
df.drop(columns=['IP Address'], inplace=True)


# Formatar 'Added Time' para 'dd/mm/aa - hh/mm/ss'
df['Added Time'] = pd.to_datetime(df['Added Time']).dt.strftime('%d/%m/%y - %H:%M:%S')

# Filtrar por idade maior que 18
df_filtrado = df[df['idade'] > 18]


# Ordenar por nome
df_ordenado = df.sort_values(by='Nome')


# Codificação da coluna 'Nome'
le_nome = LabelEncoder()
df['Nome Codificado'] = le_nome.fit_transform(df['Nome'])

# Codificação da coluna 'cor preferida'
le_cor = LabelEncoder()
df['Cor Preferida Codificada'] = le_cor.fit_transform(df['cor preferida'])

# Seleção das características (features) e do alvo (target)
X = df[['idade', 'Nome Codificado']]
y = df['Cor Preferida Codificada']

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificação de valores nulos
if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
    raise ValueError("Os dados de treino contêm valores nulos.")

# Ajuste de hiperparâmetros com GridSearchCV
tamanho_dados = len(df)
if tamanho_dados < 1000:
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
else:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 20, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Previsão e avaliação do modelo
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Identificar a cor mais escolhida
cor_mais_escolhida_codificada = df['Cor Preferida Codificada'].mode()[0]
cor_mais_escolhida = le_cor.inverse_transform([cor_mais_escolhida_codificada])[0]

# Filtrar os dados para a cor mais escolhida
df_filtrado = df[df['Cor Preferida Codificada'] == cor_mais_escolhida_codificada]

# Contar as ocorrências por idade
ocorrencias_por_idade = df_filtrado['idade'].value_counts()

# Idade com mais ocorrências
idade_mais_comum = ocorrencias_por_idade.idxmax()
ocorrencias = ocorrencias_por_idade.max()

print(f'A cor mais escolhida é: {cor_mais_escolhida}')
print(f'A idade com mais ocorrências da cor mais escolhida é: {idade_mais_comum} com {ocorrencias} ocorrências')

# Exemplo de previsão com novos dados
novo_dado = pd.DataFrame({'idade': [20], 'Nome Codificado': [le_nome.transform(['Joao, Vitor'])[0]]})
cor_prevista_codificada = best_model.predict(novo_dado)
cor_prevista = le_cor.inverse_transform(cor_prevista_codificada)
print(f'Cor prevista: {cor_prevista[0]}')


#-------------------------------------------------------------------------

# Estilizar a tabela
styled_table = df.style.background_gradient(cmap='viridis')

# Renderizar a tabela estilizada e salvar como PNG
fig, ax = plt.subplots(figsize=(20, 12))  # Aumentar o tamanho da figura
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=styled_table.data.values, colLabels=styled_table.data.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)  # Aumentar o tamanho da fonte
table.scale(2, 2)  # Aumentar a escala da tabela
plt.savefig('tabela_estilizada.png', bbox_inches='tight')
plt.close(fig)


# Renderizar a tabela ordenada e salvar como PNG
fig, ax = plt.subplots(figsize=(20, 12))  # Aumentar o tamanho da figura
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_ordenado.values, colLabels=df_ordenado.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)  # Aumentar o tamanho da fonte
table.scale(2, 2)  # Aumentar a escala da tabela
plt.savefig('tabela_ordenada.png', bbox_inches='tight')
plt.close(fig)



# Renderizar a tabela filtrada e salvar como PNG
fig, ax = plt.subplots(figsize=(20, 12))  # Aumentar o tamanho da figura
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_filtrado.values, colLabels=df_filtrado.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)  # Aumentar o tamanho da fonte
table.scale(2, 2)  # Aumentar a escala da tabela
plt.savefig('tabela_filtrada.png', bbox_inches='tight')
plt.close(fig)


#-------------------------------------------------------------------------


# Criar o gráfico de ocorrências por idade
plt.figure(figsize=(10, 6))
ocorrencias_por_idade.sort_index().plot(kind='bar', color='skyblue')
plt.title('Ocorrências por Idade')
plt.xlabel('Idade')
plt.ylabel('Número de Ocorrências')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()

# Salvar o gráfico como imagem
plt.savefig('ocorrencias_por_idade.png')
plt.close(fig)

print("Gráfico de ocorrências por idade criado com sucesso!")

# Contar as ocorrências de cada cor
ocorrencias_por_cor = df['Cor Preferida Codificada'].value_counts()
ocorrencias_por_cor.index = le_cor.inverse_transform(ocorrencias_por_cor.index)

# Criar o gráfico de ocorrências por cor
plt.figure(figsize=(10, 6))
ocorrencias_por_cor.plot(kind='bar', color='lightcoral')
plt.title('Ocorrências por Cor Preferida')
plt.xlabel('Cor')
plt.ylabel('Número de Ocorrências')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()

# Salvar o gráfico como imagem
plt.savefig('ocorrencias_por_cor.png')
plt.close(fig)

print("Gráfico de ocorrências por cor criada com sucesso!")

#--------------------------------------------------------------------
# Criar a figura
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Adicionar texto à figura
texto = f'Acurácia do modelo: {accuracy * 100:.2f}%\nCor prevista para o exemplo: {cor_prevista[0]}'
ax.text(0.5, 0.5, texto, ha='center', va='center', fontsize=20)

# Salvar a figura como PNG
plt.savefig('resultado_modelo.png', bbox_inches='tight')
plt.close(fig)

#---------------------------------------------------------------------

# Remover colunas codificadas
df_final = df.drop(columns=['Nome Codificado', 'Cor Preferida Codificada'])

# Criar a figura
fig, ax = plt.subplots(figsize=(12, 8))  # Ajuste o tamanho conforme necessário
ax.axis('tight')
ax.axis('off')

# Adicionar a tabela à figura
tbl = ax.table(cellText=df_final.values, colLabels=df_final.columns, cellLoc='center', loc='center')

# Ajustar o estilo da tabela
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)

# Salvar a figura como PNG
plt.savefig("tabela_completa.png", bbox_inches='tight', dpi=300)
plt.close(fig)