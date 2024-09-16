import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Carregar variáveis de ambiente
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Ler o arquivo JSON
data = pd.read_json("dados_dataset.json")

# Remover duplicatas
data = data.drop_duplicates(subset=['product_name'])

# Setar valores ausentes para 'Informação não disponível'
data = data.fillna("Informação não disponível")

# Função de formatação para os dados do review
def format_review(row):
    return (
        f"Data de Submissão: {row.get('submission_date', 'Não disponível')}\n"
        f"ID do Revisor: {row.get('reviewer_id', 'Não disponível')}\n"
        f"ID do Produto: {row.get('product_id', 'Não disponível')}\n"
        f"Nome do Produto: {row.get('product_name', 'Não disponível')}\n"
        f"Marca do Produto: {row.get('product_brand', 'Não disponível')}\n"
        f"Categoria do Site LV1: {row.get('site_category_lv1', 'Não disponível')}\n"
        f"Categoria do Site LV2: {row.get('site_category_lv2', 'Não disponível')}\n"
        f"Título da Revisão: {row.get('review_title', 'Não disponível')}\n"
        f"Avaliação Geral: {row.get('overall_rating', 'Não disponível')}\n"
        f"Recomendaria a um Amigo: {row.get('recommend_to_a_friend', 'Não disponível')}\n"
        f"Texto da Revisão: {row.get('review_text', 'Não disponível')}\n"
        f"Ano de Nascimento do Revisor: {row.get('reviewer_birth_year', 'Não disponível')}\n"
        f"Gênero do Revisor: {row.get('reviewer_gender', 'Não disponível')}\n"
        f"Estado do Revisor: {row.get('reviewer_state', 'Não disponível')}\n"
    )

# Gerar documentos com base nas 1000 primeiras linhas do dataset
documents = [Document(page_content=format_review(row)) for _, row in data.head(10).iterrows()]

# Imprimir documentos para depuração
print("Documentos gerados:")
for doc in documents:
    print(doc.page_content)
    print('-' * 80)

# Dividir os documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

# Imprimir chunks para depuração
unique_chunks = []
seen_texts = set()
for chunk in text_chunks:
    if chunk.page_content not in seen_texts:
        seen_texts.add(chunk.page_content)
        unique_chunks.append(chunk)

# Criar embeddings e vetorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db", 
)

# Adicionar documentos ao Chroma
vectorstore.add_documents(documents)  

# Teste a recuperação
retriever = vectorstore.as_retriever(search_kwargs={"k": len(unique_chunks)})

print("Chunks após remover duplicatas:")
for chunk in unique_chunks:
    print(chunk.page_content)
    print('-' * 80)

print(f"Total de chunks criados: {len(text_chunks)}")
print(f"Total de chunks únicos: {len(unique_chunks)}")

# Definir template e prompt
template = """Você é um assistente treinado para responder perguntas com base nas seguintes informações:
- Data de Submissão
- ID do Revisor
- ID do Produto
- Nome do Produto
- Marca do Produto
- Categoria do Site LV1
- Categoria do Site LV2
- Título da Revisão
- Avaliação Geral
- Recomendaria a um Amigo
- Texto da Revisão
- Ano de Nascimento do Revisor
- Gênero do Revisor
- Estado do Revisor

Use os pedaços de texto retornados como base para suas respostas. Se a informação não estiver disponível ou não souber a resposta, diga "Não é Possível Retornar esse Dado".

Pergunta: {question}
Contexto: {context}
Resposta:
"""

prompt = PromptTemplate.from_template(template)
output_parser = StrOutputParser()

# Inicializar modelo e cadeia de processamento
llm_model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.2)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

# Loop para receber e processar perguntas do usuário
while True:
    user_question = input("Digite sua Pergunta (ou 'Sair' para encerrar): ")
    
    # Verificar se o usuário deseja sair
    if user_question.strip().lower() == 'sair':
        print("Saindo do programa.")
        break

    print("Pergunta recebida:", user_question)
    try:
        responseQuestion = rag_chain.invoke(user_question)
        print("Resposta gerada:", responseQuestion)
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")
    print('-' * 80)
