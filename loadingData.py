import os
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

data = pd.read_json("dados_dataset.json")

data = data.fillna("Informação não disponível")

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

documents = [Document(page_content=format_review(row)) for _, row in data.head(10000).iterrows()]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

unique_chunks = []
seen_texts = set()
for chunk in text_chunks:
    if chunk.page_content not in seen_texts:
        seen_texts.add(chunk.page_content)
        unique_chunks.append(chunk)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(unique_chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print(f"Total de chunks criados: {len(text_chunks)}")
print(f"Total de chunks únicos: {len(unique_chunks)}")

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
- Recomendaria para compra, para outros clientes

Use os pedaços de texto retornados como base para suas respostas. Se a informação não estiver disponível ou não souber a resposta, diga "Não é Possível Retornar esse Dado".

Pergunta: {question}
Contexto: {context}
Resposta:
"""

prompt = PromptTemplate.from_template(template)
output_parser = StrOutputParser()

llm_model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.2)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

while True:
    user_question = input("Digite sua Pergunta (ou 'SAIR' para encerrar): ")

    if user_question.strip().upper() == "SAIR":
        print("Encerrando o programa.")
        break

    print("Pergunta recebida:", user_question)
    
    responseQuestion = rag_chain.invoke(user_question)
    
    print("Resposta gerada:", responseQuestion)

