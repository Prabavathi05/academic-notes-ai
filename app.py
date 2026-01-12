from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

def answer_with_ai(question):
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say:
"Information not available in the notes."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)


    answer = response.choices[0].message.content

    print("\nAnswer:\n", answer)
    print("\nSources:")
    for i, d in enumerate(docs,1):
        print(f"{i}. {d.page_content[:120]}...")

while True:
    q = input("\nAsk (or type exit): ")
    if q.lower() == "exit":
        break
    answer_with_ai(q)
