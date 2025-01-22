from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
import os
from Final import (
    extract_text_from_pdf, 
    extract_text_from_csv, 
    extract_text_from_docx,
    extract_text_from_excel,
    extract_text_from_ppt,
    extract_text_from_txt,
    chunk_text,
    get_custom_prompt_for_pdf_and_docx,
    get_custom_prompt_csv,
    get_custom_prompt_excel,
    get_custom_prompt_pptx,
    llm,
    SentenceTransformerEmbeddings,
    embedding_model,
    summarize_document
)
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

faiss_index = None
embeddings = SentenceTransformerEmbeddings(embedding_model)
@app.route('/logo192.png')
def serve_logo():
    return send_from_directory('public', 'logo192.png')


@app.route('/')
def home():
    return jsonify({'message': 'Server is running'})

@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['POST'])
async def upload_file():
    global faiss_index
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    try:
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == '.csv':
            text = extract_text_from_csv(file_path)
        elif file_extension in ['.doc', '.docx']:
            text = extract_text_from_docx(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            text = extract_text_from_excel(file_path)
        elif file_extension in ['.ppt', '.pptx']:
            text = extract_text_from_ppt(file_path)
        elif file_extension == '.txt':
            text = extract_text_from_txt(file_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400


        content_chunks = chunk_text(text)
        
        current_index = FAISS.from_texts(
            texts=content_chunks,
            embedding=embeddings,
            metadatas=[{"source": file.filename, "index": i} for i in range(len(content_chunks))]
        )
        
        if faiss_index is None:
            faiss_index = current_index
        else:
            faiss_index.merge_from(current_index)

        summary = await summarize_document(file_path)
        return jsonify({
            'summary': summary,
            'message': 'File processed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/query', methods=['POST', 'OPTIONS'])
async def handle_query():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    if faiss_index is None:
        return jsonify({
            'error': 'Hey! Upload a document first to get started with your questions.',
            'type': 'warning',
            'title': 'Document Required'
        }), 400
        
    data = request.json
    query = data.get('query')
    query_time = datetime.now()
    
    if not query:
        return jsonify({
            'error': 'Ready to help! Just type your question in the text box.',
            'type': 'warning',
            'title': 'Question Required'
        }), 400

    try:
        print(f"Processing query: {query}")
        retriever = faiss_index.as_retriever(search_kwargs={"k": 5, "fetch_k": 10})
        
        docstore_key = list(faiss_index.docstore._dict.keys())[0]
        file_source = faiss_index.docstore._dict[docstore_key].metadata['source']
        file_extension = os.path.splitext(file_source)[1].lower()
        
        if file_extension == '.csv':
            custom_prompt = get_custom_prompt_csv()
        elif file_extension in ['.xlsx', '.xls']:
            custom_prompt = get_custom_prompt_excel()
        elif file_extension in ['.ppt', '.pptx']:
            custom_prompt = get_custom_prompt_pptx()
        else:
            custom_prompt = get_custom_prompt_for_pdf_and_docx()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": custom_prompt,
                "verbose": True
            }
        )
        
        result = await qa_chain.ainvoke({"query": query})
        response_time = datetime.now()
        
        if not result or 'result' not in result:
            return jsonify({"error": "No response generated"}), 500
            
        return jsonify({
            "response": result['result'],
            "query_timestamp": query_time.strftime("%I:%M:%S %p"),
            "response_timestamp": response_time.strftime("%I:%M:%S %p")
        })
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

app.run(host='0.0.0.0', port=5000, debug= False)
