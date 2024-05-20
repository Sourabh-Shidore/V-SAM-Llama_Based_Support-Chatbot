

############################## Installing Libraries ##########################################################

# pip install pypdf
# pip install -q transformers einops accelerate langchain bitsandbytes
# pip install install sentence_transformers
# pip install llama-index
# pip install llama-index-llms-huggingface


################################## Importing libraries ############################################################
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
# from google.colab import drive
# drive.mount("/content/drive")
from langchain.document_loaders import PyPDFLoader
import torch


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
# from llama_index.embeddings import LangchainEmbedding
from llama_index.legacy.embeddings.langchain import LangchainEmbedding


from flask import Flask, request, jsonify, render_template, redirect, url_for
# from utils import MedicalInsurence
# import config
import traceback


###################################Loading Document ##################################################################
loader = PyPDFLoader(r"D:\my project\chat bot\lamma2\Front-end\user manual.pdf")
documents=SimpleDirectoryReader(r"D:\my project\chat bot\lamma2\Front-end").load_data()


##################################3 Defining prompt for bot ####################################################
system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


################################### Login into Hugging face ###############################################
# huggingface-cli login


################################## Importing models #############################################
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":False}
)


embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))


############################################ creating service content and document vector ###############################

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index=VectorStoreIndex.from_documents(documents,service_context=service_context)

############################################ creating a quary engine ##########################################################
query_engine=index.as_query_engine()


############################################## Flask application for chatbot ############################################

@app.route('/chatbot')
def home1():
    
    return render_template('medical_insurence.html')

@app.route('/assist_the_customer', methods = ['GET', 'POST'])
def assist_customer():
    try:
        if request.method == 'GET':
            print("+"*50)
            data = request.args.get
            print("Data :",data)
            query_1 = data('query')

            if query_1 in  ['Hi', 'hello','namste']:
                response = r'Hello Welcome to DEll Vertual Technical support machine \n How can I help You?'

            else:
                response=query_engine.query(query_1)
            
            # return jsonify({"Result":f"Predicted Medical Charges == {pred_price}"})
            return render_template('medical_insurence.html', prediction = response)

        elif request.method == 'POST':
            print("+"*50)
            data = request.args.get
            print("Data :",data)
            query_1 = data('query')

            if query_1 in  ['Hi', 'hello','namste']:
                response = r'Hello Welcome to DEll Vertual Technical support machine \n How can I help You?'

            else:
                response=query_engine.query(query_1)
            
            # return jsonify({"Result":f"Predicted Medical Charges == {pred_price}"})
            return render_template('medical_insurence.html', prediction = response)

    except:
        print(traceback.print_exc())
        return redirect(url_for('medical_insurence'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004)