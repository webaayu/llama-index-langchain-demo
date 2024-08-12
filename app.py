import streamlit as st
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
# Streamlit app setup
st.title("Youtube Chat")

# Sidebar for API tokens
hf_token = st.sidebar.text_input("Enter Hugging Face Token", type="password")

# User input for places
youtube_link = st.text_input("Enter Youtube Link")
st.write(youtube_link)
if youtube_link is not None:
    with st.spinner("Loading youtube data..."):
        #documentss = loader.load_data(ytlinks=youtube_link)
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=youtube_link)
        #documentss = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=4O1rs7mrNDo"])
        st.success("Youtube data loaded successfully!")
        # Initialize the language model
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=hf_token.strip(),
            temperature=0.7,
            max_new_tokens=200
                        )
        embed_model=HuggingFaceEmbeddings()
        transformations = [SentenceSplitter(chunk_size=1024)]
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model, transformations=transformations)

        # User input for querying
        query = st.text_input("Enter your query:")
        if query:
            with st.spinner("Processing query..."):
                query_engine=index.as_query_engine(similarity_top_k=2)
                response = query_engine.query(query) 
                st.subheader("Response:")
                st.write(response)
