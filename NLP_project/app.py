import dash
from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output, State
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import os 


# Load environment variables
load_dotenv()


# Get environment variables
csv_file_path = os.getenv("QANDA_CSV_PATH")
gardenmap_html_path = os.getenv("GARDENMAP_HTML_PATH")
mymap_html_path = os.getenv("MYMAP_HTML_PATH")

# Check if environment variables are set
if csv_file_path is None:
    raise ValueError("QANDA_CSV_PATH environment variable is not set")
if gardenmap_html_path is None:
    raise ValueError("GARDENMAP_HTML_PATH environment variable is not set")
if mymap_html_path is None:
    raise ValueError("MYMAP_HTML_PATH environment variable is not set")

# Check if files exist
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"CSV file '{csv_file_path}' not found")
if not os.path.exists(gardenmap_html_path):
    raise FileNotFoundError(f"Garden map HTML file '{gardenmap_html_path}' not found")
if not os.path.exists(mymap_html_path):
    raise FileNotFoundError(f"My map HTML file '{mymap_html_path}' not found")




loader = CSVLoader(file_path=csv_file_path)
documents = loader.load()


# Embeddings
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# Running similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=1)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Set up LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a resource to residents living in Los Angeles who hope to start their own community gardens using laws specific to LA
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practices,
in terms of length, tone of voice, logical arguments, and other details

2/ If the best practices are irrelevant, then try to mimic the style of the best practice to the prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practices of how we normally respond to prospects in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
there is no need to restate the question in the answer
If asked for contact info, refer them to Los Angeles Community Garden Council, admin@lacommunitygardens.org, 323-902-7167, 1110 N Virgil Avenue #381
Los Angeles, CA  90029

"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate chatbot response
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message = message, best_practice=best_practice)
    return response

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout of Dash app
# Define layout of Dash app
# Define layout of Dash app
# Define layout of Dash app
# Define layout of Dash app
app.layout = html.Div([
    # Sidebar with chatbot
    html.Div([
        html.H1("Los Angeles Community Garden Chatbot"),
        dcc.Textarea(id="input-message", placeholder="Ask a Question... Try: What if I dont own my land? Can I still start a garden?", style={'width': '100%', 'height': '200px'}),
        html.Button("Submit", id="submit-button", n_clicks=0),
        html.Div(id="output-response", style={'margin-top': '20px'}),
    ], style={'position': 'fixed', 'top': '0', 'bottom': '0', 'left': '0', 'width': '300px', 'background': 'linear-gradient(to right, #ADC98A, white)', 'padding': '20px', 'overflow-y': 'auto'}),
    
    # Main content area with maps
    html.Div([
        # Map with title text for finding local community garden
        html.Div([
            html.H2("Find a local community garden"),
            html.Iframe(id='map2', srcDoc=open(gardenmap_html_path, 'r').read(), width='100%', height='800px')
        ], style={'margin': 'auto', 'width': '80%', 'display': 'block', 'padding-top': '20px'}),

        # Map with title text for potential garden spots
        html.Div([
            html.H2("Potential Garden Spots (unoccupied land for sale)"),
            html.Iframe(id='map1', srcDoc=open(mymap_html_path, 'r').read(), width='100%', height='800px')
        ], style={'margin': 'auto', 'width': '80%', 'display': 'block', 'padding-top': '20px', 'margin-bottom': '50px'})
    ], style={'margin-left': '300px', 'padding-left': '20px'})  # Adjust margin-left to account for sidebar width
])


# Define callback to generate chatbot response
@app.callback(
    Output("output-response", "children"),
    [Input("submit-button", "n_clicks")],
    [State("input-message", "value")]
)
def update_output(n_clicks, message):
    if n_clicks > 0:
        response = generate_response(message)
        return response

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)

