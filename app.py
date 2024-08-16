# Import required libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import openai
import os
import re
import configparser
import io
import base64
import requests
from datetime import timedelta
import json
import time


# Configuration

# Initialize the ConfigParser
global config
config = configparser.ConfigParser()

# Read the config file
config.read('config.ini')

def save_figure(figure: go.Figure, file):
    return figure.write_image(file, format='pdf')



def llm(prompt,system,model="GPT-4",temperature=0,max_new_tokens=512):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_new_tokens)

    return response.choices[0].message.content


def call_vision_api(image_base64, model="vision", max_tokens=512):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant whose job is to summarize images in detail."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the following image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

def conversation(messages,model,temperature=0,max_new_tokens=512, image=None):    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens)
    return response.choices[0].message.content


def format_conversation(messages):
    formatted_messages = []
    for message in messages:
        if message['role'] == 'system':
            continue  # Skip system messages

        # Determine the alignment and text class based on the role
        if message['role'] == 'user':
            alignment_class = "user-message"
            text_class = "user-text"
        else:
            alignment_class = "Chappy-message"
            text_class = "Chappy-text"

        formatted_messages.append({
            'alignment_class': alignment_class,
            'text_class': text_class,
            'content': message['content']
        })
    return formatted_messages

def display_chat(formatted_messages):
    st.markdown("""
        <style>
            .message-box {
                display: flex;
                flex-direction: column;
                align-items: flex-start; /* Default alignment for Chappy */
                margin-bottom: 20px;
                width: 100%; /* Adjust the width */
            }
            .message-box.user {
                align-items: flex-end; /* Alignment for user */
            }
            .message-text {
                border-radius: 20px;
                color: white;
                display: inline-block;
                word-wrap: break-word;
                max-width: 80%; /* Adjust the max width */
                padding: 10px 15px;
            }
            .message-text a { /* Targeting links within message-text */
                color: #f934b9; /* Light pink color for links */
                text-decoration: none; /* Optional: removes underline from links */
            }
            .message-text a:hover { /* Optional: Change color on hover */
                color: #fb7dd2; /* A darker pink for hover effect */
            }
            .user-text {
                background-color: #8913db;
            }
            .Chappy-text {
                background-color: #0084ff;
            }
            .message-label {
                font-size: 0.8em;
                color: #666;
                margin-bottom: 2px;
            }
        </style>
    """, unsafe_allow_html=True)

    for message in formatted_messages:
        label = "You" if message['alignment_class'] == 'user-message' else "Chappy"

        #if message['content'].endswith("```"):
        #    message['content'] = message['content'] + "\n"

        # Isolate the first and last line
        first_line = message['content'].split('\n', 1)[0]
        last_line = message['content'].split('\n')[-1]
        # Regex pattern to match common Markdown syntax in the first line
        # Adjust the pattern as needed to match specific Markdown elements you're interested in
        markdown_pattern = r'(\#{1,6}\s|>|`{3,}|[*_]{1,2}|!\[.*?\]\([^)]*?\)|\[.*?\]\([^)]*?\))'

        if re.search(markdown_pattern, first_line):
            message['content'] = "\n\n" + message['content']

        if re.search(markdown_pattern, last_line):
            message['content'] =  message['content'] + "\n"

        st.markdown(f"<div class='message-box {message['alignment_class']}'><div class='message-label'>{label}</div><div class='message-text {message['text_class']}'>{message['content']}</div></div>", unsafe_allow_html=True)


def display_editable_conversation(conversation):
    """Display the conversation for editing, including the system message as the first element."""
    edited_conversation = []
    for idx, message in enumerate(conversation):
        user_label = "System" if message['role'] == 'system' else ("You" if message['role'] == 'user' else "AI")
        # Use a read-only field for the system message to prevent editing
        if message['role'] == 'system':
            edited_message = st.text_area(f"{user_label} (Message {idx+1}):", value=message['content'], disabled=True, key=f"message_{idx}")
        else:
            edited_message = st.text_area(f"{user_label} (Message {idx+1}):", value=message['content'], key=f"message_{idx}")
        edited_conversation.append({"role": message['role'], "content": edited_message})
    return edited_conversation

def process_edited_conversation(edited_conversation):
    """Process the edited conversation."""
    new_conversation = []
    for message in edited_conversation:
        if message['content'].strip():  # Ensure that empty messages are not included
            new_conversation.append(message)
    return new_conversation

def export_conversation(conversation):
    """Converts the conversation history to a text format, including system prompts."""
    conversation_text = ""
    for message in conversation:
        if message['role'] == 'system':
            prefix = "*_* System: "
        elif message['role'] == 'user':
            prefix = "*_* You: "
        else:  # Assuming the only other role is 'assistant'
            prefix = "*_* Chappy: "
        conversation_text += prefix + message['content'] + "\n\n"
    return conversation_text


def update_system_prompt():
    new_system_prompt = st.session_state.new_system_prompt
    if new_system_prompt:
        # Check if the first message is a system prompt and update it
        if 'conversation' in st.session_state and st.session_state['conversation']:
            if st.session_state['conversation'][0]['role'] == 'system':
                st.session_state['conversation'][0]['content'] = new_system_prompt
            else:
                st.session_state['conversation'].insert(0, {"role": "system", "content": new_system_prompt})
        else:
            st.session_state['conversation'] = [{"role": "system", "content": new_system_prompt}]

def get_image_base64(image):
    base64_image = base64.b64encode(image.read()).decode('utf-8')
    return base64_image



def parse_time(time_str):
    # Convert time remaining into total seconds
    parts = time_str.split(", ")
    minutes = int(parts[0].split()[0])
    seconds = int(parts[1].split()[0])
    total_seconds = minutes * 60 + seconds
    return total_seconds

def format_time(seconds, include_remaining=True):
    # Format seconds into "minutes, seconds"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if include_remaining:
        return f"{minutes} minutes, {remaining_seconds} seconds remaining"
    else:
        return f"{minutes} minutes, {remaining_seconds} seconds"

def unload_model(model_id):
    url = f"http://localhost:5000/unload_model"
    data = {"model_id": model_id}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            st.success(f"Model {model_id} unloaded successfully!")
            time.sleep(3)
            # Refresh the page to update the status
            st.experimental_rerun()
        else:
            st.error(f"Failed to unload model {model_id}: {response.text}")
    except Exception as e:
        st.error(f"Error unloading model {model_id}: {e}")



# Setting Page Title, Page Icon and Layout Size
st.set_page_config(
    page_title='Chappy',
    page_icon="logoF.png",
    layout='wide',
    initial_sidebar_state="expanded",
    menu_items={
    'Get Help': 'https://github.com/tatonetti-lab/chappy',
    'Report a bug': "https://github.com/tatonetti-lab/chappy/issues",
    'About': "# This is a header. This is about *chappy* !"
}

)

st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)


def main():
    """Main function to run the Streamlit application."""

    with st.sidebar:
        # Sidebar Configuration
        st.title("Version **1.1.0**")
        st.empty()
        # Add a title to instruct users to leave feedback
        st.write('**We value your feedback!**')

        link = 'https://github.com/tatonetti-lab/chappy/issues'
        st.markdown(f"<div style='text-align: center;'><a href='{link}' target='_blank' style='display: inline-block; background-color: #24292e; color: white; padding: 8px 16px; text-align: center; border-radius: 25px; text-decoration: none; border: 2px solid #FFFFFF; '>Leave Feedback on GitHub</a>", unsafe_allow_html=True)
        st.divider()
        st.title("Settings")

        # API Configuration
        client_type = st.selectbox('Select Client Type:', ['openai', 'azure'])
        api_key = st.text_input('API Key:', type='password')
        if client_type == 'azure':
            azure_endpoint = st.text_input('Azure Endpoint:')
            api_version = "2023-12-01-preview"

        # Initialize the appropriate client based on the client type from the sidebar input
        global client
        if client_type == 'openai':
            client = openai.OpenAI(api_key=api_key)
        elif client_type == 'azure':
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        else:
            st.error("Unsupported client type specified")

        # Model selection
        model = st.text_input('Enter model name:')
        if not model:
            st.warning("Please enter a model name.")
        else:
            st.title(f"Using: {model}")
        st.divider()
        st.header("Model Parameters")
        # Get user input for model parameters and provide validation
        new_tokens = st.number_input("New Tokens (number of new tokens to be generated)", value=2048)
        temperature = st.select_slider("Temperature (randomness of response)", options=[float(f"{i/100:.2f}") for i in range(1, 101)], value=0.10)
        try:
            new_tokens = int(new_tokens)
        except ValueError:
            st.warning("Please enter a valid integer for new tokens.")
        try:
            temperature = float(temperature)
            if temperature < 0 or temperature > 1:
                raise ValueError
        except ValueError:
            st.warning("Please enter a value between 0 and 1.")
            



    # Create a two-column layout
    col1, col2 = st.columns([3, 1]) # this will just call methods directly in the returned objects

    # Inside the first column, add the answer text
    with col1:
        # Main Application Content
        st.title('Chappy')
        st.subheader('DEMO VERSION')
        functionality=st.radio('Select from the following:', ['Chat','Quick Analysis Script Writer', 'Graphic Generation'])

    # Inside the second column, add the image
    with col2:
        st.image("logo.png", use_column_width=True)

    if (functionality == 'Quick Analysis Script Writer'):
        top_k = st.sidebar.number_input("Top k (number of observations to base data summary on)", value=5)
        display_data=st.sidebar.toggle("Display csv?", value=True)
    
        st.subheader('Please upload your CSV file and enter your request below:')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Display details of the uploaded file
            file_details = {
                "FileName": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": uploaded_file.size
            }
            st.write(file_details)

            # Save the uploaded file to a temporary location and read its content
            file_path = os.path.join("./data/tmp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            df = pd.read_csv(file_path)
            if display_data:
                st.dataframe(df)

            # Get or set session states for CSV description and suggested queries
            if "csv_description" not in st.session_state:
                st.session_state.csv_description = ""
            st.subheader("Enter a brief description of the CSV, including any relevant columns.")

            # Auto-fill button to generate a dataset summary
            if st.button("Auto-Fill"):
                column_data_types = df.dtypes
                columns_info = '\n'.join([f"Column Name: {col}\nData Type: {dtype}" for col, dtype in column_data_types.items()])
                st.session_state.csv_description =  llm(
                    prompt = ("The dataset head is: " + df.head(top_k).to_string() + "\n\n" + columns_info),
                    system = f"""         
                        You are a bot whose purpose is to summarize the data format based upon the column names, data types, and dataset head. The head is only for example for you to understand the structure, do not use these values in the response. Return nothing except the formatting. Write your answer for the dataset descriptions in the exact format:     
                            Column Name: [Column1 Name]
                            Data Type: [Type, e.g., Integer, String, Date]
                            Description: [Brief summary of what this column contains, any special notes about its content.]

                            Column Name: [Column2 Name]
                            Data Type: [Type, e.g., Integer, String, Date]
                            Description: [Brief summary of what this column contains, any special notes about its content.]
                            [... Repeat for all columns ...] """,
                    model=model,
                    temperature=temperature,
                    max_new_tokens = new_tokens,
                )
                st.session_state.csv_filename = uploaded_file.name

            # Display the dataset summary and allow user to modify
            csv_description = st.text_area(label='Be sure to look over and adjust as needed.',value=st.session_state.csv_description)

            # Get or set session state for suggested queries
            if "suggested_queries" not in st.session_state:
                st.session_state.suggested_queries = ""
            
            # Auto-fill button to suggest analyses
            if st.button("Suggest Analyses"):
                st.session_state.suggested_queries = llm(
                    prompt = ("The dataset summary is: " + csv_description),
                    system = 'Provide 3 insightful analysis suggestions based on the dataset summary.',
                    model=model,
                    temperature=temperature,
                    max_new_tokens = new_tokens
                )
            st.write(st.session_state.suggested_queries)

            # Take user input for their analysis/query
            user_input = st.text_area("Your Request")

            # Get or set session state for the generated code
            if "generated_code" not in st.session_state:
                st.session_state.generated_code = ""

            # Generate Python code based on the user's query
            if st.button('Generate Code'):
                with st.spinner('Writing Script...'): 
                    response = llm(
                        prompt = user_input,
                        system = f'''Write a python script to address the user instructions using the following dataset: ##{csv_description}##. Load the data from: ##{st.session_state.csv_filename} ##.''',
                        max_new_tokens = new_tokens,
                        temperature = temperature,
                    )
                    st.session_state.generated_code = response
                    st.write(st.session_state.generated_code)

    if (functionality == 'Graphic Generation'):
        # Add top k to sidebar
        top_k = st.sidebar.number_input("Top k (number of observations to base data summary on)", value=5)
        display_data=st.sidebar.toggle("Display csv?", value=True)

        st.subheader('Please upload your CSV file and enter your request below:')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Display details of the uploaded file
            file_details = {
                "FileName": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": uploaded_file.size
            }
            st.write(file_details)

            # Save the uploaded file to a temporary location and read its content
            file_path = os.path.join("./data/tmp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            df = pd.read_csv(file_path)
            if display_data:
                st.dataframe(df)

            # Get or set session states for CSV description and suggested queries
            if "csv_description" not in st.session_state:
                st.session_state.csv_description = ""
            st.subheader("Enter a brief description of the CSV, including any relevant columns.")

            # Auto-fill button to generate a dataset summary
            if st.button("Auto-Fill"):
                column_data_types = df.dtypes
                columns_info = '\n'.join([f"Column Name: {col}\nData Type: {dtype}" for col, dtype in column_data_types.items()])
                st.session_state.csv_description =  llm(
                    prompt = ("The dataset head is: " + df.head(top_k).to_string() + "\n\n" + columns_info),
                    system = f"""         
                        You are a bot whose purpose is to summarize the data format based upon the column names, data types, and dataset head. The head is only for example for you to understand the structure, do not use these values in the response. Return nothing except the formatting. Write your answer for the dataset descriptions in the exact format:     
                            Column Name: [Column1 Name]
                            Data Type: [Type, e.g., Integer, String, Date]
                            Description: [Brief summary of what this column contains, any special notes about its content.]

                            Column Name: [Column2 Name]
                            Data Type: [Type, e.g., Integer, String, Date]
                            Description: [Brief summary of what this column contains, any special notes about its content.]
                            [... Repeat for all columns ...] """,
                    model=model,
                    temperature=temperature,
                    max_new_tokens = new_tokens
                )
                st.session_state.csv_filename = uploaded_file.name

            # Display the dataset summary and allow user to modify
            csv_description = st.text_area(label='Be sure to look over and adjust as needed.',value=st.session_state.csv_description)

            # Get or set session state for suggested queries
            if "suggested_plots" not in st.session_state:
                st.session_state.suggested_plots = ""
                st.session_state.suggested_plots_code = ""

            # Auto-fill button to suggest analyses
            if st.button("Suggest Plots"):
                st.session_state.suggested_plots = llm(
                    prompt = ("The dataset summary is: " + csv_description),
                    system = 'Provide 2 insightful plot suggestions based on the dataset summary. Treat all objects in the description as strings. Avoid using dates in your suggestions.', 
                    model=model,
                    temperature=temperature,
                    max_new_tokens = new_tokens
                )
                st.session_state.suggested_plots_code = llm(
                    prompt = f"Description: {st.session_state.suggested_plots}",
                    system = f"Based on the description given, return strictly python code to generate plotly graphics named plot1 and plot2. Very importantly, do not return any text besides the script and do not show the plots. Always use markdown formatting. Use the following dataset: ##{csv_description}## and load the data from: ##{file_path}##. Most imoprtantly, ensure that the code can run without error,", 
                    model=model,
                    temperature=temperature,
                    max_new_tokens = new_tokens
                )
            st.write(st.session_state.suggested_plots)
            st.write(st.session_state.suggested_plots_code)
            # Conditional execution to generate plots
            if 'suggested_plots_code' in st.session_state and st.session_state.suggested_plots_code:
                if st.button("Generate Suggested Plots"):
                    cleaned_code = re.findall(r"```python(.*?)```", st.session_state.suggested_plots_code, re.DOTALL)
                    cleaned_code = '\n'.join(cleaned_code).strip()
                    # st.session_state.suggested_plots_code.replace("```python", "").replace("```", "")
                    # Create a dictionary to hold variables defined in the exec() scope
                    local_vars = {}
                    exec(cleaned_code, globals(), local_vars)

                    # Check if plot1 and plot2 are defined in the local_vars dictionary
                    if 'plot1' in local_vars and 'plot2' in local_vars:
                        st.session_state.suggested_plot1 = local_vars['plot1']
                        st.session_state.suggested_plot2 = local_vars['plot2']

            # Conditional display of plots
            if 'suggested_plot1' in st.session_state and st.session_state.suggested_plot1:
                st.plotly_chart(st.session_state.suggested_plot1)
            else:
                st.session_state.suggested_plot1 = None

            if 'suggested_plot2' in st.session_state and st.session_state.suggested_plot2:
                st.plotly_chart(st.session_state.suggested_plot2)
            else:
                st.session_state.suggested_plot2 = None

            if "user_plots" not in st.session_state:
                st.session_state.user_plots = ""
                st.session_state.user_plots_code = ""       

            st.subheader("Enter your custom plot request:")
            st.session_state.user_plot = st.text_area("Your Request")


            # Generate Python code based on the user's query
            if st.button('Generate Code') and st.session_state.user_plot:
                # Take user input for their analysis/query
                with st.spinner('Writing Script...'): 
                    st.session_state.user_plots_code = llm(
                        prompt = f"Description: {st.session_state.user_plot}",
                        system = f"Based on the description given, return strictly python code to generate a plotly graphic named fig with: fig = go.Figure(). Very importantly, do not return any text besides the script. Do not use fig.show() as we are printing the graph later. Always use markdown formatting. Use the following dataset: ##{csv_description}## and load the data from: ##{file_path}##. Most imoprtantly, ensure that the code can run without error,", 
                        model=model,
                        temperature=temperature,
                        max_new_tokens = new_tokens
                    )


            st.write(st.session_state.user_plots_code)
            if 'user_plots_code' in st.session_state and st.session_state.user_plots_code:
                if st.button("Generate User Plot"):
                    cleaned_code = re.findall(r"```python(.*?)```", st.session_state.user_plots_code, re.DOTALL)
                    cleaned_code = '\n'.join(cleaned_code).strip()
                    local_vars = {}
                    exec(cleaned_code, globals(), local_vars)

                    # Check if user_plot is defined in the local_vars dictionary
                    if 'fig' in local_vars:
                        st.session_state.user_generated_plot = local_vars['fig']

            # Conditional display of user plot
            if 'user_generated_plot' in st.session_state and st.session_state.user_generated_plot:
                st.plotly_chart(st.session_state.user_generated_plot)
                # Create an in-memory buffer
                buffer = io.BytesIO()
                st.session_state.user_generated_plot.write_image(file=buffer, format="pdf")
                # Download the pdf from the buffer
                st.download_button(
                    label="Download PDF",
                    data=buffer,
                    file_name="figure.pdf",
                    mime="application/pdf",
                )

                #st.plotly_chart(fig)

                
                # Display the plot
                #st.plotly_chart(st.session_state.user_generated_plot)
            else:
                st.session_state.user_generated_plot = None

    if (functionality == 'Chat'):
        # Get the prompts from the config.ini file
        prompt_options = config.items('prompts') if config.has_section('prompts') else []
        prompt_labels = [option[0] for option in prompt_options]  # Get the prompt names
        prompt_values = [option[1] for option in prompt_options]  # Get the prompt values
        prompt_labels.append('Custom')
        prompt_values.append('custom_prompt') 


        st.sidebar.title("Choose a Prompt")
        # Get the selected prompt label from the radio button
        selected_prompt_label = st.sidebar.radio("Select a prompt:", prompt_labels, format_func=lambda x: x.replace('_', ' ').title())

        # Determine the index of the selected prompt label in the prompt_labels list
        selected_prompt_index = prompt_labels.index(selected_prompt_label)

        if selected_prompt_label == 'Custom':
            # If the user selects "Custom", show a text input field to enter a custom prompt
            custom_prompt = st.sidebar.text_input("Enter your custom prompt:")
            selected_prompt_value = custom_prompt  # Use the custom prompt as the selected value
        else:
            # Use the index to fetch the corresponding value from prompt_values
            selected_prompt_value = prompt_values[selected_prompt_index]
        


        # Initialize conversation in session_state if not present
        if 'conversation' not in st.session_state or not st.session_state['conversation']:
            st.session_state['conversation'] = []
            st.session_state['conversation'].insert(0, {"role": "system", "content": selected_prompt_value})

        # Format and display the conversation
        formatted_messages = format_conversation(st.session_state['conversation'])
        display_chat(formatted_messages)

        if 'reset_input' in st.session_state and st.session_state.reset_input:
            st.session_state.user_input = ""
            st.session_state.reset_input = False

        st.write("")

        with st.form(key='message_form'):
            # Use session state to hold the value of the input box
            user_input = st.text_area("Type your message here...", key="user_input", height=100, value=st.session_state.get('user_input', ''))
            send_pressed = st.form_submit_button("Send")

        if send_pressed and user_input:
            # Update conversation history with user input
            st.session_state['conversation'][0]['content']=selected_prompt_value

            st.session_state['conversation'].append({"role": "user", "content": user_input})

            # Get Chappy response
            Chappy_response = conversation(st.session_state['conversation'], model, temperature, new_tokens)

            # Update conversation history with Chappy response
            st.session_state['conversation'].append({"role": "assistant", "content": Chappy_response})

            # Clear the input box by setting its value in the session state to an empty string
            st.session_state.reset_input = True

            # Rerun the app to update the conversation display
            st.rerun()

        # Add 'Edit Conversation' button in sidebar if in chat mode
        st.sidebar.divider()
        if st.sidebar.button("Edit Conversation", key="edit_conversation_button"):
            st.session_state['edit_mode'] = True

        # Display editable conversation if in edit mode
        if st.session_state.get('edit_mode', False):
            edited_conversation = display_editable_conversation(st.session_state['conversation'])
            if st.button("Save Changes"):
                st.session_state['conversation'] = process_edited_conversation(edited_conversation)
                st.session_state['edit_mode'] = False
                st.rerun()
        
        # Add a button to clear the conversation
        if st.sidebar.button("Clear Conversation", key="clear_conversation_button"):
            st.session_state['conversation'] = []  # Reset the conversation
            st.rerun()  # Rerun the app to update the conversation display
                    
        conversation_text = export_conversation(st.session_state['conversation'])
        # Filename for the download which includes the current date and time for uniqueness
        filename = f"conversation_{pd.Timestamp('now').strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        st.sidebar.download_button(label="Download Conversation",
                                data=conversation_text,
                                file_name=filename,
                                mime='text/plain')
        

        st.sidebar.title("Upload an image:")
        uploaded_image = st.sidebar.file_uploader("👀", type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            st.session_state['uploaded_image'] = uploaded_image
            if st.sidebar.button('Process Uploaded Image'):
                # Convert the uploaded image to base64
                base64_image = get_image_base64(uploaded_image)

                # Call the vision API with the base64 image
                Chappy_response = call_vision_api(base64_image, model="vision", max_tokens=512)

                # Update conversation history with Chappy's response
                st.session_state['conversation'].append({"role": "assistant", "content": Chappy_response})

                st.session_state['uploaded_image'] = None

                # Rerun the app to update the conversation display, but now it won't reprocess the image
                st.rerun()

        st.sidebar.divider()
        st.sidebar.title("Upload past conversation:")
        uploaded_file = st.sidebar.file_uploader("🧠", type=['txt'], key="file_uploader")

        if uploaded_file is not None:
            # Use a session state variable to hold the file temporarily
            st.session_state['uploaded_file'] = uploaded_file
            
            # Provide a button to confirm the processing of the uploaded file
            if st.sidebar.button('Process Uploaded Conversation'):
                # Ensure there's a file to process
                if 'uploaded_file' in st.session_state:
                    uploaded_file = st.session_state['uploaded_file']
                    
                    # Read and process the file
                    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    
                    # Parse the uploaded conversation
                    uploaded_conversation = []
                    for line in string_data.split("*_* "):
                        if line.startswith("You: "):
                            message_content = line.replace("You: ", "", 1)
                            role = 'user'
                        elif line.startswith("Chappy: "):
                            message_content = line.replace("Chappy: ", "", 1)
                            role = 'assistant'  # Adjust based on your application's roles
                        elif line.startswith("System: "):
                            message_content = line.replace("System: ", "", 1)
                            role = 'system'  # Adjust based on your application's roles
                        else:
                            continue  # Skip lines that don't match the expected format
                        
                        uploaded_conversation.append({"role": role, "content": message_content})
                    
                    # Update the current conversation
                    st.session_state['conversation'] = uploaded_conversation
                    st.session_state['uploaded_file'] = None  # Clear the uploaded file after processing
                    
                    # Inform the user of success and refresh the display
                    st.sidebar.success("Uploaded conversation processed successfully.")
                    st.rerun()
                else:
                    st.sidebar.error("No file uploaded.")


    st.sidebar.title("Brought to you by the Tatonetti Lab")
    st.sidebar.empty()
    st.sidebar.image("tlab-logo-large.png", width=100)
    st.sidebar.divider()
    #st.sidebar.image("cedars.png", width=300)

if __name__ == "__main__":
    main()
