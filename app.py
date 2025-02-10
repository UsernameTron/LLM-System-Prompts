import streamlit as st
import os
import time
from utils import PromptProcessor, PromptMetrics
import pandas as pd

# Initialize session state
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []

# Initialize PromptProcessor with caching
@st.cache_data()
def get_prompt_processor():
    return PromptProcessor(os.path.join(os.path.dirname(__file__), 'knowledge_base.yaml'))

def display_metrics(metrics: PromptMetrics):
    """Display prompt generation metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Response Time', f'{metrics.retrieval_time*1000:.1f}ms')
    with col2:
        st.metric('Prompt Length', metrics.prompt_length)
    with col3:
        st.metric('Has Structure', '✅' if metrics.has_structure else '❌')
    with col4:
        st.metric('Has Actions', '✅' if metrics.has_actions else '❌')

def update_metrics_history(metrics: PromptMetrics, model: str, submodel: str):
    """Update metrics history for monitoring."""
    st.session_state.metrics_history.append({
        'timestamp': time.time(),
        'model': model,
        'submodel': submodel,
        'response_time_ms': metrics.retrieval_time * 1000,
        'prompt_length': metrics.prompt_length,
        'has_structure': metrics.has_structure,
        'has_actions': metrics.has_actions
    })

# Main UI
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title('AI System Prompt Generator')
st.markdown('''
Generate optimized system prompts for various AI models.
Choose your model and customize the prompt with additional context.
''')

# Initialize prompt processor
processor = get_prompt_processor()

# Check if we're on mobile
is_mobile = st.query_params.get('embed', '') == 'true'

# Model Configuration - Show in main content on mobile, sidebar on desktop
config_container = st if is_mobile else st.sidebar

with config_container:
    st.header('Model Configuration')
    
    # Model Selection with increased height for mobile touch
    models = processor.get_models()
    selected_model = st.selectbox(
        'Select AI Model',
        models,
        key='model_select',
        help='Tap to select an AI model',
    )
    
    # Dynamic Submodel Selection with increased height for mobile touch
    submodels = processor.get_submodels(selected_model)
    selected_submodel = st.selectbox(
        'Select Submodel',
        submodels,
        key='submodel_select',
        help='Tap to select a submodel (if available)',
    ) if submodels else None
    
    # Display metrics history
    if st.session_state.metrics_history:
        st.header('Performance Metrics')
        df = pd.DataFrame(st.session_state.metrics_history)
        avg_response = df['response_time_ms'].mean()
        st.metric('Avg Response Time', f'{avg_response:.1f}ms')
        
        if len(df) > 1:
            st.line_chart(df.set_index('timestamp')['response_time_ms'])

# Main content area - Adjust layout based on device
if is_mobile:
    # Single column layout for mobile
    st.header('Additional Context')
    context = st.text_area(
        'Add specific details or requirements (optional):',
        help='Provide additional context or specific requirements for your prompt.',
        max_chars=1000,
        height=150  # Smaller height on mobile
    )
    
    st.header('Selected Configuration')
    st.markdown(f'''
    - **Model**: {selected_model}
    - **Submodel**: {selected_submodel if selected_submodel else 'N/A'}
    ''')
else:
    # Two column layout for desktop
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header('Additional Context')
        context = st.text_area(
            'Add specific details or requirements (optional):',
            help='Provide additional context or specific requirements for your prompt.',
            max_chars=1000
        )
    
    with col2:
        st.header('Configuration')
        st.markdown(f'''
        - **Model**: {selected_model}
        - **Submodel**: {selected_submodel if selected_submodel else 'N/A'}
        ''')

# Generate prompt button - Make it more prominent on mobile
button_style = """<style>
    .stButton>button {
        width: 100%;
        padding: 1rem;
        font-size: 1.2rem;
    }
</style>"""
st.markdown(button_style, unsafe_allow_html=True)

if st.button('Generate System Prompt', type='primary', use_container_width=True):
    with st.spinner('Generating optimized system prompt...'):
        # Sanitize user input
        sanitized_context = processor.sanitize_input(context) if context else None
        
        # Generate prompt with metrics
        prompt, metrics = processor.generate_prompt(
            selected_model, 
            selected_submodel, 
            sanitized_context
        )
        
        if prompt:
            # Update and display metrics
            update_metrics_history(metrics, selected_model, selected_submodel)
            
            # Display metrics in a more mobile-friendly way
            if is_mobile:
                st.header('Performance Metrics')
                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    st.metric('Response Time', f'{metrics.retrieval_time*1000:.1f}ms')
                    st.metric('Has Structure', '✅' if metrics.has_structure else '❌')
                with metrics_cols[1]:
                    st.metric('Prompt Length', metrics.prompt_length)
                    st.metric('Has Actions', '✅' if metrics.has_actions else '❌')
            else:
                display_metrics(metrics)
            
            # Display the generated prompt
            st.header('Generated System Prompt')
            
            # Adjust text area height based on device
            prompt_height = 150 if is_mobile else 300
            st.text_area('Prompt', prompt, height=prompt_height)
            
            # Add copy button with mobile-friendly styling
            st.code(prompt, language='text')
            
            # Warning if prompt doesn't meet quality standards
            if not metrics.has_structure or not metrics.has_actions:
                st.warning('⚠️ The prompt may need improvement. Try adding more context.')
        else:
            st.error('Could not generate prompt. Please try different options.')
