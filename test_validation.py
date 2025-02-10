import pytest
import os
import yaml
import hashlib
from utils import PromptProcessor
from collections import defaultdict

@pytest.fixture
def knowledge_base_path():
    return os.path.join(os.path.dirname(__file__), 'knowledge_base.yaml')

@pytest.fixture
def prompt_processor(knowledge_base_path):
    return PromptProcessor(knowledge_base_path)

def test_knowledge_base_exists(knowledge_base_path):
    """Verify knowledge base YAML file exists"""
    assert os.path.exists(knowledge_base_path), "Knowledge base YAML file not found"

def test_prompts_directory_exists():
    """Verify prompts directory exists"""
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    assert os.path.exists(prompts_dir), "Prompts directory not found"

def test_knowledge_base_structure(knowledge_base_path):
    """Verify knowledge base has correct structure and all required models"""
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        kb = yaml.safe_load(f)
    
    required_models = {
        "Anthropic Claude", "OpenAI GPT", "Google Gemini", 
        "Mistral/Mixtral", "Microsoft Copilot", "Perplexity AI",
        "xAI Grok", "Meta Llama", "Other Open-Source Models"
    }
    
    assert set(kb.keys()) == required_models, f"Missing required models. Found: {set(kb.keys())}"

def test_all_markdown_files_exist(knowledge_base_path):
    """Verify all referenced markdown files exist"""
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        kb = yaml.safe_load(f)
    
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    missing_files = []
    
    for model, submodels in kb.items():
        if isinstance(submodels, dict):
            for _, md_file in submodels.items():
                file_path = os.path.join(prompts_dir, md_file)
                if not os.path.exists(file_path):
                    missing_files.append(md_file)
        else:
            file_path = os.path.join(prompts_dir, submodels)
            if not os.path.exists(file_path):
                missing_files.append(submodels)
    
    assert not missing_files, f"Missing markdown files: {missing_files}"

def test_no_duplicate_prompts(knowledge_base_path):
    """Verify no two models share the same prompt content"""
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        kb = yaml.safe_load(f)
    
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    content_hashes = defaultdict(list)
    
    for model, submodels in kb.items():
        if isinstance(submodels, dict):
            for submodel, md_file in submodels.items():
                file_path = os.path.join(prompts_dir, md_file)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        content_hashes[content_hash].append(f"{model}/{submodel}")
        else:
            file_path = os.path.join(prompts_dir, submodels)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    content_hashes[content_hash].append(model)
    
    duplicates = {k: v for k, v in content_hashes.items() if len(v) > 1}
    assert not duplicates, f"Found duplicate prompts: {duplicates}"

def test_prompt_retrieval_performance(prompt_processor):
    """Test prompt retrieval performance meets requirements"""
    import time
    
    # Test 100 concurrent requests
    start_time = time.time()
    for _ in range(100):
        prompt, metrics = prompt_processor.generate_prompt("OpenAI GPT", "GPT-4")
        assert metrics.retrieval_time < 0.2, f"Retrieval time exceeded 200ms: {metrics.retrieval_time * 1000}ms"
        assert prompt is not None, "Prompt retrieval failed"
    
    total_time = time.time() - start_time
    assert total_time < 20, f"Total processing time for 100 requests exceeded 20 seconds: {total_time}s"

def test_submodel_visibility(prompt_processor):
    """Verify submodels are only available after primary model selection"""
    # Before selecting a model, no submodels should be available
    assert not prompt_processor.get_submodels(""), "Submodels should not be available without primary model"
    
    # After selecting OpenAI GPT, specific submodels should be available
    submodels = prompt_processor.get_submodels("OpenAI GPT")
    expected_submodels = {
        "GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo", "ChatGPT iOS", 
        "ChatGPT Android", "OpenAI Assistants API", "OpenAI Deep Research"
    }
    assert set(submodels) == expected_submodels, "Incorrect submodels for OpenAI GPT"

def test_memory_usage():
    """Verify memory usage stays within limits"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    assert memory_usage < 500, f"Memory usage exceeded 500MB: {memory_usage}MB"

def test_prompt_validation(prompt_processor):
    """Test prompt validation and error handling"""
    # Test with valid model/submodel
    prompt, metrics = prompt_processor.generate_prompt("OpenAI GPT", "GPT-4")
    assert prompt is not None, "Failed to retrieve valid prompt"
    assert metrics.has_structure, "Generated prompt lacks required structure"
    
    # Test with invalid model
    prompt, metrics = prompt_processor.generate_prompt("InvalidModel", "InvalidSubmodel")
    assert prompt is None, "Should return None for invalid model"
    
    # Test with valid model but invalid submodel
    prompt, metrics = prompt_processor.generate_prompt("OpenAI GPT", "InvalidSubmodel")
    assert prompt is None, "Should return None for invalid submodel"
