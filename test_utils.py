import pytest
import os
from utils import PromptProcessor, PromptValidator
import time

@pytest.fixture
def prompt_processor():
    return PromptProcessor(os.path.join(os.path.dirname(__file__), 'system_prompts.json'))

def test_prompt_validator_structure():
    """Test prompt structure validation."""
    valid_prompts = [
        "1) First point\n2) Second point",
        "1. First point\n2. Second point",
        "Task: 1) Do this 2) Do that"
    ]
    invalid_prompts = [
        "Just some text without structure",
        "No numbered points here",
        ""
    ]
    
    for prompt in valid_prompts:
        assert PromptValidator.validate_structure(prompt)
    
    for prompt in invalid_prompts:
        assert not PromptValidator.validate_structure(prompt)

def test_prompt_validator_content():
    """Test prompt content validation."""
    valid_prompts = [
        "This is a complete prompt with sufficient length and no placeholders.",
        "1) First point with good content\n2) Second point with details\n3) Third point explaining things"
    ]
    invalid_prompts = [
        "Too short",
        "Contains a {placeholder} which is not allowed",
        "Has a [template] that should be replaced"
    ]
    
    for prompt in valid_prompts:
        assert PromptValidator.validate_content(prompt)
    
    for prompt in invalid_prompts:
        assert not PromptValidator.validate_content(prompt)

def test_prompt_processor_initialization(prompt_processor):
    """Test PromptProcessor initialization and knowledge base loading."""
    assert prompt_processor.knowledge_base is not None
    assert isinstance(prompt_processor.knowledge_base, dict)
    assert len(prompt_processor.knowledge_base) > 0
    assert prompt_processor.load_time > 0

def test_model_retrieval(prompt_processor):
    """Test model and submodel retrieval."""
    models = prompt_processor.get_models()
    assert len(models) > 0
    assert "ChatGPT" in models
    
    # Test submodel retrieval
    submodels = prompt_processor.get_submodels("ChatGPT")
    assert len(submodels) > 0
    assert "GPT-3.5" in submodels
    assert "GPT-4" in submodels

def test_use_case_retrieval(prompt_processor):
    """Test use case retrieval."""
    use_cases = prompt_processor.get_use_cases("ChatGPT", "GPT-4")
    assert len(use_cases) > 0
    assert "default" in use_cases
    assert "resume_optimization" in use_cases

def test_prompt_generation(prompt_processor):
    """Test prompt generation with metrics."""
    # Test successful prompt generation
    prompt, metrics = prompt_processor.generate_prompt(
        "ChatGPT", 
        "GPT-4", 
        "resume_optimization",
        "Testing context"
    )
    
    assert prompt is not None
    assert len(prompt) > 0
    assert metrics.retrieval_time < 0.2  # 200ms requirement
    assert metrics.prompt_length > 0
    assert metrics.has_structure
    assert metrics.has_actions
    
    # Test with invalid model/submodel
    prompt, metrics = prompt_processor.generate_prompt(
        "InvalidModel",
        "InvalidSubmodel",
        "InvalidUseCase"
    )
    assert prompt is None

def test_input_sanitization(prompt_processor):
    """Test input sanitization."""
    unsafe_inputs = [
        "Unsafe <script>alert('xss')</script>",
        "Injection attempt; DROP TABLE prompts;",
        "Path traversal ../../../etc/passwd"
    ]
    
    for unsafe_input in unsafe_inputs:
        sanitized = prompt_processor.sanitize_input(unsafe_input)
        assert "<script>" not in sanitized
        assert ";" not in sanitized
        assert "../" not in sanitized
        assert len(sanitized) > 0

def test_performance_requirements(prompt_processor):
    """Test performance requirements."""
    start_time = time.time()
    
    # Test 100 concurrent prompt generations
    for _ in range(100):
        prompt, metrics = prompt_processor.generate_prompt(
            "ChatGPT",
            "GPT-4",
            "resume_optimization"
        )
        assert metrics.retrieval_time < 0.2  # 200ms per request
        assert prompt is not None
    
    total_time = time.time() - start_time
    assert total_time < 20  # Reasonable time for 100 requests
