import json
import yaml
import time
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PromptMetrics:
    retrieval_time: float
    prompt_length: int
    has_structure: bool
    has_actions: bool

class PromptValidator:
    @staticmethod
    def validate_structure(prompt: str) -> bool:
        """Validate if the prompt has proper structure with numbered points."""
        return bool(re.search(r'\d+[.)]', prompt))
    
    @staticmethod
    def validate_content(prompt: str) -> bool:
        """Validate if prompt has sufficient content and no placeholders."""
        min_length = 50
        has_placeholders = bool(re.search(r'\{.*?\}|\[.*?\]', prompt))
        return len(prompt) >= min_length and not has_placeholders

class PromptProcessor:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = knowledge_base_path
        self.prompts_dir = os.path.join(os.path.dirname(knowledge_base_path), 'prompts')
        self.knowledge_base = {}
        self.load_time = 0
        self.reload_knowledge_base()
    
    def reload_knowledge_base(self) -> None:
        """Load the knowledge base with performance tracking."""
        start_time = time.time()
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                if self.knowledge_base_path.endswith('.json'):
                    self.knowledge_base = json.load(f)
                elif self.knowledge_base_path.endswith('.yaml') or self.knowledge_base_path.endswith('.yml'):
                    self.knowledge_base = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported knowledge base format")
            self.load_time = time.time() - start_time
            logger.info(f"Knowledge base loaded in {self.load_time:.3f} seconds")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            self.knowledge_base = {}

    def get_models(self) -> List[str]:
        """Get available primary AI models."""
        return list(self.knowledge_base.keys())

    def get_submodels(self, model: str) -> List[str]:
        """Get available submodels for a given primary model.
           Returns empty list if model not found or has no submodels."""
        if not model:
            return []
        try:
            value = self.knowledge_base.get(model, {})
            if isinstance(value, dict):
                return list(value.keys())
            return []
        except Exception as e:
            logger.warning(f"Error getting submodels for {model}: {str(e)}")
            return []

    def get_markdown_path(self, model: str, submodel: str = "") -> Optional[str]:
        """Get the path to the markdown file for the given model/submodel."""
        try:
            value = self.knowledge_base.get(model)
            if not value:
                return None
            
            if isinstance(value, dict):
                if not submodel or submodel not in value:
                    return None
                md_file = value[submodel]
            else:
                md_file = value
                
            return os.path.join(self.prompts_dir, md_file)
        except Exception as e:
            logger.error(f"Error getting markdown path for {model}/{submodel}: {str(e)}")
            return None

    def sanitize_input(self, text: str) -> str:
        """Sanitize user-provided input."""
        if not text:
            return ""
        # Remove potentially harmful patterns
        text = re.sub(r'[<>]', '', text)
        text = re.sub(r';.*?;', '', text)
        text = re.sub(r'\.\./|~/', '', text)
        return text.strip()

    def generate_prompt(self, model: str, submodel: str = "", 
                       context: Optional[str] = None) -> Tuple[Optional[str], PromptMetrics]:
        """Generate a system prompt by reading the corresponding markdown file."""
        start_time = time.time()
        try:
            md_path = self.get_markdown_path(model, submodel)
            if not md_path or not os.path.exists(md_path):
                logger.error(f"Markdown file not found for {model}/{submodel}")
                return None, PromptMetrics(time.time()-start_time, 0, False, False)

            with open(md_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()

            if context:
                sanitized_context = self.sanitize_input(context)
                prompt_content += f"\n\nAdditional Information:\n{sanitized_context}"

            metrics = PromptMetrics(
                retrieval_time=time.time() - start_time,
                prompt_length=len(prompt_content),
                has_structure=PromptValidator.validate_structure(prompt_content),
                has_actions=bool(re.search(r'\d+[.)]', prompt_content))
            )
            return prompt_content, metrics
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return None, PromptMetrics(time.time()-start_time, 0, False, False)