"""LLM-based VibeCoach using open-source models."""

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from configs.settings import config
from src.utils.logger import app_logger

class LLMRecommender:
    """Class for generating activity recommendations using open-source LLMs."""
    
    def __init__(self):
        """Initialize the LLM recommender with the model and tokenizer."""
        try:
            # Get model name from config
            self.model_name = config.llm_config.model_name
            
            # Load model and tokenizer
            app_logger.info(f"Loading model and tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Check if model supports 4-bit quantization
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=getattr(torch, config.llm_config.torch_dtype),
                    device_map=config.llm_config.device,
                    load_in_4bit=True
                )
            except Exception as e:
                app_logger.warning(f"4-bit quantization not supported, falling back to full precision: {str(e)}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=getattr(torch, config.llm_config.torch_dtype),
                    device_map=config.llm_config.device
                )
            
            # Create generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=config.llm_config.device
            )
            
            # Check if model uses special tokens
            self.uses_special_tokens = all(
                token in self.tokenizer.special_tokens_map.values()
                for token in ["<|system|>", "<|user|>", "<|assistant|>"]
            )
            
            app_logger.info("LLMRecommender initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize LLMRecommender: {str(e)}")
            raise

    def _format_prompt(self, emotion: str, num_recommendations: int) -> str:
        """Format the prompt based on model capabilities."""
        if self.uses_special_tokens:
            return f"""<|system|>
You are a helpful activity recommendation assistant. Provide specific and personalized activities based on emotions.

<|user|>
Based on the emotion '{emotion}', suggest {num_recommendations} specific and personalized activities that would be appropriate and helpful. Each activity should be:
1. Specific and actionable
2. Include a brief description of why it's suitable
3. Be realistic and achievable
4. Consider different aspects of well-being (physical, mental, social)

Format each recommendation as a single line.

<|assistant|>"""
        else:
            return f"""You are a helpful activity recommendation assistant. Based on the emotion '{emotion}', suggest {num_recommendations} specific and personalized activities that would be appropriate and helpful.

Each activity should be:
1. Specific and actionable
2. Include a brief description of why it's suitable
3. Be realistic and achievable
4. Consider different aspects of well-being (physical, mental, social)

Format each recommendation as a single line.

Activities:"""

    def _process_response(self, response: str) -> List[str]:
        """Process the model's response into recommendations."""
        # Clean the response
        if self.uses_special_tokens:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Split into lines and clean
        recommendations = [
            line.strip() for line in response.split('\n')
            if line.strip() and not any(
                token in line for token in ["<|system|>", "<|user|>", "<|assistant|>"]
            )
        ]
        
        return recommendations

    def get_recommendations(self, emotion: str, num_recommendations: int = 5) -> List[str]:
        """
        Get activity recommendations based on the detected emotion.
        
        Args:
            emotion: The detected emotion
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of activity recommendations
        """
        try:
            # Construct the prompt
            prompt = self._format_prompt(emotion, num_recommendations)
            
            # Generate recommendations
            response = self.pipe(
                prompt,
                max_new_tokens=config.llm_config.max_tokens,
                temperature=config.llm_config.temperature,
                top_p=0.9,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Process the response
            generated_text = response[0]['generated_text']
            recommendations = self._process_response(generated_text)
            
            # Ensure we have the requested number of recommendations
            recommendations = recommendations[:num_recommendations]
            if len(recommendations) < num_recommendations:
                fallback_recommendations = [
                    "Take a mindful walk in nature - helps reset your emotional state",
                    "Practice deep breathing exercises - promotes relaxation and clarity",
                    "Listen to calming music - helps regulate emotions",
                    "Write in a journal - provides emotional release and perspective",
                    "Engage in light physical activity - boosts mood and energy"
                ]
                recommendations.extend(
                    fallback_recommendations[:num_recommendations - len(recommendations)]
                )
            
            app_logger.info(f"Generated {len(recommendations)} recommendations for emotion: {emotion}")
            return recommendations
            
        except Exception as e:
            app_logger.error(f"Failed to generate recommendations: {str(e)}")
            # Return fallback recommendations
            return [
                "Take a mindful walk in nature - helps reset your emotional state",
                "Practice deep breathing exercises - promotes relaxation and clarity",
                "Listen to calming music - helps regulate emotions",
                "Write in a journal - provides emotional release and perspective",
                "Engage in light physical activity - boosts mood and energy"
            ][:num_recommendations]

    @staticmethod
    def list_available_models() -> List[str]:
        """
        List available open-source models that can be used for recommendations.
        
        Returns:
            List of model names and their descriptions
        """
        return [
            f"{name}: {model}" for name, model in config.llm_config.available_models.items()
        ] 